//! JIT compiler implementation.

use crate::{
    Backend, Builder, Bytecode, EvmContext, EvmStack, Inst, InstData, InstFlags, IntCC, JitEvmFn,
    Result, I256_MIN, TEST_SUSPEND,
};
use revm_interpreter::{opcode as op, Contract, Gas, InstructionResult};
use revm_jit_backend::{
    Attribute, BackendTypes, FunctionAttributeLocation, OptimizationLevel, TypeMethods,
};
use revm_jit_callbacks::{Callback, Callbacks};
use revm_primitives::{BlockEnv, CfgEnv, Env, SpecId, TxEnv, U256};
use std::{
    fmt::Write as _,
    io::Write,
    mem,
    path::{Path, PathBuf},
    sync::atomic::AtomicPtr,
};

const STACK_CAP: usize = 1024;
// const WORD_SIZE: usize = 32;

// TODO: ~~Cannot find function if `compile` is called a second time.~~
// `get_function` finalizes the module, making it impossible to add more functions.
// TODO: Refactor the API to allow for multiple functions to be compiled.

// TODO: Add `nuw`/`nsw` flags to stack length arithmetic.

// TODO: Somehow have a config to tell the backend to assume that stack stores are unobservable,
// making it eliminate redundant stores for values outside the stack length when optimized away.
// E.g. `PUSH0 POP` gets fully optimized away, but the `store i256 0, ptr %stack` will still get
// emitted.
// Use this when `stack` is passed in arguments.

// TODO: Test on big-endian hardware.
// It probably doesn't work when loading Rust U256 into native endianness.

/// JIT compiler for EVM bytecode.
#[allow(missing_debug_implementations)]
pub struct JitEvm<B: Backend> {
    backend: B,
    out_dir: Option<PathBuf>,
    config: FcxConfig,
    function_counter: usize,
    callbacks: Callbacks<B>,
    dump_assembly: bool,
    dump_unopt_assembly: bool,
}

impl<B: Backend + Default> Default for JitEvm<B> {
    fn default() -> Self {
        Self::new(B::default())
    }
}

impl<B: Backend> JitEvm<B> {
    /// Creates a new instance of the JIT compiler with the given backend.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            out_dir: None,
            config: FcxConfig::default(),
            function_counter: 0,
            callbacks: Callbacks::new(),
            dump_assembly: true,
            dump_unopt_assembly: false,
        }
    }

    /// Dumps the IR and potential to the given directory after compilation.
    ///
    /// Disables dumping if `output_dir` is `None`.
    ///
    /// Creates a subdirectory with the name of the backend in the given directory.
    pub fn set_dump_to(&mut self, output_dir: Option<PathBuf>) {
        self.backend.set_is_dumping(output_dir.is_some());
        self.config.comments_enabled = output_dir.is_some();
        self.out_dir = output_dir;
    }

    /// Dumps assembly to the output directory.
    ///
    /// This can be quite slow.
    ///
    /// Defaults to `true`.
    pub fn dump_assembly(&mut self, yes: bool) {
        self.dump_assembly = yes;
    }

    /// Dumps the unoptimized assembly to the output directory.
    ///
    /// This can be quite slow.
    ///
    /// Defaults to `false`.
    pub fn dump_unopt_assembly(&mut self, yes: bool) {
        self.dump_unopt_assembly = yes;
    }

    /// Returns the optimization level.
    pub fn opt_level(&self) -> OptimizationLevel {
        self.backend.opt_level()
    }

    /// Sets the optimization level.
    ///
    /// Note that some backends may not support setting the optimization level after initialization.
    ///
    /// Defaults to the backend's initial optimization level.
    pub fn set_opt_level(&mut self, level: OptimizationLevel) {
        self.backend.set_opt_level(level);
    }

    /// Sets whether to enable debug assertions.
    ///
    /// These are useful for debugging, but they do a moderate performance penalty due to the
    /// insertion of extra checks and removal of certain assumptions.
    ///
    /// Defaults to `cfg!(debug_assertions)`.
    pub fn set_debug_assertions(&mut self, yes: bool) {
        self.backend.set_debug_assertions(yes);
        self.config.debug_assertions = yes;
    }

    /// Sets whether to enable frame pointers.
    ///
    /// This is useful for profiling and debugging, but it incurs a very slight performance penalty.
    ///
    /// Defaults to `cfg!(debug_assertions)`.
    pub fn set_frame_pointers(&mut self, yes: bool) {
        self.config.frame_pointers = yes;
    }

    /// Sets whether to allocate the stack locally.
    ///
    /// If this is set to `true`, the stack pointer argument will be ignored and the stack will be
    /// allocated in the function.
    ///
    /// This setting will fail at runtime if the bytecode suspends execution, as it cannot be
    /// restored afterwards.
    ///
    /// Defaults to `false`.
    pub fn set_local_stack(&mut self, yes: bool) {
        self.config.local_stack = yes;
    }

    /// Sets whether to treat the stack length as observable outside the function.
    ///
    /// This also implies that the length is loaded in the beginning of the function, meaning
    /// that a function can be executed with an initial stack.
    ///
    /// If this is set to `true`, the stack length must be passed in the arguments.
    ///
    /// This is useful to inspect the stack length after the function has been executed, but it does
    /// incur a performance penalty as the length will be stored at all return sites.
    ///
    /// Defaults to `false`.
    pub fn set_inspect_stack_length(&mut self, yes: bool) {
        self.config.inspect_stack_length = yes;
    }

    /// Sets whether to disable gas accounting.
    ///
    /// Greatly improves compilation speed and performance, at the cost of not being able to check
    /// for gas exhaustion.
    ///
    /// Use with care.
    ///
    /// Defaults to `false`.
    pub fn set_disable_gas(&mut self, disable_gas: bool) {
        self.config.gas_disabled = disable_gas;
    }

    /// Compiles the given EVM bytecode into a JIT function.
    pub fn compile(
        &mut self,
        name: Option<&str>,
        bytecode: &[u8],
        spec_id: SpecId,
    ) -> Result<JitEvmFn> {
        let bytecode = debug_time!("parse", || self.parse_bytecode(bytecode, spec_id))?;
        debug_time!("compile", || self.compile_bytecode(name, &bytecode))
    }

    /// Frees all functions compiled by this JIT compiler.
    ///
    /// # Safety
    ///
    /// Because this function invalidates any pointers retrived from the corresponding module, it
    /// should only be used when none of the functions from that module are currently executing and
    /// none of the `fn` pointers are called afterwards.
    pub unsafe fn free_all_functions(&mut self) -> Result<()> {
        self.callbacks.clear();
        self.function_counter = 0;
        self.backend.free_all_functions()
    }

    fn parse_bytecode<'a>(&mut self, bytecode: &'a [u8], spec_id: SpecId) -> Result<Bytecode<'a>> {
        let mut bytecode = trace_time!("new bytecode", || Bytecode::new(bytecode, spec_id));
        trace_time!("analyze", || bytecode.analyze())?;
        Ok(bytecode)
    }

    fn compile_bytecode(
        &mut self,
        name: Option<&str>,
        bytecode: &Bytecode<'_>,
    ) -> Result<JitEvmFn> {
        let name = name.unwrap_or("evm_bytecode");
        let mname = &self.mangle_name(name, bytecode.spec_id)[..];

        let mut dump_dir = PathBuf::new();
        if let Some(out_dir) = &self.out_dir {
            dump_dir.push(out_dir);
            dump_dir.push(name);
            dump_dir.push(format!("{:?}", bytecode.spec_id));
            std::fs::create_dir_all(&dump_dir)?;
        }
        let dumping = self.out_dir.is_some();

        if dumping {
            trace_time!("dump bytecode", || Self::dump_bytecode(&dump_dir, bytecode))?;
        }

        let bcx = trace_time!("make builder", || Self::make_builder_function(
            &mut self.backend,
            &self.config,
            mname,
        ))?;

        trace_time!("translate", || FunctionCx::translate(
            bcx,
            &self.config,
            &mut self.callbacks,
            bytecode,
        ))?;

        let verify = |b: &mut B| trace_time!("verify", || b.verify_function(mname));
        if dumping {
            trace_time!("dump unopt IR", || {
                let path = dump_dir.join("unopt").with_extension(self.backend.ir_extension());
                self.backend.dump_ir(&path)
            })?;

            // Dump IR before verifying for better debugging.
            verify(&mut self.backend)?;

            if self.dump_assembly && self.dump_unopt_assembly {
                trace_time!("dump unopt disasm", || {
                    let path = dump_dir.join("unopt.s");
                    self.backend.dump_disasm(&path)
                })?;
            }
        } else {
            verify(&mut self.backend)?;
        }

        trace_time!("optimize", || self.backend.optimize_function(mname))?;

        if dumping {
            trace_time!("dump opt IR", || {
                let path = dump_dir.join("opt").with_extension(self.backend.ir_extension());
                self.backend.dump_ir(&path)
            })?;

            if self.dump_assembly {
                trace_time!("dump opt disasm", || {
                    let path = dump_dir.join("opt.s");
                    self.backend.dump_disasm(&path)
                })?;
            }
        }

        let addr = trace_time!("finalize", || self.backend.get_function(mname))?;
        Ok(JitEvmFn::new(unsafe { std::mem::transmute(addr) }))
    }

    fn make_builder_function<'a>(
        backend: &'a mut B,
        config: &FcxConfig,
        name: &str,
    ) -> Result<B::Builder<'a>> {
        fn align_size<T>(i: usize) -> (usize, usize, usize) {
            (i, mem::align_of::<T>(), mem::size_of::<T>())
        }

        let i8 = backend.type_int(8);
        let ptr = backend.type_ptr();
        let (ret, params, param_names, ptr_attrs) = (
            Some(i8),
            &[ptr, ptr, ptr, ptr, ptr, ptr],
            &[
                "arg.gas.addr",
                "arg.stack.addr",
                "arg.stack_len.addr",
                "arg.env.addr",
                "arg.contract.addr",
                "arg.ecx.addr",
            ],
            &[
                align_size::<Gas>(0),
                align_size::<EvmStack>(1),
                align_size::<usize>(2),
                align_size::<Env>(3),
                align_size::<Contract>(4),
                align_size::<EvmContext<'_>>(5),
            ],
        );
        debug_assert_eq!(params.len(), param_names.len());
        let linkage = revm_jit_backend::Linkage::Public;
        let mut bcx = backend.build_function(name, ret, params, param_names, linkage)?;

        // Function attributes.
        let function_attributes = [
            Attribute::WillReturn,      // Always returns.
            Attribute::NoFree,          // No memory deallocation.
            Attribute::NoSync,          // No thread synchronization.
            Attribute::NativeTargetCpu, // Optimization.
            Attribute::Speculatable,    // No undefined behavior.
            Attribute::NoRecurse,       // Revm is not recursive.
        ]
        .into_iter()
        .chain(config.frame_pointers.then_some(Attribute::AllFramePointers))
        // We can unwind in panics, which are present only in debug assertions.
        .chain((!config.debug_assertions).then_some(Attribute::NoUnwind));
        for attr in function_attributes {
            bcx.add_function_attribute(None, attr, FunctionAttributeLocation::Function);
        }

        // Pointer argument attributes.
        if !config.debug_assertions {
            for &(i, align, dereferenceable) in ptr_attrs {
                let attrs = [
                    Attribute::NoCapture,
                    Attribute::NoUndef,
                    Attribute::Align(align as u64),
                    Attribute::Dereferenceable(dereferenceable as u64),
                ]
                .into_iter()
                // `Gas` is aliased in `EvmContext`.
                .chain((i != 0).then_some(Attribute::NoAlias));
                for attr in attrs {
                    let loc = FunctionAttributeLocation::Param(i as _);
                    bcx.add_function_attribute(None, attr, loc);
                }
            }
        }

        Ok(bcx)
    }

    fn dump_bytecode(dump_dir: &Path, bytecode: &Bytecode<'_>) -> Result<()> {
        std::fs::create_dir_all(dump_dir)?;

        std::fs::write(dump_dir.join("evm.bin"), bytecode.code)?;

        std::fs::write(dump_dir.join("evm.hex"), revm_primitives::hex::encode(bytecode.code))?;

        let file = std::fs::File::create(dump_dir.join("evm.txt"))?;
        let mut file = std::io::BufWriter::new(file);

        let header = format!("{:^6} | {:^6} | {:^80} | {}", "ic", "pc", "opcode", "instruction");
        writeln!(file, "{header}")?;
        writeln!(file, "{}", "-".repeat(header.len()))?;
        for (inst, (pc, opcode)) in bytecode.opcodes().with_pc().enumerate() {
            let data = bytecode.inst(inst);
            let opcode = opcode.to_string();
            writeln!(file, "{inst:^6} | {pc:^6} | {opcode:<80} | {data:?}")?;
        }

        file.flush()?;

        Ok(())
    }

    fn mangle_name(&mut self, base: &str, spec_id: SpecId) -> String {
        let name = format!("{base}_{spec_id:?}_{}", self.function_counter);
        self.function_counter += 1;
        name
    }
}

/// A list of incoming values for a block.
type Incoming<B> = Vec<(<B as BackendTypes>::Value, <B as BackendTypes>::BasicBlock)>;

#[derive(Clone, Debug)]
struct FcxConfig {
    comments_enabled: bool,
    debug_assertions: bool,
    frame_pointers: bool,

    local_stack: bool,
    inspect_stack_length: bool,
    gas_disabled: bool,
}

impl Default for FcxConfig {
    fn default() -> Self {
        Self {
            debug_assertions: cfg!(debug_assertions),
            comments_enabled: false,
            frame_pointers: cfg!(debug_assertions),
            local_stack: false,
            inspect_stack_length: false,
            gas_disabled: false,
        }
    }
}

struct FunctionCx<'a, B: Backend> {
    // Configuration.
    comments_enabled: bool,
    disable_gas: bool,

    /// The backend's function builder.
    bcx: B::Builder<'a>,

    // Common types.
    isize_type: B::Type,
    word_type: B::Type,
    address_type: B::Type,
    i8_type: B::Type,

    // Locals.
    /// The stack length. Either passed in the arguments as a pointer or allocated locally.
    stack_len: Pointer<B>,
    /// The stack value. Constant throughout the function, either passed in the arguments as a
    /// pointer or allocated locally.
    stack: Pointer<B>,
    /// The amount of gas remaining. `i64`. See [`Gas`].
    gas_remaining: Pointer<B>,
    /// The amount of gas remaining, without accounting for memory expansion. `i64`. See [`Gas`].
    gas_remaining_nomem: Pointer<B>,
    /// The environment. Constant throughout the function.
    env: B::Value,
    /// The contract. Constant throughout the function.
    contract: B::Value,
    /// The EVM context. Opaque pointer, only passed to callbacks.
    ecx: B::Value,
    len_before: B::Value,
    len_offset: i8,

    /// The bytecode being translated.
    bytecode: &'a Bytecode<'a>,
    /// All entry blocks for each instruction.
    inst_entries: Vec<B::BasicBlock>,
    /// The current instruction being translated.
    current_inst: Inst,

    /// `dynamic_jump_table` incoming values.
    incoming_dynamic_jumps: Incoming<B>,
    /// The dynamic jump table block where all dynamic jumps branch to.
    dynamic_jump_table: B::BasicBlock,

    /// `return_block` incoming values.
    incoming_returns: Incoming<B>,
    /// The return block that all return instructions branch to.
    return_block: B::BasicBlock,

    /// `resume_block` switch values.
    resume_blocks: Incoming<B>,
    /// `suspend_block` incoming values.
    suspend_blocks: Incoming<B>,
    /// The suspend block that all suspend instructions branch to.
    suspend_block: B::BasicBlock,

    /// Callbacks.
    callbacks: &'a mut Callbacks<B>,
}

impl<'a, B: Backend> FunctionCx<'a, B> {
    /// Translates an EVM bytecode into a native function.
    ///
    /// Example pseudo-code:
    ///
    /// ```ignore (pseudo-code)
    /// // `cfg(will_suspend) = bytecode.will_suspend()`: `true` if it contains a
    /// // `*CALL*` or `CREATE*` instruction.
    /// fn evm_bytecode(args: ...) {
    ///     setup_locals();
    ///
    ///     #[cfg(debug_assertions)]
    ///     if args.<ptr>.is_null() { panic!("...") };
    ///
    ///     load_arguments();
    ///
    ///     #[cfg(will_suspend)]
    ///     resume: {
    ///         goto match ecx.resume_at {
    ///             0 => inst0,
    ///             1 => first_call_or_create_inst + 1, // + 1 as in the block after.
    ///             2 => second_call_or_create_inst + 1,
    ///             ... => ...,
    ///             _ => unreachable, // Assumed to be valid.
    ///         };
    ///     };
    ///     
    ///     op.inst0: { /* ... */ };
    ///     op.inst1: { /* ... */ };
    ///     // ...
    ///     #[cfg(will_suspend)]
    ///     first_call_or_create_inst: {
    ///          // ...
    ///          goto suspend(1);
    ///     };
    ///     // ...
    ///     // There will always be at least one diverging instruction.
    ///     op.stop: {
    ///         goto return(InstructionResult::Stop);
    ///     };
    ///
    ///     #[cfg(will_suspend)]
    ///     suspend(resume_at: u32): {
    ///         ecx.resume_at = resume_at;
    ///         goto return(InstructionResult::CallOrCreate);
    ///     };
    ///
    ///     // All paths lead to here.
    ///     return(ir: InstructionResult): {
    ///         #[cfg(inspect_stack_length)]
    ///         *args.stack_len = stack_len;
    ///         return ir;
    ///     }
    /// }
    /// ```
    #[allow(rustdoc::invalid_rust_codeblocks)] // Syntax highlighting.
    fn translate(
        mut bcx: B::Builder<'a>,
        config: &FcxConfig,
        callbacks: &'a mut Callbacks<B>,
        bytecode: &'a Bytecode<'a>,
    ) -> Result<()> {
        // Get common types.
        let isize_type = bcx.type_ptr_sized_int();
        let i8_type = bcx.type_int(8);
        let i64_type = bcx.type_int(64);
        let address_type = bcx.type_int(160);
        let word_type = bcx.type_int(256);

        // Set up entry block.
        let gas_ptr = bcx.fn_param(0);
        let gas_remaining = {
            let offset = bcx.iconst(isize_type, mem::offset_of!(pf::Gas, remaining) as i64);
            let name = "gas.remaining.addr";
            let base = PointerBase::Address(bcx.gep(i8_type, gas_ptr, &[offset], name));
            Pointer { ty: isize_type, base }
        };
        let gas_remaining_nomem = {
            let offset = bcx.iconst(isize_type, mem::offset_of!(pf::Gas, remaining_nomem) as i64);
            let name = "gas.remaining_nomem.addr";
            let base = PointerBase::Address(bcx.gep(i8_type, gas_ptr, &[offset], name));
            Pointer { ty: i64_type, base }
        };

        let sp_arg = bcx.fn_param(1);
        let stack = {
            let base = if !config.local_stack {
                PointerBase::Address(sp_arg)
            } else {
                let stack_type = bcx.type_array(word_type, STACK_CAP as _);
                let stack_slot = bcx.new_stack_slot(stack_type, "stack.addr");
                PointerBase::StackSlot(stack_slot)
            };
            Pointer { ty: word_type, base }
        };

        let stack_len_arg = bcx.fn_param(2);
        let stack_len = {
            // This is initialized later in `post_entry_block`.
            let base = PointerBase::StackSlot(bcx.new_stack_slot(isize_type, "len.addr"));
            Pointer { ty: isize_type, base }
        };

        let env = bcx.fn_param(3);
        let contract = bcx.fn_param(4);
        let ecx = bcx.fn_param(5);

        // Create all instruction entry blocks.
        let dynamic_jump_fail = bcx.create_block("dynamic_jump_fail");
        let inst_entries: Vec<_> = bytecode
            .iter_all_insts()
            .map(|(i, data)| {
                if data.is_dead_code() {
                    dynamic_jump_fail
                } else {
                    bcx.create_block(&op_block_name_with(i, data, ""))
                }
            })
            .collect();
        assert!(!inst_entries.is_empty(), "translating empty bytecode");

        let dynamic_jump_table = bcx.create_block("dynamic_jump_table");
        let suspend_block = bcx.create_block("suspend");
        let return_block = bcx.create_block("return");

        let mut fx = FunctionCx {
            comments_enabled: config.comments_enabled,
            disable_gas: config.gas_disabled,

            isize_type,
            address_type,
            word_type,
            i8_type,
            stack_len,
            stack,
            gas_remaining,
            gas_remaining_nomem,
            env,
            contract,
            ecx,
            len_before: bcx.iconst(isize_type, 0),
            len_offset: 0,
            bcx,

            bytecode,
            inst_entries,
            current_inst: usize::MAX,

            incoming_dynamic_jumps: Vec::new(),
            dynamic_jump_table,
            resume_blocks: Vec::new(),
            suspend_blocks: Vec::new(),
            suspend_block,
            incoming_returns: Vec::new(),
            return_block,

            callbacks,
        };

        // We store the stack length if requested or necessary due to the bytecode.
        let store_stack_length = config.inspect_stack_length || bytecode.will_suspend();

        // Add debug assertions for the parameters.
        if config.debug_assertions {
            fx.pointer_panic_with_bool(
                !config.gas_disabled,
                gas_ptr,
                "gas pointer",
                "gas metering is enabled",
            );
            fx.pointer_panic_with_bool(
                !config.local_stack,
                sp_arg,
                "stack pointer",
                "local stack is disabled",
            );
            fx.pointer_panic_with_bool(
                store_stack_length,
                stack_len_arg,
                "stack length pointer",
                if config.inspect_stack_length {
                    "stack length inspection is enabled"
                } else {
                    "bytecode suspends execution"
                },
            );
            fx.pointer_panic_with_bool(true, env, "env pointer", "");
            fx.pointer_panic_with_bool(true, contract, "contract pointer", "");
            fx.pointer_panic_with_bool(true, ecx, "EVM context pointer", "");
        }

        // The bytecode is guaranteed to have at least one instruction.
        let first_inst_block = fx.inst_entries[0];
        let current_block = fx.current_block();
        let post_entry_block = fx.bcx.create_block_after(current_block, "entry.post");
        let resume_block = fx.bcx.create_block_after(post_entry_block, "resume");
        fx.bcx.br(post_entry_block);
        // Important: set the first resume target to be the start of the instructions.
        if fx.bytecode.will_suspend() {
            fx.add_resume_at(first_inst_block);
        }

        // Translate individual instructions into their respective blocks.
        for (inst, _) in bytecode.iter_insts() {
            fx.translate_inst(inst)?;
        }

        // Finalize the dynamic jump table.
        let i32_type = fx.bcx.type_int(32);
        if bytecode.has_dynamic_jumps() {
            let default = dynamic_jump_fail;
            fx.bcx.switch_to_block(default);
            fx.build_return_imm(InstructionResult::InvalidJump);

            fx.bcx.switch_to_block(fx.dynamic_jump_table);
            let index = fx.bcx.phi(fx.word_type, &fx.incoming_dynamic_jumps);
            // TODO: Reduce isn't really right.
            let index = fx.bcx.ireduce(i32_type, index);
            let targets = fx
                .bytecode
                .iter_insts()
                .filter(|(_, data)| data.opcode == op::JUMPDEST)
                .map(|(inst, data)| {
                    let pc = fx.bcx.iconst(i32_type, data.pc as i64);
                    (pc, fx.inst_entries[inst])
                })
                .collect::<Vec<_>>();
            fx.bcx.switch(index, default, &targets);
        } else {
            // No dynamic jumps.
            debug_assert!(fx.incoming_dynamic_jumps.is_empty());
            fx.bcx.switch_to_block(fx.dynamic_jump_table);
            fx.bcx.unreachable();
            fx.bcx.switch_to_block(dynamic_jump_fail);
            fx.bcx.unreachable();
        }

        // Finalize the suspend and resume blocks. Must come before the return block.
        // Also here is where the stack length is initialized.
        let load_len_at_start = |fx: &mut Self| {
            // Loaded from args only for the config.
            if config.inspect_stack_length {
                let stack_len = fx.bcx.load(fx.isize_type, stack_len_arg, "stack_len");
                fx.stack_len.store(&mut fx.bcx, stack_len);
            } else {
                fx.stack_len.store_imm(&mut fx.bcx, 0);
            }
        };
        if bytecode.will_suspend() {
            let get_ecx_resume_at = |fx: &mut Self| {
                let offset =
                    fx.bcx.iconst(fx.isize_type, mem::offset_of!(EvmContext<'_>, resume_at) as i64);
                let name = "ecx.resume_at.addr";
                fx.bcx.gep(fx.i8_type, fx.ecx, &[offset], name)
            };

            // Resume block: load the `resume_at` value and switch to the corresponding block.
            // Invalid values are treated as unreachable.
            {
                let default = fx.bcx.create_block_after(resume_block, "resume_invalid");
                fx.bcx.switch_to_block(default);
                if config.debug_assertions {
                    fx.call_panic("invalid resume value");
                } else {
                    fx.bcx.unreachable();
                }

                // Special-case the zero block to load 0 into the length if possible.
                let resume_is_zero_block =
                    fx.bcx.create_block_after(resume_block, "resume_is_zero");

                fx.bcx.switch_to_block(post_entry_block);
                let resume_at = get_ecx_resume_at(&mut fx);
                let resume_at = fx.bcx.load(i32_type, resume_at, "resume_at");
                let is_resume_zero = fx.bcx.icmp_imm(IntCC::Equal, resume_at, 0);
                fx.bcx.brif(is_resume_zero, resume_is_zero_block, resume_block);

                fx.bcx.switch_to_block(resume_is_zero_block);
                load_len_at_start(&mut fx);
                fx.bcx.br(first_inst_block);

                // Dispatch to the resume block.
                fx.bcx.switch_to_block(resume_block);
                let stack_len = fx.bcx.load(fx.isize_type, stack_len_arg, "stack_len");
                fx.stack_len.store(&mut fx.bcx, stack_len);
                fx.resume_blocks[0].1 = default; // Zero case is handled above.
                fx.bcx.switch(resume_at, default, &fx.resume_blocks);
            }

            // Suspend block: store the `resume_at` value and return `CallOrCreate`.
            {
                fx.bcx.switch_to_block(fx.suspend_block);
                let resume_value = fx.bcx.phi(i32_type, &fx.suspend_blocks);
                let resume_at = get_ecx_resume_at(&mut fx);
                fx.bcx.store(resume_value, resume_at);

                let ret = fx.bcx.iconst(fx.i8_type, InstructionResult::CallOrCreate as i64);
                fx.incoming_returns.push((ret, fx.suspend_block));
                fx.bcx.br(fx.return_block);
            }
        } else {
            debug_assert!(fx.resume_blocks.is_empty());
            debug_assert!(fx.suspend_blocks.is_empty());

            fx.bcx.switch_to_block(post_entry_block);
            load_len_at_start(&mut fx);
            fx.bcx.br(first_inst_block);

            fx.bcx.switch_to_block(resume_block);
            fx.bcx.unreachable();
            fx.bcx.switch_to_block(fx.suspend_block);
            fx.bcx.unreachable();
        }

        // Finalize the return block.
        fx.bcx.switch_to_block(fx.return_block);
        let return_value = fx.bcx.phi(fx.i8_type, &fx.incoming_returns);
        if store_stack_length {
            let len = fx.stack_len.load(&mut fx.bcx, "stack_len");
            fx.bcx.store(len, stack_len_arg);
        }
        fx.bcx.ret(&[return_value]);

        Ok(())
    }

    fn translate_inst(&mut self, inst: Inst) -> Result<()> {
        self.current_inst = inst;
        let data = self.bytecode.inst(inst);
        let entry_block = self.inst_entries[inst];
        self.bcx.switch_to_block(entry_block);

        let opcode = data.opcode;

        let branch_to_next_opcode = |this: &mut Self| {
            debug_assert!(
                !this.bytecode.is_instr_diverging(inst),
                "attempted to branch to next instruction in a diverging instruction: {data:?}",
            );
            if let Some(next) = this.inst_entries.get(inst + 1) {
                this.bcx.br(*next);
            }
        };
        // Currently a noop.
        // let epilogue = |this: &mut Self| {
        //     this.bcx.seal_block(entry_block);
        // };

        /// Makes sure to run cleanup code and return.
        /// Use `no_branch` to skip the branch to the next opcode.
        /// Use `build` to build the return instruction and skip the branch.
        macro_rules! goto_return {
            ($comment:expr) => {
                branch_to_next_opcode(self);
                goto_return!(no_branch $comment);
            };
            (no_branch $comment:expr) => {
                if self.comments_enabled {
                    self.add_comment($comment);
                }
                // epilogue(self);
                return Ok(());
            };
            (build $ret:expr) => {{
                self.build_return_imm($ret);
                goto_return!(no_branch "");
            }};
        }

        // Assert that we already skipped the block.
        debug_assert!(!data.flags.contains(InstFlags::DEAD_CODE));

        if cfg!(test) && opcode == TEST_SUSPEND {
            self.suspend();
            goto_return!(no_branch "");
        }

        // Disabled instructions don't pay gas.
        if data.flags.contains(InstFlags::DISABLED) {
            goto_return!(build InstructionResult::NotActivated);
        }
        if data.flags.contains(InstFlags::UNKNOWN) {
            goto_return!(build InstructionResult::OpcodeNotFound);
        }

        // Pay static gas.
        if !self.disable_gas {
            if let Some(static_gas) = data.static_gas() {
                self.gas_cost_imm(static_gas as u64);
            }
        }

        if data.flags.contains(InstFlags::SKIP_LOGIC) {
            goto_return!("skipped");
        }

        // Stack I/O.
        self.len_offset = 0;
        'stack_io: {
            let (mut inp, out) = data.stack_io().both();

            if data.is_legacy_static_jump() {
                inp -= 1;
            }

            let may_underflow = inp > 0;
            let diff = out as i64 - inp as i64;
            let may_overflow = diff > 0;

            if !(may_overflow || may_underflow) {
                break 'stack_io;
            }
            self.len_before = self.stack_len.load(&mut self.bcx, "stack_len");

            let underflow = |this: &mut Self| {
                this.bcx.icmp_imm(IntCC::UnsignedLessThan, this.len_before, inp as i64)
            };
            let overflow = |this: &mut Self| {
                this.bcx.icmp_imm(
                    IntCC::UnsignedGreaterThan,
                    this.len_before,
                    STACK_CAP as i64 - diff,
                )
            };

            if may_underflow && may_overflow {
                let underflow = underflow(self);
                let overflow = overflow(self);
                let cond = self.bcx.bitor(underflow, overflow);
                let ret = {
                    let under =
                        self.bcx.iconst(self.i8_type, InstructionResult::StackUnderflow as i64);
                    let over =
                        self.bcx.iconst(self.i8_type, InstructionResult::StackOverflow as i64);
                    self.bcx.select(underflow, under, over)
                };
                self.build_failure_inner(true, cond, ret);
            } else if may_underflow {
                let cond = underflow(self);
                self.build_failure(cond, InstructionResult::StackUnderflow);
            } else if may_overflow {
                let cond = overflow(self);
                self.build_failure(cond, InstructionResult::StackOverflow);
            } else {
                unreachable!("in stack_io without underflow or overflow");
            }

            if diff != 0 {
                let len_changed = self.bcx.iadd_imm(self.len_before, diff);
                self.stack_len.store(&mut self.bcx, len_changed);
            }
        }

        // Macro utils.
        macro_rules! unop {
            ($op:ident) => {{
                let mut a = self.pop();
                a = self.bcx.$op(a);
                self.push(a);
            }};
        }

        macro_rules! binop {
            ($op:ident) => {{
                let [a, b] = self.popn();
                let r = self.bcx.$op(a, b);
                self.push(r);
            }};
            (@rev $op:ident) => {{
                let [a, b] = self.popn();
                let r = self.bcx.$op(b, a);
                self.push(r);
            }};
            (@if_not_zero $op:ident) => {{
                // TODO: `select` might not have the same semantics in all backends.
                let [a, b] = self.popn();
                let b_is_zero = self.bcx.icmp_imm(IntCC::Equal, b, 0);
                let zero = self.bcx.iconst_256(U256::ZERO);
                let op_result = self.bcx.$op(a, b);
                let r = self.bcx.select(b_is_zero, zero, op_result);
                self.push(r);
            }};
        }

        macro_rules! field {
            // Gets the pointer to a field.
            ($field:ident; @get $($paths:path),*; $($spec:tt).*) => {
                self.get_field(self.$field, 0 $(+ mem::offset_of!($paths, $spec))*, stringify!($field.$($spec).*.addr))
            };
            // Gets and loads the pointer to a field.
            // The value is loaded as a native-endian 256-bit integer.
            // `@[endian]` is the endianness of the value. If native, omit it.
            ($field:ident; @load $(@[endian = $endian:tt])? $ty:expr, $($paths:path),*; $($spec:tt).*) => {{
                let ptr = field!($field; @get $($paths),*; $($spec).*);
                #[allow(unused_mut)]
                let mut value = self.bcx.load($ty, ptr, stringify!($field.$($spec).*));
                $(
                    if !cfg!(target_endian = $endian) {
                        value = self.bcx.bswap(value);
                    }
                )?
                value
            }};
            // Gets, loads, extends (if necessary), and pushes the value of a field to the stack.
            // `@[endian]` is the endianness of the value. If native, omit it.
            ($field:ident; @push $(@[endian = $endian:tt])? $ty:expr, $($rest:tt)*) => {{
                let mut value = field!($field; @load $(@[endian = $endian])? $ty, $($rest)*);
                if self.bcx.type_bit_width($ty) < 256 {
                    value = self.bcx.zext(self.word_type, value);
                }
                self.push(value);
            }};
        }
        macro_rules! env_field {
            ($($tt:tt)*) => { field!(env; $($tt)*) };
        }
        macro_rules! contract_field {
            ($($tt:tt)*) => { field!(contract; $($tt)*) };
        }

        match data.opcode {
            op::STOP => goto_return!(build InstructionResult::Stop),

            op::ADD => binop!(iadd),
            op::MUL => binop!(imul),
            op::SUB => binop!(isub),
            op::DIV => binop!(@if_not_zero udiv),
            op::SDIV => {
                let [a, b] = self.popn();
                let b_is_zero = self.bcx.icmp_imm(IntCC::Equal, b, 0);
                let r = self.bcx.lazy_select(
                    b_is_zero,
                    self.word_type,
                    |bcx, block| {
                        bcx.set_cold_block(block);
                        bcx.iconst_256(U256::ZERO)
                    },
                    |bcx, _op_block| {
                        let min = bcx.iconst_256(I256_MIN);
                        let is_weird_sdiv_edge_case = {
                            let a_is_min = bcx.icmp(IntCC::Equal, a, min);
                            let b_is_neg1 = bcx.icmp_imm(IntCC::Equal, b, -1);
                            bcx.bitand(a_is_min, b_is_neg1)
                        };
                        let sdiv_result = bcx.sdiv(a, b);
                        bcx.select(is_weird_sdiv_edge_case, min, sdiv_result)
                    },
                );
                self.push(r);
            }
            op::MOD => binop!(@if_not_zero urem),
            op::SMOD => binop!(@if_not_zero srem),
            op::ADDMOD => {
                let sp = self.sp_after_input();
                let _ = self.callback(Callback::AddMod, &[sp]);
            }
            op::MULMOD => {
                let sp = self.sp_after_input();
                let _ = self.callback(Callback::MulMod, &[sp]);
            }
            op::EXP => {
                let sp = self.sp_after_input();
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::Exp, &[self.ecx, sp, spec_id]);
            }
            op::SIGNEXTEND => {
                // From the yellow paper:
                /*
                let [ext, x] = stack.pop();
                let t = 256 - 8 * (ext + 1);
                let mut result = x;
                result[..t] = [x[t]; t]; // Index by bits.
                */

                let [ext, x] = self.popn();
                // For 31 we also don't need to do anything.
                let might_do_something = self.bcx.icmp_imm(IntCC::UnsignedLessThan, ext, 31);
                let r = self.bcx.lazy_select(
                    might_do_something,
                    self.word_type,
                    |bcx, _block| {
                        // Adapted from revm: https://github.com/bluealloy/revm/blob/fda371f73aba2c30a83c639608be78145fd1123b/crates/interpreter/src/instructions/arithmetic.rs#L89
                        // let bit_index = 8 * ext + 7;
                        // let bit = (x >> bit_index) & 1 != 0;
                        // let mask = (1 << bit_index) - 1;
                        // let r = if bit { x | !mask } else { *x & mask };

                        // let bit_index = 8 * ext + 7;
                        let bit_index = bcx.imul_imm(ext, 8);
                        let bit_index = bcx.iadd_imm(bit_index, 7);

                        // let bit = (x >> bit_index) & 1 != 0;
                        let one = bcx.iconst_256(U256::from(1));
                        let bit = bcx.ushr(x, bit_index);
                        let bit = bcx.bitand(bit, one);
                        let bit = bcx.icmp_imm(IntCC::NotEqual, bit, 0);

                        // let mask = (1 << bit_index) - 1;
                        let mask = bcx.ishl(one, bit_index);
                        let mask = bcx.isub_imm(mask, 1);

                        // let r = if bit { x | !mask } else { *x & mask };
                        let not_mask = bcx.bitnot(mask);
                        let sext = bcx.bitor(x, not_mask);
                        let zext = bcx.bitand(x, mask);
                        bcx.select(bit, sext, zext)
                    },
                    |_bcx, _block| x,
                );
                self.push(r);
            }

            op::LT | op::GT | op::SLT | op::SGT | op::EQ => {
                let cond = match opcode {
                    op::LT => IntCC::UnsignedLessThan,
                    op::GT => IntCC::UnsignedGreaterThan,
                    op::SLT => IntCC::SignedLessThan,
                    op::SGT => IntCC::SignedGreaterThan,
                    op::EQ => IntCC::Equal,
                    _ => unreachable!(),
                };

                let [a, b] = self.popn();
                let r = self.bcx.icmp(cond, a, b);
                let r = self.bcx.zext(self.word_type, r);
                self.push(r);
            }
            op::ISZERO => {
                let a = self.pop();
                let r = self.bcx.icmp_imm(IntCC::Equal, a, 0);
                let r = self.bcx.zext(self.word_type, r);
                self.push(r);
            }
            op::AND => binop!(bitand),
            op::OR => binop!(bitor),
            op::XOR => binop!(bitxor),
            op::NOT => unop!(bitnot),
            op::BYTE => {
                let [index, value] = self.popn();
                let cond = self.bcx.icmp_imm(IntCC::UnsignedLessThan, index, 32);
                let byte = {
                    // (value >> (31 - index) * 8) & 0xFF
                    let thirty_one = self.bcx.iconst_256(U256::from(31));
                    let shift = self.bcx.isub(thirty_one, index);
                    let shift = self.bcx.imul_imm(shift, 8);
                    let shifted = self.bcx.ushr(value, shift);
                    let mask = self.bcx.iconst_256(U256::from(0xFF));
                    self.bcx.bitand(shifted, mask)
                };
                let zero = self.bcx.iconst_256(U256::ZERO);
                let r = self.bcx.select(cond, byte, zero);
                self.push(r);
            }
            op::SHL => binop!(@rev ishl),
            op::SHR => binop!(@rev ushr),
            op::SAR => binop!(@rev sshr),

            op::KECCAK256 => {
                let sp = self.sp_after_input();
                self.callback_ir(Callback::Keccak256, &[self.ecx, sp]);
            }

            op::ADDRESS => {
                contract_field!(@push @[endian = "big"] self.address_type, Contract; address)
            }
            op::BALANCE => {
                let sp = self.sp_after_input();
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::Balance, &[self.ecx, sp, spec_id]);
            }
            op::ORIGIN => {
                env_field!(@push @[endian = "big"] self.address_type, Env, TxEnv; tx.caller)
            }
            op::CALLER => {
                contract_field!(@push @[endian = "big"] self.address_type, Contract; caller)
            }
            op::CALLVALUE => {
                contract_field!(@push @[endian = "little"] self.word_type, Contract; value)
            }
            op::CALLDATALOAD => {
                let index = self.pop();
                let len_ptr = contract_field!(@get Contract, pf::Bytes; input.len);
                let len = self.bcx.load(self.isize_type, len_ptr, "input.len");
                let len = self.bcx.zext(self.word_type, len);
                let in_bounds = self.bcx.icmp(IntCC::UnsignedLessThan, index, len);

                let ptr = contract_field!(@get Contract, pf::Bytes; input.ptr);
                let zero = self.bcx.iconst_256(U256::ZERO);
                let value = self.lazy_select(
                    in_bounds,
                    self.word_type,
                    |this, _| {
                        let ptr = this.bcx.load(this.bcx.type_ptr(), ptr, "contract.input.ptr");
                        let index = this.bcx.ireduce(this.isize_type, index);
                        let calldata = this.bcx.gep(this.i8_type, ptr, &[index], "calldata.addr");

                        // `32.min(contract.input.len() - index)`
                        let slice_len = {
                            let len = this.bcx.ireduce(this.isize_type, len);
                            let diff = this.bcx.isub(len, index);
                            let max = this.bcx.iconst(this.isize_type, 32);
                            this.bcx.umin(diff, max)
                        };

                        let tmp = this.bcx.new_stack_slot(this.word_type, "calldata.addr");
                        this.bcx.stack_store(zero, tmp);
                        let tmp_addr = this.bcx.stack_addr(tmp);
                        this.bcx.memcpy(tmp_addr, calldata, slice_len);
                        let mut value = this.bcx.stack_load(this.word_type, tmp, "calldata.i256");
                        if cfg!(target_endian = "little") {
                            value = this.bcx.bswap(value);
                        }
                        value
                    },
                    |_, _| zero,
                );
                self.push(value);
            }
            op::CALLDATASIZE => {
                contract_field!(@push self.isize_type, Contract, pf::Bytes; input.len)
            }
            op::CALLDATACOPY => {
                let sp = self.sp_after_input();
                self.callback_ir(Callback::CallDataCopy, &[self.ecx, sp]);
            }
            op::CODESIZE => {
                contract_field!(@push self.isize_type, Contract, pf::BytecodeLocked; bytecode.original_len)
            }
            op::CODECOPY => {
                let sp = self.sp_after_input();
                self.callback_ir(Callback::CodeCopy, &[self.ecx, sp]);
            }

            op::GASPRICE => {
                env_field!(@push @[endian = "little"] self.word_type, Env, TxEnv; tx.gas_price)
            }
            op::EXTCODESIZE => {
                let sp = self.sp_after_input();
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::ExtCodeSize, &[self.ecx, sp, spec_id]);
            }
            op::EXTCODECOPY => {
                let sp = self.sp_after_input();
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::ExtCodeCopy, &[self.ecx, sp, spec_id]);
            }
            op::RETURNDATASIZE => {
                field!(ecx; @push self.isize_type, EvmContext<'_>, pf::Slice; return_data.len)
            }
            op::RETURNDATACOPY => {
                let sp = self.sp_after_input();
                self.callback_ir(Callback::ReturnDataCopy, &[self.ecx, sp]);
            }
            op::EXTCODEHASH => {
                let sp = self.sp_after_input();
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::ExtCodeHash, &[self.ecx, sp, spec_id]);
            }
            op::BLOCKHASH => {
                let sp = self.sp_after_input();
                self.callback_ir(Callback::BlockHash, &[self.ecx, sp]);
            }
            op::COINBASE => {
                env_field!(@push @[endian = "big"] self.address_type, Env, BlockEnv; block.coinbase)
            }
            op::TIMESTAMP => {
                env_field!(@push @[endian = "little"] self.word_type, Env, BlockEnv; block.timestamp)
            }
            op::NUMBER => {
                env_field!(@push @[endian = "little"] self.word_type, Env, BlockEnv; block.number)
            }
            op::DIFFICULTY => {
                let value = if self.bytecode.spec_id.is_enabled_in(SpecId::MERGE) {
                    // Option<[u8; 32]> == { u8, [u8; 32] }
                    let opt = env_field!(@get Env, BlockEnv; block.prevrandao);
                    let is_some = {
                        let name = "env.block.prevrandao.is_some";
                        let is_some = self.bcx.load(self.i8_type, opt, name);
                        self.bcx.icmp_imm(IntCC::NotEqual, is_some, 0)
                    };
                    let some = {
                        let one = self.bcx.iconst(self.isize_type, 1);
                        let ptr =
                            self.bcx.gep(self.i8_type, opt, &[one], "env.block.prevrandao.ptr");
                        let mut v = self.bcx.load(self.word_type, ptr, "env.block.prevrandao");
                        if !cfg!(target_endian = "big") {
                            v = self.bcx.bswap(v);
                        }
                        v
                    };
                    let none = self.bcx.iconst_256(U256::ZERO);
                    self.bcx.select(is_some, some, none)
                } else {
                    env_field!(@load @[endian = "little"] self.word_type, Env, BlockEnv; block.difficulty)
                };
                self.push(value);
            }
            op::GASLIMIT => {
                env_field!(@push @[endian = "little"] self.word_type, Env, BlockEnv; block.gas_limit)
            }
            op::CHAINID => env_field!(@push self.bcx.type_int(64), Env, CfgEnv; cfg.chain_id),
            op::SELFBALANCE => {
                let slot = self.sp_at_top();
                self.callback_ir(Callback::SelfBalance, &[self.ecx, slot]);
            }
            op::BASEFEE => {
                env_field!(@push @[endian = "little"] self.word_type, Env, BlockEnv; block.basefee)
            }
            op::BLOBHASH => {
                let sp = self.sp_after_input();
                let _ = self.callback(Callback::BlobHash, &[self.ecx, sp]);
            }
            op::BLOBBASEFEE => {
                let len = self.len_before();
                let slot = self.sp_at(len);
                let _ = self.callback(Callback::BlobBaseFee, &[self.ecx, slot]);
            }

            op::POP => { /* Already handled in stack_io */ }
            op::MLOAD => {
                let sp = self.sp_after_input();
                self.callback_ir(Callback::Mload, &[self.ecx, sp]);
            }
            op::MSTORE => {
                let sp = self.sp_after_input();
                self.callback_ir(Callback::Mstore, &[self.ecx, sp]);
            }
            op::MSTORE8 => {
                let sp = self.sp_after_input();
                self.callback_ir(Callback::Mstore8, &[self.ecx, sp]);
            }
            op::SLOAD => {
                let sp = self.sp_after_input();
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::Sload, &[self.ecx, sp, spec_id]);
            }
            op::SSTORE => {
                self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
                let sp = self.sp_after_input();
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::Sstore, &[self.ecx, sp, spec_id]);
            }
            op::JUMP | op::JUMPI => {
                if data.flags.contains(InstFlags::INVALID_JUMP) {
                    self.build_return_imm(InstructionResult::InvalidJump);
                } else {
                    let is_static = data.flags.contains(InstFlags::STATIC_JUMP);
                    let target = if is_static {
                        let target_inst = data.data as usize;
                        debug_assert_eq!(
                            *self.bytecode.inst(target_inst),
                            op::JUMPDEST,
                            "jumping to non-JUMPDEST; target_inst={target_inst}",
                        );
                        self.inst_entries[target_inst]
                    } else {
                        // Dynamic jump.
                        debug_assert!(self.bytecode.has_dynamic_jumps());
                        let target = self.pop();
                        self.incoming_dynamic_jumps
                            .push((target, self.bcx.current_block().unwrap()));
                        self.dynamic_jump_table
                    };

                    if opcode == op::JUMPI {
                        let cond_word = self.pop();
                        let cond = self.bcx.icmp_imm(IntCC::NotEqual, cond_word, 0);
                        let next = self.inst_entries[inst + 1];
                        self.bcx.brif(cond, target, next);
                    } else {
                        self.bcx.br(target);
                    }
                    self.inst_entries[inst] = self.bcx.current_block().unwrap();
                }

                goto_return!(no_branch "");
            }
            op::PC => {
                let pc = self.bcx.iconst_256(U256::from(data.pc));
                self.push(pc);
            }
            op::MSIZE => {
                let msize = self.callback(Callback::Msize, &[self.ecx]).unwrap();
                let msize = self.bcx.zext(self.word_type, msize);
                self.push(msize);
            }
            op::GAS => {
                let remaining = self.load_gas_remaining();
                let remaining = self.bcx.zext(self.word_type, remaining);
                self.push(remaining);
            }
            op::JUMPDEST => {
                self.bcx.nop();
            }
            op::TLOAD => {
                let sp = self.sp_after_input();
                let _ = self.callback(Callback::Tload, &[self.ecx, sp]);
            }
            op::TSTORE => {
                self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
                let sp = self.sp_after_input();
                let _ = self.callback(Callback::Tstore, &[self.ecx, sp]);
            }

            op::PUSH0 => {
                let value = self.bcx.iconst_256(U256::ZERO);
                self.push(value);
            }
            op::PUSH1..=op::PUSH32 => {
                // NOTE: This can be None if the bytecode is invalid.
                let imm = self.bytecode.get_imm_of(data);
                let value = imm.map(U256::from_be_slice).unwrap_or_default();
                let value = self.bcx.iconst_256(value);
                self.push(value);
            }

            op::DUP1..=op::DUP16 => self.dup(opcode - op::DUP1 + 1),

            op::SWAP1..=op::SWAP16 => self.swap(opcode - op::SWAP1 + 1),

            op::LOG0..=op::LOG4 => {
                self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
                let n = opcode - op::LOG0;
                let sp = self.sp_after_input();
                let n = self.bcx.iconst(self.i8_type, n as i64);
                self.callback_ir(Callback::Log, &[self.ecx, sp, n]);
            }

            op::CREATE => {
                self.create_common(false);
                goto_return!(no_branch "");
            }
            op::CALL => {
                self.call_common(CallKind::Call);
                goto_return!(no_branch "");
            }
            op::CALLCODE => {
                self.call_common(CallKind::CallCode);
                goto_return!(no_branch "");
            }
            op::RETURN => {
                self.return_common(InstructionResult::Return);
                goto_return!(no_branch "");
            }
            op::DELEGATECALL => {
                self.call_common(CallKind::DelegateCall);
                goto_return!(no_branch "");
            }
            op::CREATE2 => {
                self.create_common(true);
                goto_return!(no_branch "");
            }

            op::STATICCALL => {
                self.call_common(CallKind::StaticCall);
                goto_return!(no_branch "");
            }

            op::REVERT => {
                self.return_common(InstructionResult::Revert);
                goto_return!(no_branch "");
            }
            op::INVALID => goto_return!(build InstructionResult::InvalidFEOpcode),
            op::SELFDESTRUCT => {
                self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
                let sp = self.sp_after_input();
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::SelfDestruct, &[self.ecx, sp, spec_id]);
                goto_return!(build InstructionResult::SelfDestruct);
            }

            _ => unreachable!("unimplemented instructions: {}", data.to_op_in(self.bytecode)),
        }

        goto_return!("normal exit");
    }

    /// Pushes a 256-bit value onto the stack.
    fn push(&mut self, value: B::Value) {
        self.pushn(&[value]);
    }

    /// Pushes 256-bit values onto the stack.
    fn pushn(&mut self, values: &[B::Value]) {
        let len_start = self.len_before();
        for &value in values {
            let len = self.bcx.iadd_imm(len_start, self.len_offset as i64);
            self.len_offset += 1;
            let sp = self.sp_at(len);
            self.bcx.store(value, sp);
        }
    }

    /// Removes the topmost element from the stack and returns it.
    fn pop(&mut self) -> B::Value {
        self.popn::<1>()[0]
    }

    /// Removes the topmost `N` elements from the stack and returns them.
    ///
    /// If `load` is `false`, returns just the pointers.
    fn popn<const N: usize>(&mut self) -> [B::Value; N] {
        debug_assert_ne!(N, 0);

        let len_start = self.len_before();
        std::array::from_fn(|i| {
            self.len_offset -= 1;
            let len = self.bcx.iadd_imm(len_start, self.len_offset as i64);
            let sp = self.sp_at(len);
            let name = b'a' + i as u8;
            self.load_word(sp, core::str::from_utf8(&[name]).unwrap())
        })
    }

    /// Returns an error if the current context is a static call.
    fn fail_if_staticcall(&mut self, ret: InstructionResult) {
        let ptr =
            self.get_field(self.ecx, mem::offset_of!(EvmContext<'_>, is_static), "ecx.is_static");
        let bool = self.bcx.type_int(1);
        let is_static = self.bcx.load(bool, ptr, "is_static");
        self.build_failure(is_static, ret)
    }

    /// Duplicates the `n`th value from the top of the stack.
    /// `n` cannot be `0`.
    fn dup(&mut self, n: u8) {
        debug_assert_ne!(n, 0);
        let len = self.len_before();
        let sp = self.sp_from_top(len, n as usize);
        let value = self.load_word(sp, &format!("dup{n}"));
        self.push(value);
    }

    /// Swaps the topmost value with the `n`th value from the top.
    fn swap(&mut self, n: u8) {
        debug_assert_ne!(n, 0);
        let len = self.len_before();
        // Load a.
        let a_sp = self.sp_from_top(len, n as usize + 1);
        let a = self.load_word(a_sp, "swap.a");
        // Load b.
        let b_sp = self.sp_from_top(len, 1);
        let b = self.load_word(b_sp, "swap.top");
        // Store.
        self.bcx.store(a, b_sp);
        self.bcx.store(b, a_sp);
    }

    /// `RETURN` or `REVERT` instruction.
    fn return_common(&mut self, ir: InstructionResult) {
        let sp = self.sp_after_input();
        let ir_const = self.bcx.iconst(self.i8_type, ir as i64);
        self.callback_ir(Callback::DoReturn, &[self.ecx, sp, ir_const]);
        self.build_return_imm(ir);
    }

    fn create_common(&mut self, is_create2: bool) {
        self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
        let sp = self.sp_after_input();
        let is_create2 = self.bcx.iconst(self.bcx.type_int(1), is_create2 as i64);
        self.callback_ir(Callback::Create, &[self.ecx, sp, is_create2]);
        self.suspend();
    }

    fn call_common(&mut self, call_kind: CallKind) {
        // TODO
        let _ = call_kind;
        self.suspend();
    }

    /// Suspend execution, storing the resume point in the context.
    fn suspend(&mut self) {
        // Register the next instruction as the resume block.
        let value = self.add_resume_at(self.inst_entries[self.current_inst + 1]);

        // Register the current block as the suspend block.
        let value = self.bcx.iconst(self.bcx.type_int(32), value as i64);
        self.suspend_blocks.push((value, self.bcx.current_block().unwrap()));

        // Branch to the suspend block.
        self.bcx.br(self.suspend_block);
    }

    /// Adds a resume point and returns its index.
    fn add_resume_at(&mut self, block: B::BasicBlock) -> usize {
        let value = self.resume_blocks.len();
        let value_ = self.bcx.iconst(self.bcx.type_int(32), value as i64);
        self.resume_blocks.push((value_, block));
        value
    }

    /// Loads the word at the given pointer.
    fn load_word(&mut self, ptr: B::Value, name: &str) -> B::Value {
        self.bcx.load(self.word_type, ptr, name)
    }

    /// Gets the stack length before the current instruction.
    fn len_before(&mut self) -> B::Value {
        self.len_before
    }

    /// Returns the spec ID as a value.
    fn const_spec_id(&mut self) -> B::Value {
        self.bcx.iconst(self.i8_type, self.bytecode.spec_id as i64)
    }

    /// Gets the environment field at the given offset.
    fn get_field(&mut self, ptr: B::Value, offset: usize, name: &str) -> B::Value {
        let offset = self.bcx.iconst(self.isize_type, offset as i64);
        self.bcx.gep(self.i8_type, ptr, &[offset], name)
    }

    /// Loads the gas used.
    fn load_gas_remaining(&mut self) -> B::Value {
        self.gas_remaining.load(&mut self.bcx, "gas_remaining")
    }

    /// Stores the gas used.
    fn store_gas_remaining(&mut self, value: B::Value) {
        self.gas_remaining.store(&mut self.bcx, value);
    }

    /// Returns the stack pointer at the top (`&stack[stack.len]`).
    fn sp_at_top(&mut self) -> B::Value {
        let len = self.len_before();
        self.sp_at(len)
    }

    /// Returns the stack pointer after the input has been popped
    /// (`&stack[stack.len - op.input()]`).
    fn sp_after_input(&mut self) -> B::Value {
        let mut len = self.len_before();
        let input_len = self.current_inst().stack_io().input();
        if input_len > 0 {
            len = self.bcx.isub_imm(len, input_len as i64);
        }
        self.sp_at(len)
    }

    /// Returns the stack pointer at `len` (`&stack[len]`).
    fn sp_at(&mut self, len: B::Value) -> B::Value {
        let ptr = self.stack.addr(&mut self.bcx);
        self.bcx.gep(self.word_type, ptr, &[len], "sp")
    }

    /// Returns the stack pointer at `len` from the top (`&stack[CAPACITY - len]`).
    fn sp_from_top(&mut self, len: B::Value, n: usize) -> B::Value {
        debug_assert_ne!(n, 0);
        let len = self.bcx.isub_imm(len, n as i64);
        self.sp_at(len)
    }

    /// Builds a gas cost deduction for an immediate value.
    fn gas_cost_imm(&mut self, cost: u64) {
        if self.disable_gas || cost == 0 {
            return;
        }
        let value = self.bcx.iconst(self.isize_type, cost as i64);
        self.gas_cost(value);
    }

    /// Builds a gas cost deduction for a value.
    fn gas_cost(&mut self, cost: B::Value) {
        if self.disable_gas {
            return;
        }

        // `Gas::record_cost`
        let gas_remaining = self.load_gas_remaining();
        let (res, overflow) = self.bcx.usub_overflow(gas_remaining, cost);
        self.build_failure(overflow, InstructionResult::OutOfGas);

        let nomem = self.gas_remaining_nomem.load(&mut self.bcx, "gas_remaining_nomem");
        let nomem = self.bcx.isub(nomem, cost);
        self.gas_remaining_nomem.store(&mut self.bcx, nomem);

        self.store_gas_remaining(res);
    }

    /*
    /// `if success_cond { ... } else { return ret }`
    fn build_failure_inv(&mut self, success_cond: B::Value, ret: InstructionResult) {
        self.build_failure_imm_inner(false, success_cond, ret);
    }
    */

    /// `if failure_cond { return ret } else { ... }`
    fn build_failure(&mut self, failure_cond: B::Value, ret: InstructionResult) {
        self.build_failure_imm_inner(true, failure_cond, ret);
    }

    fn build_failure_imm_inner(
        &mut self,
        is_failure: bool,
        cond: B::Value,
        ret: InstructionResult,
    ) {
        let ret = self.bcx.iconst(self.i8_type, ret as i64);
        self.build_failure_inner(is_failure, cond, ret);
    }

    fn build_failure_inner(&mut self, is_failure: bool, cond: B::Value, ret: B::Value) {
        let failure = self.create_block_after_current("fail");
        let target = self.create_block_after(failure, "contd");
        if is_failure {
            self.bcx.brif(cond, failure, target);
        } else {
            self.bcx.brif(cond, target, failure);
        }

        self.bcx.set_cold_block(failure);
        self.bcx.switch_to_block(failure);
        self.build_return(ret);

        self.bcx.switch_to_block(target);
    }

    /// Builds a branch to the return block.
    fn build_return_imm(&mut self, ret: InstructionResult) {
        let ret_value = self.bcx.iconst(self.i8_type, ret as i64);
        self.build_return(ret_value);
        if self.comments_enabled {
            self.add_comment(&format!("return {ret:?}"));
        }
    }

    /// Builds a branch to the return block.
    fn build_return(&mut self, ret: B::Value) {
        self.incoming_returns.push((ret, self.bcx.current_block().unwrap()));
        self.bcx.br(self.return_block);
    }

    // Pointer must not be null if `must_be_set` is true.
    fn pointer_panic_with_bool(
        &mut self,
        must_be_set: bool,
        ptr: B::Value,
        name: &str,
        extra: &str,
    ) {
        if !must_be_set {
            return;
        }
        let panic_cond = self.bcx.is_null(ptr);
        let mut msg = format!("revm_jit panic: {name} must not be null");
        if !extra.is_empty() {
            write!(msg, " ({extra})").unwrap();
        }
        self.build_panic_cond(panic_cond, &msg);
    }

    fn build_panic_cond(&mut self, cond: B::Value, msg: &str) {
        let failure = self.create_block_after_current("panic");
        let target = self.create_block_after(failure, "contd");
        self.bcx.brif(cond, failure, target);

        // `panic` is already marked as a cold function call.
        // self.bcx.set_cold_block(failure);
        self.bcx.switch_to_block(failure);
        self.call_panic(msg);

        self.bcx.switch_to_block(target);
    }

    fn call_panic(&mut self, msg: &str) {
        let function = self.callback_function(Callback::Panic);
        let ptr = self.bcx.str_const(msg);
        let len = self.bcx.iconst(self.isize_type, msg.len() as i64);
        let _ = self.bcx.call(function, &[ptr, len]);
        self.bcx.unreachable();
    }

    /// Build a call to a callback that returns an [`InstructionResult`].
    fn callback_ir(&mut self, callback: Callback, args: &[B::Value]) {
        let ret = self.callback(callback, args).expect("callback does not return a value");
        let failure = self.bcx.icmp_imm(IntCC::NotEqual, ret, InstructionResult::Continue as i64);
        self.build_failure_inner(true, failure, ret);
    }

    /// Build a call to a callback.
    #[must_use]
    fn callback(&mut self, callback: Callback, args: &[B::Value]) -> Option<B::Value> {
        let function = self.callback_function(callback);
        self.bcx.call(function, args)
    }

    fn callback_function(&mut self, callback: Callback) -> B::Function {
        self.callbacks.get(callback, &mut self.bcx)
    }

    /// Adds a comment to the current instruction.
    fn add_comment(&mut self, comment: &str) {
        if comment.is_empty() {
            return;
        }
        self.bcx.add_comment_to_current_inst(comment);
    }

    /// Returns the current instruction.
    fn current_inst(&self) -> &InstData {
        self.bytecode.inst(self.current_inst)
    }

    /// Returns the current block.
    fn current_block(&mut self) -> B::BasicBlock {
        // There always is a block present.
        self.bcx.current_block().expect("no blocks")
    }

    /*
    /// Creates a named block.
    fn create_block(&mut self, name: &str) -> B::BasicBlock {
        let name = self.op_block_name(name);
        self.bcx.create_block(&name)
    }
    */

    /// Creates a named block after the given block.
    fn create_block_after(&mut self, after: B::BasicBlock, name: &str) -> B::BasicBlock {
        let name = self.op_block_name(name);
        self.bcx.create_block_after(after, &name)
    }

    /// Creates a named block after the current block.
    fn create_block_after_current(&mut self, name: &str) -> B::BasicBlock {
        let after = self.current_block();
        self.create_block_after(after, name)
    }

    /// Returns the block name for the current opcode with the given suffix.
    fn op_block_name(&self, name: &str) -> String {
        if self.current_inst == usize::MAX {
            return format!("entry.{name}");
        }
        op_block_name_with(self.current_inst, self.bytecode.inst(self.current_inst), name)
    }

    #[inline]
    fn lazy_select(
        &mut self,
        cond: B::Value,
        ty: B::Type,
        then_value: impl FnOnce(&mut Self, B::BasicBlock) -> B::Value,
        else_value: impl FnOnce(&mut Self, B::BasicBlock) -> B::Value,
    ) -> B::Value {
        let this1 = unsafe { std::mem::transmute::<&mut Self, &mut Self>(self) };
        let this2 = unsafe { std::mem::transmute::<&mut Self, &mut Self>(self) };
        self.bcx.lazy_select(
            cond,
            ty,
            #[inline]
            move |_bcx, block| then_value(this1, block),
            #[inline]
            move |_bcx, block| else_value(this2, block),
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct Pointer<B: Backend> {
    /// The type of the pointee.
    ty: B::Type,
    /// The base of the pointer. Either an address or a stack slot.
    base: PointerBase<B>,
}

#[derive(Clone, Copy, Debug)]
enum PointerBase<B: Backend> {
    Address(B::Value),
    StackSlot(B::StackSlot),
}

impl<B: Backend> Pointer<B> {
    /// Returns `true` if the pointer is an address.
    #[allow(dead_code)]
    fn is_address(&self) -> bool {
        matches!(self.base, PointerBase::Address(_))
    }

    /// Returns `true` if the pointer is a stack slot.
    #[allow(dead_code)]
    fn is_stack_slot(&self) -> bool {
        matches!(self.base, PointerBase::StackSlot(_))
    }

    /// Loads the value from the pointer.
    fn load(&self, bcx: &mut B::Builder<'_>, name: &str) -> B::Value {
        match self.base {
            PointerBase::Address(ptr) => bcx.load(self.ty, ptr, name),
            PointerBase::StackSlot(slot) => bcx.stack_load(self.ty, slot, name),
        }
    }

    /// Stores the value to the pointer.
    fn store(&self, bcx: &mut B::Builder<'_>, value: B::Value) {
        match self.base {
            PointerBase::Address(ptr) => bcx.store(value, ptr),
            PointerBase::StackSlot(slot) => bcx.stack_store(value, slot),
        }
    }

    /// Stores the value to the pointer.
    fn store_imm(&self, bcx: &mut B::Builder<'_>, value: i64) {
        let value = bcx.iconst(self.ty, value);
        self.store(bcx, value)
    }

    /// Gets the address of the pointer.
    fn addr(&self, bcx: &mut B::Builder<'_>) -> B::Value {
        match self.base {
            PointerBase::Address(ptr) => ptr,
            PointerBase::StackSlot(slot) => bcx.stack_addr(slot),
        }
    }
}

enum CallKind {
    Call,
    CallCode,
    DelegateCall,
    StaticCall,
}

// HACK: Need these structs' fields to be public for `offset_of!`.
// `pf == private_fields`.
mod pf {
    use super::*;
    use revm_primitives::JumpMap;

    #[allow(dead_code)]
    pub(super) struct Bytes {
        pub(super) ptr: *const u8,
        pub(super) len: usize,
        // inlined "trait object"
        data: AtomicPtr<()>,
        vtable: &'static Vtable,
    }
    #[allow(dead_code)]
    struct Vtable {
        /// fn(data, ptr, len)
        clone: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> Bytes,
        /// fn(data, ptr, len)
        ///
        /// takes `Bytes` to value
        to_vec: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> Vec<u8>,
        /// fn(data, ptr, len)
        drop: unsafe fn(&mut AtomicPtr<()>, *const u8, usize),
    }

    #[allow(dead_code)]
    pub(super) struct BytecodeLocked {
        pub(super) bytecode: Bytes,
        pub(super) original_len: usize,
        jump_map: JumpMap,
    }

    #[repr(C)] // See core::ptr::metadata::PtrComponents
    #[allow(dead_code)]
    pub(super) struct Slice {
        pub(super) ptr: *const u8,
        pub(super) len: usize,
    }

    #[allow(dead_code)]
    pub(super) struct Gas {
        /// The initial gas limit. This is constant throughout execution.
        pub(super) limit: u64,
        /// The remaining gas.
        pub(super) remaining: u64,
        /// The remaining gas, without memory expansion.
        pub(super) remaining_nomem: u64,
        /// The **last** memory expansion cost.
        memory: u64,
        /// Refunded gas. This is used only at the end of execution.
        refunded: i64,
    }
}

fn op_block_name_with(op: Inst, data: &InstData, with: &str) -> String {
    let data = data.to_op();
    if with.is_empty() {
        format!("op.{op}.{data}")
    } else {
        format!("op.{op}.{data}.{with}")
    }
}

#[cfg(test)]
#[allow(clippy::needless_update)]
mod tests {
    use super::*;
    use crate::*;
    use interpreter::{DummyHost, Host};
    use primitives::{BlobExcessGasAndPrice, HashMap, LogData, B256};
    use revm_interpreter::{gas, opcode as op};
    use revm_primitives::{hex, keccak256, spec_to_generic, Address, Bytes, KECCAK_EMPTY};
    use std::{fmt, sync::OnceLock};

    #[cfg(feature = "llvm")]
    use llvm::inkwell::context::Context as LlvmContext;

    const I256_MAX: U256 = U256::from_limbs([
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x7FFFFFFFFFFFFFFF,
    ]);

    macro_rules! build_push32 {
        ($code:ident[$i:ident], $x:expr) => {{
            $code[$i] = op::PUSH32;
            $i += 1;
            $code[$i..$i + 32].copy_from_slice(&$x.to_be_bytes::<32>());
            $i += 32;
        }};
    }

    fn bytecode_unop(op: u8, a: U256) -> [u8; 34] {
        let mut code = [0; 34];
        let mut i = 0;
        build_push32!(code[i], a);
        code[i] = op;
        code
    }

    fn bytecode_binop(op: u8, a: U256, b: U256) -> [u8; 67] {
        let mut code = [0; 67];
        let mut i = 0;
        build_push32!(code[i], b);
        build_push32!(code[i], a);
        code[i] = op;
        code
    }

    fn bytecode_ternop(op: u8, a: U256, b: U256, c: U256) -> [u8; 100] {
        let mut code = [0; 100];
        let mut i = 0;
        build_push32!(code[i], c);
        build_push32!(code[i], b);
        build_push32!(code[i], a);
        code[i] = op;
        code
    }

    macro_rules! with_matrix {
        ($run:ident) => {
            #[cfg(feature = "llvm")]
            mod llvm {
                use super::*;

                fn run_llvm(jit: &mut JitEvm<JitEvmLlvmBackend<'_>>) {
                    set_test_dump(jit, module_path!());
                    $run(jit);
                }

                #[test]
                fn unopt() {
                    with_llvm_backend_jit(OptimizationLevel::None, run_llvm);
                }

                #[test]
                fn opt() {
                    with_llvm_backend_jit(OptimizationLevel::Aggressive, run_llvm);
                }
            }
        };
    }

    macro_rules! tests {
        ($($group:ident { $($t:tt)* })*) => { uint! {
            $(
                mod $group {
                    tests!(@cases $($t)*);
                }
            )*
        }};

        (@cases $( $name:ident($($t:tt)*) ),* $(,)?) => {
            $(
                mod $name {
                    use super::super::*;

                    fn _run<B: Backend>(jit: &mut JitEvm<B>) {
                        run_case_built(tests!(@case $($t)*), jit)
                    }

                    with_matrix!(_run);
                }
            )*
        };

        (@case @raw { $($fields:tt)* }) => { &TestCase { $($fields)* ..Default::default() } };

        (@case $op:expr $(, $args:expr)* $(,)? => $($ret:expr),* $(,)? $(; op_gas($op_gas:expr))?) => {
            &TestCase {
                bytecode: &tests!(@bytecode $op, $($args),*),
                expected_stack: &[$($ret),*],
                expected_gas: tests!(@gas $op $(, $op_gas)?; $($args),*),
                ..Default::default()
            }
        };

        (@bytecode $op:expr, $a:expr) => { bytecode_unop($op, $a) };
        (@bytecode $op:expr, $a:expr, $b:expr) => { bytecode_binop($op, $a, $b) };
        (@bytecode $op:expr, $a:expr, $b:expr, $c:expr) => { bytecode_ternop($op, $a, $b, $c) };

        (@gas $op:expr; $($args:expr),+) => {
            tests!(@gas
                $op,
                DEF_OPINFOS[$op as usize].static_gas().expect(stringify!($op)) as u64;
                $($args),+
            )
        };
        (@gas $op:expr, $op_gas:expr; $($args:expr),+) => {
            $op_gas + tests!(@gas_base $($args),+)
        };
        (@gas_base $a:expr) => { 3 };
        (@gas_base $a:expr, $b:expr) => { 6 };
        (@gas_base $a:expr, $b:expr, $c:expr) => { 9 };
    }

    tests! {
        ret {
            empty(@raw {
                bytecode: &[],
                expected_gas: 0,
            }),
            no_stop(@raw {
                bytecode: &[op::PUSH0],
                expected_stack: &[U256::ZERO],
                expected_gas: 2,
            }),
            stop(@raw {
                bytecode: &[op::STOP],
                expected_gas: 0,
            }),
            invalid(@raw {
                bytecode: &[op::INVALID],
                expected_return: InstructionResult::InvalidFEOpcode,
                expected_gas: 0,
            }),
            unknown(@raw {
                bytecode: &[0x21],
                expected_return: InstructionResult::OpcodeNotFound,
                expected_gas: 0,
            }),
            underflow1(@raw {
                bytecode: &[op::ADD],
                expected_return: InstructionResult::StackUnderflow,
                expected_gas: 3,
            }),
            underflow2(@raw {
                bytecode: &[op::PUSH0, op::ADD],
                expected_return: InstructionResult::StackUnderflow,
                expected_stack: &[U256::ZERO],
                expected_gas: 5,
            }),
            underflow3(@raw {
                bytecode: &[op::PUSH0, op::POP, op::ADD],
                expected_return: InstructionResult::StackUnderflow,
                expected_gas: 7,
            }),
            underflow4(@raw {
                bytecode: &[op::PUSH0, op::ADD, op::POP],
                expected_return: InstructionResult::StackUnderflow,
                expected_stack: &[U256::ZERO],
                expected_gas: 5,
            }),
        }

        spec_id {
            push0_merge(@raw {
                bytecode: &[op::PUSH0],
                spec_id: SpecId::MERGE,
                expected_return: InstructionResult::NotActivated,
                expected_gas: 0,
            }),
            push0_shanghai(@raw {
                bytecode: &[op::PUSH0],
                spec_id: SpecId::SHANGHAI,
                expected_stack: &[U256::ZERO],
                expected_gas: 2,
            }),
            push0_cancun(@raw {
                bytecode: &[op::PUSH0],
                spec_id: SpecId::CANCUN,
                expected_stack: &[U256::ZERO],
                expected_gas: 2,
            }),
        }

        control_flow {
            basic_jump(@raw {
                bytecode: &[op::PUSH1, 3, op::JUMP, op::JUMPDEST],
                expected_gas: 3 + 8 + 1,
            }),
            unmodified_stack_after_push_jump(@raw {
                bytecode: &[op::PUSH1, 3, op::JUMP, op::JUMPDEST, op::PUSH0, op::ADD],
                expected_return: InstructionResult::StackUnderflow,
                expected_stack: &[U256::ZERO],
                expected_gas: 3 + 8 + 1 + 2 + 3,
            }),
            basic_jump_if(@raw {
                bytecode: &[op::PUSH1, 1, op::PUSH1, 5, op::JUMPI, op::JUMPDEST],
                expected_gas: 3 + 3 + 10 + 1,
            }),
            unmodified_stack_after_push_jumpif(@raw {
                bytecode: &[op::PUSH1, 1, op::PUSH1, 5, op::JUMPI, op::JUMPDEST, op::PUSH0, op::ADD],
                expected_return: InstructionResult::StackUnderflow,
                expected_stack: &[U256::ZERO],
                expected_gas: 3 + 3 + 10 + 1 + 2 + 3,
            }),
            basic_loop(@raw {
                bytecode: &[
                    op::PUSH1, 3,  // i=3
                    op::JUMPDEST,  // i
                    op::PUSH1, 1,  // 1, i
                    op::SWAP1,     // i, 1
                    op::SUB,       // i-1
                    op::DUP1,      // i-1, i-1
                    op::PUSH1, 2,  // dst, i-1, i-1
                    op::JUMPI,     // i=i-1
                    op::POP,       //
                    op::PUSH1, 69, // 69
                ],
                expected_stack: &[69_U256],
                expected_gas: 3 + (1 + 3 + 3 + 3 + 3 + 3 + 10) * 3 + 2 + 3,
            }),
            pc(@raw {
                bytecode: &[op::PC, op::PC, op::PUSH1, 69, op::PC, op::PUSH0, op::PC],
                expected_stack: &[0_U256, 1_U256, 69_U256, 4_U256, 0_U256, 6_U256],
                expected_gas: 2 + 2 + 3 + 2 + 2 + 2,
            }),
        }

        arith {
            add1(op::ADD, 0_U256, 0_U256 => 0_U256),
            add2(op::ADD, 1_U256, 2_U256 => 3_U256),
            add3(op::ADD, 255_U256, 255_U256 => 510_U256),
            add4(op::ADD, U256::MAX, 1_U256 => 0_U256),
            add5(op::ADD, U256::MAX, 2_U256 => 1_U256),

            sub1(op::SUB, 3_U256, 2_U256 => 1_U256),
            sub2(op::SUB, 1_U256, 2_U256 => -1_U256),
            sub3(op::SUB, 1_U256, 3_U256 => (-1_U256).wrapping_sub(1_U256)),
            sub4(op::SUB, 255_U256, 255_U256 => 0_U256),

            mul1(op::MUL, 1_U256, 2_U256 => 2_U256),
            mul2(op::MUL, 32_U256, 32_U256 => 1024_U256),
            mul3(op::MUL, U256::MAX, 2_U256 => U256::MAX.wrapping_sub(1_U256)),

            div1(op::DIV, 32_U256, 32_U256 => 1_U256),
            div2(op::DIV, 1_U256, 2_U256 => 0_U256),
            div3(op::DIV, 2_U256, 2_U256 => 1_U256),
            div4(op::DIV, 3_U256, 2_U256 => 1_U256),
            div5(op::DIV, 4_U256, 2_U256 => 2_U256),
            div_by_zero1(op::DIV, 0_U256, 0_U256 => 0_U256),
            div_by_zero2(op::DIV, 32_U256, 0_U256 => 0_U256),

            rem1(op::MOD, 32_U256, 32_U256 => 0_U256),
            rem2(op::MOD, 1_U256, 2_U256 => 1_U256),
            rem3(op::MOD, 2_U256, 2_U256 => 0_U256),
            rem4(op::MOD, 3_U256, 2_U256 => 1_U256),
            rem5(op::MOD, 4_U256, 2_U256 => 0_U256),
            rem_by_zero1(op::MOD, 0_U256, 0_U256 => 0_U256),
            rem_by_zero2(op::MOD, 32_U256, 0_U256 => 0_U256),

            sdiv1(op::SDIV, 32_U256, 32_U256 => 1_U256),
            sdiv2(op::SDIV, 1_U256, 2_U256 => 0_U256),
            sdiv3(op::SDIV, 2_U256, 2_U256 => 1_U256),
            sdiv4(op::SDIV, 3_U256, 2_U256 => 1_U256),
            sdiv5(op::SDIV, 4_U256, 2_U256 => 2_U256),
            sdiv_by_zero1(op::SDIV, 0_U256, 0_U256 => 0_U256),
            sdiv_by_zero2(op::SDIV, 32_U256, 0_U256 => 0_U256),
            sdiv_min_by_1(op::SDIV, I256_MIN, 1_U256 => -I256_MIN),
            sdiv_min_by_minus_1(op::SDIV, I256_MIN, -1_U256 => I256_MIN),
            sdiv_max1(op::SDIV, I256_MAX, 1_U256 => I256_MAX),
            sdiv_max2(op::SDIV, I256_MAX, -1_U256 => -I256_MAX),

            srem1(op::SMOD, 32_U256, 32_U256 => 0_U256),
            srem2(op::SMOD, 1_U256, 2_U256 => 1_U256),
            srem3(op::SMOD, 2_U256, 2_U256 => 0_U256),
            srem4(op::SMOD, 3_U256, 2_U256 => 1_U256),
            srem5(op::SMOD, 4_U256, 2_U256 => 0_U256),
            srem_by_zero1(op::SMOD, 0_U256, 0_U256 => 0_U256),
            srem_by_zero2(op::SMOD, 32_U256, 0_U256 => 0_U256),

            addmod1(op::ADDMOD, 1_U256, 2_U256, 3_U256 => 0_U256),
            addmod2(op::ADDMOD, 1_U256, 2_U256, 4_U256 => 3_U256),
            addmod3(op::ADDMOD, 1_U256, 2_U256, 2_U256 => 1_U256),
            addmod4(op::ADDMOD, 32_U256, 32_U256, 69_U256 => 64_U256),

            mulmod1(op::MULMOD, 0_U256, 0_U256, 1_U256 => 0_U256),
            mulmod2(op::MULMOD, 69_U256, 0_U256, 1_U256 => 0_U256),
            mulmod3(op::MULMOD, 0_U256, 1_U256, 2_U256 => 0_U256),
            mulmod4(op::MULMOD, 69_U256, 1_U256, 2_U256 => 1_U256),
            mulmod5(op::MULMOD, 69_U256, 1_U256, 30_U256 => 9_U256),
            mulmod6(op::MULMOD, 69_U256, 2_U256, 100_U256 => 38_U256),

            exp1(op::EXP, 0_U256, 0_U256 => 1_U256; op_gas(10)),
            exp2(op::EXP, 2_U256, 0_U256 => 1_U256; op_gas(10)),
            exp3(op::EXP, 2_U256, 1_U256 => 2_U256; op_gas(60)),
            exp4(op::EXP, 2_U256, 2_U256 => 4_U256; op_gas(60)),
            exp5(op::EXP, 2_U256, 3_U256 => 8_U256; op_gas(60)),
            exp6(op::EXP, 2_U256, 4_U256 => 16_U256; op_gas(60)),
            exp_overflow(op::EXP, 2_U256, 256_U256 => 0_U256; op_gas(110)),

            signextend1(op::SIGNEXTEND, 0_U256, 0_U256 => 0_U256),
            signextend2(op::SIGNEXTEND, 1_U256, 0_U256 => 0_U256),
            signextend3(op::SIGNEXTEND, 0_U256, -1_U256 => -1_U256),
            signextend4(op::SIGNEXTEND, 1_U256, -1_U256 => -1_U256),
            signextend5(op::SIGNEXTEND, 0_U256, 0x7f_U256 => 0x7f_U256),
            signextend6(op::SIGNEXTEND, 0_U256, 0x80_U256 => -0x80_U256),
            signextend7(op::SIGNEXTEND, 0_U256, 0xff_U256 => U256::MAX),
            signextend8(op::SIGNEXTEND, 1_U256, 0x7fff_U256 => 0x7fff_U256),
            signextend8_extra(op::SIGNEXTEND, 1_U256, 0xff7fff_U256 => 0x7fff_U256),
            signextend9(op::SIGNEXTEND, 1_U256, 0x8000_U256 => -0x8000_U256),
            signextend9_extra(op::SIGNEXTEND, 1_U256, 0x118000_U256 => -0x8000_U256),
            signextend10(op::SIGNEXTEND, 1_U256, 0xffff_U256 => U256::MAX),
        }

        cmp {
            lt1(op::LT, 1_U256, 2_U256 => 1_U256),
            lt2(op::LT, 2_U256, 1_U256 => 0_U256),
            lt3(op::LT, 1_U256, 1_U256 => 0_U256),
            lt4(op::LT, -1_U256, 1_U256 => 0_U256),

            gt1(op::GT, 1_U256, 2_U256 => 0_U256),
            gt2(op::GT, 2_U256, 1_U256 => 1_U256),
            gt3(op::GT, 1_U256, 1_U256 => 0_U256),
            gt4(op::GT, -1_U256, 1_U256 => 1_U256),

            slt1(op::SLT, 1_U256, 2_U256 => 1_U256),
            slt2(op::SLT, 2_U256, 1_U256 => 0_U256),
            slt3(op::SLT, 1_U256, 1_U256 => 0_U256),
            slt4(op::SLT, -1_U256, 1_U256 => 1_U256),

            sgt1(op::SGT, 1_U256, 2_U256 => 0_U256),
            sgt2(op::SGT, 2_U256, 1_U256 => 1_U256),
            sgt3(op::SGT, 1_U256, 1_U256 => 0_U256),
            sgt4(op::SGT, -1_U256, 1_U256 => 0_U256),

            eq1(op::EQ, 1_U256, 2_U256 => 0_U256),
            eq2(op::EQ, 2_U256, 1_U256 => 0_U256),
            eq3(op::EQ, 1_U256, 1_U256 => 1_U256),

            iszero1(op::ISZERO, 0_U256 => 1_U256),
            iszero2(op::ISZERO, 1_U256 => 0_U256),
            iszero3(op::ISZERO, 2_U256 => 0_U256),
        }

        bitwise {
            and1(op::AND, 0_U256, 0_U256 => 0_U256),
            and2(op::AND, 1_U256, 1_U256 => 1_U256),
            and3(op::AND, 1_U256, 2_U256 => 0_U256),
            and4(op::AND, 255_U256, 255_U256 => 255_U256),

            or1(op::OR, 0_U256, 0_U256 => 0_U256),
            or2(op::OR, 1_U256, 2_U256 => 3_U256),
            or3(op::OR, 1_U256, 3_U256 => 3_U256),
            or4(op::OR, 2_U256, 2_U256 => 2_U256),

            xor1(op::XOR, 0_U256, 0_U256 => 0_U256),
            xor2(op::XOR, 1_U256, 2_U256 => 3_U256),
            xor3(op::XOR, 1_U256, 3_U256 => 2_U256),
            xor4(op::XOR, 2_U256, 2_U256 => 0_U256),

            not1(op::NOT, 0_U256 => U256::MAX),
            not2(op::NOT, U256::MAX => 0_U256),
            not3(op::NOT, 1_U256 => U256::MAX.wrapping_sub(1_U256)),

            byte1(op::BYTE, 0_U256, 0x1122334400000000000000000000000000000000000000000000000000000000_U256 => 0x11_U256),
            byte2(op::BYTE, 1_U256, 0x1122334400000000000000000000000000000000000000000000000000000000_U256 => 0x22_U256),
            byte3(op::BYTE, 2_U256, 0x1122334400000000000000000000000000000000000000000000000000000000_U256 => 0x33_U256),
            byte4(op::BYTE, 3_U256, 0x1122334400000000000000000000000000000000000000000000000000000000_U256 => 0x44_U256),
            byte5(op::BYTE, 4_U256, 0x1122334400000000000000000000000000000000000000000000000000000000_U256 => 0x00_U256),
            byte_oob0(op::BYTE, 31_U256, U256::MAX => 0xFF_U256),
            byte_oob1(op::BYTE, 32_U256, U256::MAX => 0_U256),
            byte_oob2(op::BYTE, 33_U256, U256::MAX => 0_U256),

            // shift operand order is reversed for some reason:
            // shift, x
            shl1(op::SHL, 0_U256, 1_U256 => 1_U256),
            shl2(op::SHL, 1_U256, 1_U256 => 2_U256),
            shl3(op::SHL, 2_U256, 1_U256 => 4_U256),

            shr1(op::SHR, 0_U256, 1_U256 => 1_U256),
            shr2(op::SHR, 1_U256, 2_U256 => 1_U256),
            shr3(op::SHR, 2_U256, 4_U256 => 1_U256),

            sar1(op::SAR, 0_U256, 1_U256 => 1_U256),
            sar2(op::SAR, 1_U256, 2_U256 => 1_U256),
            sar3(op::SAR, 2_U256, 4_U256 => 1_U256),
            sar4(op::SAR, 1_U256, -1_U256 => -1_U256),
            sar5(op::SAR, 2_U256, -1_U256 => -1_U256),
        }

        system {
            gas(@raw {
                bytecode: &[op::GAS, op::GAS, op::JUMPDEST, op::GAS],
                expected_stack: &[DEF_GAS_LIMIT_U256 - 2_U256, DEF_GAS_LIMIT_U256 - 4_U256, DEF_GAS_LIMIT_U256 - 7_U256],
                expected_gas: 2 + 2 + 1 + 2,
            }),
            keccak256_empty1(@raw {
                bytecode: &[op::PUSH0, op::PUSH0, op::KECCAK256],
                expected_stack: &[KECCAK_EMPTY.into()],
                expected_gas: 2 + 2 + gas::KECCAK256,
            }),
            keccak256_empty2(@raw {
                bytecode: &[op::PUSH0, op::PUSH1, 32, op::KECCAK256],
                expected_stack: &[KECCAK_EMPTY.into()],
                expected_gas: 2 + 3 + gas::KECCAK256,
            }),
            keccak256_1(@raw {
                bytecode: &[op::PUSH1, 32, op::PUSH0, op::KECCAK256],
                expected_stack: &[keccak256([0; 32]).into()],
                expected_memory: &[0; 32],
                expected_gas: 3 + 2 + (gas::keccak256_cost(32).unwrap() + 3),
            }),
            keccak256_2(@raw {
                bytecode: &[op::PUSH2, 0x69, 0x42, op::PUSH0, op::MSTORE, op::PUSH1, 0x20, op::PUSH0, op::KECCAK256],
                expected_stack: &[keccak256(0x6942_U256.to_be_bytes::<32>()).into()],
                expected_memory: &0x6942_U256.to_be_bytes::<32>(),
                expected_gas: 3 + 2 + (3 + 3) + 3 + 2 + gas::keccak256_cost(32).unwrap(),
            }),

            address(@raw {
                bytecode: &[op::ADDRESS, op::ADDRESS],
                expected_stack: &[DEF_ADDR.into_word().into(), DEF_ADDR.into_word().into()],
                expected_gas: 4,
            }),
            origin(@raw {
                bytecode: &[op::ORIGIN, op::ORIGIN],
                expected_stack: &[def_env().tx.caller.into_word().into(), def_env().tx.caller.into_word().into()],
                expected_gas: 4,
            }),
            caller(@raw {
                bytecode: &[op::CALLER, op::CALLER],
                expected_stack: &[DEF_CALLER.into_word().into(), DEF_CALLER.into_word().into()],
                expected_gas: 4,
            }),
            callvalue(@raw {
                bytecode: &[op::CALLVALUE, op::CALLVALUE],
                expected_stack: &[DEF_VALUE, DEF_VALUE],
                expected_gas: 4,
            }),
        }

        calldata {
            calldataload1(@raw {
                bytecode: &[op::PUSH0, op::CALLDATALOAD],
                expected_stack: &[U256::from_be_slice(&DEF_CD[..32])],
                expected_gas: 2 + 3,
            }),
            calldataload2(@raw {
                bytecode: &[op::PUSH1, 63, op::CALLDATALOAD],
                expected_stack: &[0xaa00000000000000000000000000000000000000000000000000000000000000_U256],
                expected_gas: 3 + 3,
            }),
            calldatasize(@raw {
                bytecode: &[op::CALLDATASIZE, op::CALLDATASIZE],
                expected_stack: &[U256::from(DEF_CD.len()), U256::from(DEF_CD.len())],
                expected_gas: 2 + 2,
            }),
            calldatacopy(@raw {
                bytecode: &[op::PUSH1, 32, op::PUSH0, op::PUSH0, op::CALLDATACOPY],
                expected_memory: &DEF_CD[..32],
                expected_gas: 3 + 2 + 2 + (gas::verylowcopy_cost(32).unwrap() + 3),
            }),
        }

        code {
            codesize(@raw {
                bytecode: &[op::CODESIZE, op::CODESIZE],
                expected_stack: &[2_U256, 2_U256],
                expected_gas: 2 + 2,
            }),
            codecopy(@raw {
                bytecode: &[op::PUSH1, 5, op::PUSH0, op::PUSH0, op::CODECOPY],
                expected_memory: &hex!("60055f5f39000000000000000000000000000000000000000000000000000000"),
                expected_gas: 3 + 2 + 2 + (gas::verylowcopy_cost(32).unwrap() + gas::memory_gas(1)),
            }),
        }

        returndata {
            returndatasize(@raw {
                bytecode: &[op::RETURNDATASIZE, op::RETURNDATASIZE],
                expected_stack: &[64_U256, 64_U256],
                expected_gas: 2 + 2,
            }),
            returndatacopy(@raw {
                bytecode: &[op::PUSH1, 32, op::PUSH0, op::PUSH0, op::RETURNDATACOPY],
                expected_memory: &DEF_RD[..32],
                expected_gas: 3 + 2 + 2 + (gas::verylowcopy_cost(32).unwrap() + gas::memory_gas(1)),
            }),
        }

        extcode {
            extcodesize1(op::EXTCODESIZE, DEF_ADDR.into_word().into() => 0_U256;
                op_gas(100)),
            extcodesize2(op::EXTCODESIZE, OTHER_ADDR.into_word().into() => U256::from(def_codemap()[&OTHER_ADDR].len());
                op_gas(100)),
            extcodecopy1(@raw {
                bytecode: &[op::PUSH0, op::PUSH0, op::PUSH0, op::PUSH0, op::EXTCODECOPY],
                expected_memory: &[],
                expected_gas: 2 + 2 + 2 + 2 + 100,
            }),
            extcodecopy2(@raw {
                // bytecode: &[op::PUSH1, 64, op::PUSH0, op::PUSH0, op::PUSH20, OTHER_ADDR, op::EXTCODECOPY],
                bytecode: &hex!("6040 5f 5f 736969696969696969696969696969696969696969 3c"),
                expected_memory: &{
                    let mut mem = [0; 64];
                    let code = def_codemap()[&OTHER_ADDR].original_bytes();
                    mem[..code.len()].copy_from_slice(&code);
                    mem
                },
                expected_gas: 3 + 2 + 2 + 3 + (100 + 12),
            }),
            extcodehash1(op::EXTCODEHASH, DEF_ADDR.into_word().into() => KECCAK_EMPTY.into();
                op_gas(100)),
            extcodehash2(op::EXTCODEHASH, OTHER_ADDR.into_word().into() => def_codemap()[&OTHER_ADDR].hash_slow().into();
                op_gas(100)),
        }

        env {
            gas_price(@raw {
                bytecode: &[op::GASPRICE],
                expected_stack: &[def_env().tx.gas_price],
                expected_gas: 2,
            }),
            blockhash0(op::BLOCKHASH, DEF_BN - 0_U256 => 0_U256),
            blockhash1(op::BLOCKHASH, DEF_BN - 1_U256 => DEF_BN - 1_U256),
            blockhash2(op::BLOCKHASH, DEF_BN - 255_U256 => DEF_BN - 255_U256),
            blockhash3(op::BLOCKHASH, DEF_BN - 256_U256 => DEF_BN - 256_U256),
            blockhash4(op::BLOCKHASH, DEF_BN - 257_U256 => 0_U256),
            coinbase(@raw {
                bytecode: &[op::COINBASE, op::COINBASE],
                expected_stack: &[def_env().block.coinbase.into_word().into(), def_env().block.coinbase.into_word().into()],
                expected_gas: 4,
            }),
            timestamp(@raw {
                bytecode: &[op::TIMESTAMP, op::TIMESTAMP],
                expected_stack: &[def_env().block.timestamp, def_env().block.timestamp],
                expected_gas: 4,
            }),
            number(@raw {
                bytecode: &[op::NUMBER, op::NUMBER],
                expected_stack: &[def_env().block.number, def_env().block.number],
                expected_gas: 4,
            }),
            difficulty(@raw {
                bytecode: &[op::DIFFICULTY, op::DIFFICULTY],
                spec_id: SpecId::GRAY_GLACIER,
                expected_stack: &[def_env().block.difficulty, def_env().block.difficulty],
                expected_gas: 4,
            }),
            difficulty_prevrandao(@raw {
                bytecode: &[op::DIFFICULTY, op::DIFFICULTY],
                spec_id: SpecId::MERGE,
                expected_stack: &[def_env().block.prevrandao.unwrap().into(), def_env().block.prevrandao.unwrap().into()],
                expected_gas: 4,
            }),
            gaslimit(@raw {
                bytecode: &[op::GASLIMIT, op::GASLIMIT],
                expected_stack: &[def_env().block.gas_limit, def_env().block.gas_limit],
                expected_gas: 4,
            }),
            chainid(@raw {
                bytecode: &[op::CHAINID, op::CHAINID],
                expected_stack: &[U256::from(def_env().cfg.chain_id), U256::from(def_env().cfg.chain_id)],
                expected_gas: 4,
            }),
            selfbalance(@raw {
                bytecode: &[op::SELFBALANCE, op::SELFBALANCE],
                expected_stack: &[0xba_U256, 0xba_U256],
                expected_gas: 10,
            }),
            basefee(@raw {
                bytecode: &[op::BASEFEE, op::BASEFEE],
                expected_stack: &[def_env().block.basefee, def_env().block.basefee],
                expected_gas: 4,
            }),
            blobhash0(@raw {
                bytecode: &[op::PUSH0, op::BLOBHASH],
                expected_stack: &[def_env().tx.blob_hashes[0].into()],
                expected_gas: 2 + 3,
            }),
            blobhash1(@raw {
                bytecode: &[op::PUSH1, 1, op::BLOBHASH],
                expected_stack: &[def_env().tx.blob_hashes[1].into()],
                expected_gas: 3 + 3,
            }),
            blobhash2(@raw {
                bytecode: &[op::PUSH1, 2, op::BLOBHASH],
                expected_stack: &[0_U256],
                expected_gas: 3 + 3,
            }),
            blobbasefee(@raw {
                bytecode: &[op::BLOBBASEFEE, op::BLOBBASEFEE],
                expected_stack: &[U256::from(def_env().block.get_blob_gasprice().unwrap()), U256::from(def_env().block.get_blob_gasprice().unwrap())],
                expected_gas: 4,
            }),
        }

        memory {
            mload1(@raw {
                bytecode: &[op::PUSH0, op::MLOAD],
                expected_stack: &[0_U256],
                expected_memory: &[0; 32],
                expected_gas: 2 + (3 + gas::memory_gas(1)),
            }),
            mload2(@raw {
                bytecode: &[op::PUSH1, 1, op::MLOAD],
                expected_stack: &[0_U256],
                expected_memory: &[0; 64],
                expected_gas: 3 + (3 + gas::memory_gas(2)),
            }),
            mload3(@raw {
                bytecode: &[op::PUSH1, 32, op::MLOAD],
                expected_stack: &[0_U256],
                expected_memory: &[0; 64],
                expected_gas: 3 + (3 + gas::memory_gas(2)),
            }),
            mload4(@raw {
                bytecode: &[op::PUSH1, 33, op::MLOAD],
                expected_stack: &[0_U256],
                expected_memory: &[0; 96],
                expected_gas: 3 + (3 + gas::memory_gas(3)),
            }),
            mstore1(@raw {
                bytecode: &[op::PUSH0, op::PUSH0, op::MSTORE],
                expected_memory: &[0; 32],
                expected_gas: 2 + 2 + (3 + gas::memory_gas(1)),
            }),
            mstore8_1(@raw {
                bytecode: &[op::PUSH0, op::PUSH0, op::MSTORE8],
                expected_memory: &[0; 32],
                expected_gas: 2 + 2 + (3 + gas::memory_gas(1)),
            }),
            mstore8_2(@raw {
                bytecode: &[op::PUSH2, 0x69, 0x69, op::PUSH0, op::MSTORE8],
                expected_memory: &{
                    let mut mem = [0; 32];
                    mem[0] = 0x69;
                    mem
                },
                expected_gas: 3 + 2 + (3 + gas::memory_gas(1)),
            }),
            msize1(@raw {
                bytecode: &[op::MSIZE, op::MSIZE],
                expected_stack: &[0_U256, 0_U256],
                expected_gas: 2 + 2,
            }),
            msize2(@raw {
                bytecode: &[op::MSIZE, op::PUSH0, op::MLOAD, op::POP, op::MSIZE, op::PUSH1, 1, op::MLOAD, op::POP, op::MSIZE],
                expected_stack: &[0_U256, 32_U256, 64_U256],
                expected_memory: &[0; 64],
                expected_gas: 2 + 2 + (3 + gas::memory_gas(1)) + 2 + 2 + 3 + (3 + (gas::memory_gas(2) - gas::memory_gas(1))) + 2 + 2,
            }),
        }

        host {
            balance(op::BALANCE, 0_U256 => 0_U256; op_gas(100)),
            sload1(@raw {
                bytecode: &[op::PUSH1, 69, op::SLOAD],
                expected_stack: &[42_U256],
                expected_gas: 3 + 100,
            }),
            sload2(@raw {
                bytecode: &[op::PUSH1, 70, op::SLOAD],
                expected_stack: &[0_U256],
                expected_gas: 3 + 2100,
            }),
            sload3(@raw {
                bytecode: &[op::PUSH1, 0xff, op::SLOAD],
                expected_stack: &[0_U256],
                expected_gas: 3 + 2100,
            }),
            sstore1(@raw {
                bytecode: &[op::PUSH1, 200, op::SLOAD, op::PUSH1, 100, op::PUSH1, 200, op::SSTORE, op::PUSH1, 200, op::SLOAD],
                expected_stack: &[0_U256, 100_U256],
                expected_gas: GAS_WHAT_THE_INTERPRETER_SAYS,
                assert_host: Some(|host| {
                    assert_eq!(host.storage.get(&200_U256), Some(&100_U256));
                }),
            }),
            tload(@raw {
                bytecode: &[op::PUSH1, 69, op::TLOAD],
                expected_stack: &[0_U256],
                expected_gas: 3 + 100,
                assert_host: Some(|host| {
                    assert!(host.transient_storage.is_empty());
                }),
            }),
            tstore(@raw {
                bytecode: &[op::PUSH1, 69, op::TLOAD, op::PUSH1, 42, op::PUSH1, 69, op::TSTORE, op::PUSH1, 69, op::TLOAD],
                expected_stack: &[0_U256, 42_U256],
                expected_gas: 3 + 100 + 3 + 3 + 100 + 3 + 100,
                assert_host: Some(|host| {
                    assert_eq!(host.transient_storage.get(&69_U256), Some(&42_U256));
                }),
            }),
            log0(@raw {
                bytecode: &[op::PUSH0, op::PUSH0, op::LOG0],
                expected_gas: 2 + 2 + gas::log_cost(0, 0).unwrap(),
                assert_host: Some(|host| {
                    assert_eq!(host.log, [primitives::Log {
                        address: DEF_ADDR,
                        data: LogData::new(vec![], Bytes::new()).unwrap(),
                    }]);
                }),
            }),
            log0_data(@raw {
                bytecode: &[op::PUSH2, 0x69, 0x42, op::PUSH0, op::MSTORE, op::PUSH1, 32, op::PUSH0, op::LOG0],
                expected_memory: &0x6942_U256.to_be_bytes::<32>(),
                expected_gas: 3 + 2 + (3 + gas::memory_gas(1)) + 3 + 2 + gas::log_cost(0, 32).unwrap(),
                assert_host: Some(|host| {
                    assert_eq!(host.log, [primitives::Log {
                        address: DEF_ADDR,
                        data: LogData::new(vec![], Bytes::copy_from_slice(&0x6942_U256.to_be_bytes::<32>())).unwrap(),
                    }]);
                }),
            }),
            log1(@raw {
                bytecode: &[op::PUSH0, op::PUSH0, op::PUSH0, op::LOG1],
                expected_gas: 2 + 2 + 2 + gas::log_cost(1, 0).unwrap(),
                assert_host: Some(|host| {
                    assert_eq!(host.log, [primitives::Log {
                        address: DEF_ADDR,
                        data: LogData::new(vec![B256::ZERO], Bytes::new()).unwrap(),
                    }]);
                }),
            }),
            // TODO: create
            // TODO: call
            // TODO: callcode
            // TODO: return
            // TODO: delegatecall
            // TODO: create2
            // TODO: staticcall
            // TODO: revert
            // TODO: selfdestruct
        }
    }

    struct TestCase<'a> {
        bytecode: &'a [u8],
        spec_id: SpecId,

        expected_return: InstructionResult,
        expected_stack: &'a [U256],
        expected_memory: &'a [u8],
        expected_gas: u64,
        assert_host: Option<fn(&mut TestHost)>,
    }

    impl Default for TestCase<'_> {
        fn default() -> Self {
            Self {
                bytecode: &[],
                spec_id: DEF_SPEC,
                expected_return: InstructionResult::Stop,
                expected_stack: &[],
                expected_memory: &[],
                expected_gas: 0,
                assert_host: None,
            }
        }
    }

    impl fmt::Debug for TestCase<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("TestCase")
                .field("bytecode", &format_bytecode(self.bytecode))
                .field("spec_id", &self.spec_id)
                .field("expected_return", &self.expected_return)
                .field("expected_stack", &self.expected_stack)
                .field("expected_memory", &MemDisplay(self.expected_memory))
                .field("expected_gas", &self.expected_gas)
                .field("assert_host", &self.assert_host.is_some())
                .finish()
        }
    }

    /* ----------------------------------------- runner ----------------------------------------- */

    // Default values.
    const DEF_SPEC: SpecId = SpecId::CANCUN;
    const DEF_OPINFOS: &[OpcodeInfo; 256] = op_info_map(DEF_SPEC);

    const DEF_GAS_LIMIT: u64 = 100_000;
    const DEF_GAS_LIMIT_U256: U256 = U256::from_le_slice(&DEF_GAS_LIMIT.to_le_bytes());

    /// Default code address.
    const DEF_ADDR: Address = Address::repeat_byte(0xba);
    const DEF_CALLER: Address = Address::repeat_byte(0xca);
    static DEF_CD: &[u8] = &[0xaa; 64];
    static DEF_RD: &[u8] = &[0xbb; 64];
    const DEF_VALUE: U256 = uint!(123_456_789_U256);
    static DEF_ENV: OnceLock<Env> = OnceLock::new();
    static DEF_STORAGE: OnceLock<HashMap<U256, U256>> = OnceLock::new();
    static DEF_CODEMAP: OnceLock<HashMap<Address, primitives::Bytecode>> = OnceLock::new();
    const OTHER_ADDR: Address = Address::repeat_byte(0x69);
    const DEF_BN: U256 = uint!(500_U256);

    const GAS_WHAT_THE_INTERPRETER_SAYS: u64 = u64::MAX - 1000;

    fn def_env() -> &'static Env {
        DEF_ENV.get_or_init(|| Env {
            cfg: {
                let mut cfg = CfgEnv::default();
                cfg.chain_id = 69;
                cfg
            },
            block: BlockEnv {
                number: DEF_BN,
                coinbase: Address::repeat_byte(0xcb),
                timestamp: U256::from(2),
                gas_limit: U256::from(3),
                basefee: U256::from(4),
                difficulty: U256::from(5),
                prevrandao: Some(U256::from(6).into()),
                blob_excess_gas_and_price: Some(BlobExcessGasAndPrice::new(50)),
            },
            tx: TxEnv {
                caller: Address::repeat_byte(0xcc),
                gas_limit: DEF_GAS_LIMIT,
                gas_price: U256::from(7),
                transact_to: primitives::TransactTo::Call(DEF_ADDR),
                value: DEF_VALUE,
                data: DEF_CD.into(),
                nonce: None,
                chain_id: Some(420), // Different from `cfg.chain_id`.
                access_list: vec![],
                gas_priority_fee: None,
                blob_hashes: vec![B256::repeat_byte(0xb7), B256::repeat_byte(0xb8)],
                max_fee_per_blob_gas: None,
            },
        })
    }

    fn def_storage() -> &'static HashMap<U256, U256> {
        DEF_STORAGE.get_or_init(|| {
            HashMap::from([
                (U256::from(0), U256::from(1)),
                (U256::from(1), U256::from(2)),
                (U256::from(69), U256::from(42)),
            ])
        })
    }

    fn def_codemap() -> &'static HashMap<Address, primitives::Bytecode> {
        DEF_CODEMAP.get_or_init(|| {
            HashMap::from([
                //
                (
                    OTHER_ADDR,
                    primitives::Bytecode::new_raw(Bytes::from_static(&[
                        op::PUSH1,
                        0x69,
                        op::PUSH1,
                        0x42,
                        op::ADD,
                        op::STOP,
                    ])),
                ),
            ])
        })
    }

    /// Wrapper around `DummyHost` that provides a stable environment and storage for testing.
    struct TestHost {
        host: DummyHost,
        code_map: &'static HashMap<Address, primitives::Bytecode>,
    }

    impl TestHost {
        fn new() -> Self {
            Self {
                host: DummyHost {
                    env: def_env().clone(),
                    storage: def_storage().clone(),
                    transient_storage: HashMap::new(),
                    log: Vec::new(),
                },
                code_map: def_codemap(),
            }
        }
    }

    impl std::ops::Deref for TestHost {
        type Target = DummyHost;

        fn deref(&self) -> &Self::Target {
            &self.host
        }
    }

    impl std::ops::DerefMut for TestHost {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.host
        }
    }

    impl Host for TestHost {
        fn env(&self) -> &Env {
            self.host.env()
        }

        fn env_mut(&mut self) -> &mut Env {
            self.host.env_mut()
        }

        fn load_account(&mut self, address: Address) -> Option<(bool, bool)> {
            self.host.load_account(address)
        }

        fn block_hash(&mut self, number: U256) -> Option<B256> {
            Some(number.into())
        }

        fn balance(&mut self, address: Address) -> Option<(U256, bool)> {
            Some((U256::from(*address.last().unwrap()), false))
        }

        fn code(&mut self, address: Address) -> Option<(primitives::Bytecode, bool)> {
            self.code_map
                .get(&address)
                .map(|b| (b.clone(), false))
                .or(Some((primitives::Bytecode::new(), false)))
        }

        fn code_hash(&mut self, address: Address) -> Option<(B256, bool)> {
            self.code_map
                .get(&address)
                .map(|b| (b.hash_slow(), false))
                .or(Some((KECCAK_EMPTY, false)))
        }

        fn sload(&mut self, address: Address, index: U256) -> Option<(U256, bool)> {
            self.host.sload(address, index)
        }

        fn sstore(
            &mut self,
            address: Address,
            index: U256,
            value: U256,
        ) -> Option<interpreter::SStoreResult> {
            self.host.sstore(address, index, value)
        }

        fn tload(&mut self, address: Address, index: U256) -> U256 {
            self.host.tload(address, index)
        }

        fn tstore(&mut self, address: Address, index: U256, value: U256) {
            self.host.tstore(address, index, value)
        }

        fn log(&mut self, log: primitives::Log) {
            self.host.log(log)
        }

        fn selfdestruct(
            &mut self,
            _address: Address,
            _target: Address,
        ) -> Option<interpreter::SelfDestructResult> {
            Some(interpreter::SelfDestructResult {
                had_value: false,
                target_exists: true,
                is_cold: false,
                previously_destroyed: false,
            })
        }
    }

    fn with_evm_context<F: FnOnce(&mut EvmContext<'_>) -> R, R>(bytecode: &[u8], f: F) -> R {
        let contract = Contract {
            input: Bytes::from_static(DEF_CD),
            bytecode: revm_interpreter::analysis::to_analysed(revm_primitives::Bytecode::new_raw(
                Bytes::copy_from_slice(bytecode),
            ))
            .try_into()
            .unwrap(),
            hash: keccak256(bytecode),
            address: DEF_ADDR,
            caller: DEF_CALLER,
            value: DEF_VALUE,
        };

        let mut interpreter = revm_interpreter::Interpreter::new(contract, DEF_GAS_LIMIT, false);
        interpreter.return_data_buffer = Bytes::from_static(DEF_RD);

        let mut host = TestHost::new();

        f(&mut EvmContext::from_interpreter(&mut interpreter, &mut host))
    }

    #[cfg(feature = "llvm")]
    fn with_llvm_context(f: impl FnOnce(&LlvmContext)) {
        thread_local! {
            static TLS_LLVM_CONTEXT: LlvmContext = LlvmContext::create();
        }

        TLS_LLVM_CONTEXT.with(f);
    }

    #[cfg(feature = "llvm")]
    fn with_llvm_backend(opt_level: OptimizationLevel, f: impl FnOnce(JitEvmLlvmBackend<'_>)) {
        with_llvm_context(|cx| f(new_llvm_backend(cx, opt_level).unwrap()))
    }

    #[cfg(feature = "llvm")]
    fn with_llvm_backend_jit(
        opt_level: OptimizationLevel,
        f: fn(&mut JitEvm<JitEvmLlvmBackend<'_>>),
    ) {
        with_llvm_backend(opt_level, |backend| f(&mut JitEvm::new(backend)));
    }

    fn set_test_dump<B: Backend>(jit: &mut JitEvm<B>, module_path: &str) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
        let mut dump_path = root.to_path_buf();
        dump_path.push("target");
        dump_path.push("tests_dump");
        // Skip `revm_jit::compiler::tests`.
        for part in module_path.split("::").skip(3) {
            dump_path.push(part);
        }
        dump_path.push(format!("{:?}", jit.opt_level()));
        jit.set_dump_to(Some(dump_path));
    }

    fn run_case_built<B: Backend>(test_case: &TestCase<'_>, jit: &mut JitEvm<B>) {
        let TestCase {
            bytecode,
            spec_id,
            expected_return,
            expected_stack,
            expected_memory,
            expected_gas,
            assert_host,
        } = *test_case;
        jit.set_inspect_stack_length(true);
        let f = jit.compile(None, bytecode, spec_id).unwrap();

        let mut stack = EvmStack::new();
        let mut stack_len = 0;
        with_evm_context(bytecode, |ecx| {
            // Interpreter.
            let table =
                spec_to_generic!(test_case.spec_id, op::make_instruction_table::<_, SPEC>());
            let mut interpreter = ecx.to_interpreter(Default::default());
            let memory = interpreter.take_memory();
            let mut int_host = TestHost::new();
            interpreter.run(memory, &table, &mut int_host);
            assert_eq!(
                interpreter.instruction_result, expected_return,
                "interpreter return value mismatch"
            );
            assert_eq!(interpreter.stack.data(), expected_stack, "interpreter stack mismatch");
            assert_eq!(
                MemDisplay(interpreter.shared_memory.context_memory()),
                MemDisplay(expected_memory),
                "interpreter memory mismatch"
            );
            let mut expected_gas = expected_gas;
            if expected_gas == GAS_WHAT_THE_INTERPRETER_SAYS {
                println!("asked for interpreter gas: {}", interpreter.gas.spent());
                expected_gas = interpreter.gas.spent();
            } else {
                assert_eq!(interpreter.gas.spent(), expected_gas, "interpreter gas mismatch");
            }
            if let Some(assert_host) = assert_host {
                assert_host(&mut int_host);
            }

            // JIT.
            let actual_return = unsafe { f.call(Some(&mut stack), Some(&mut stack_len), ecx) };
            assert_eq!(actual_return, expected_return, "return value mismatch");
            let actual_stack =
                stack.as_slice().iter().take(stack_len).map(|x| x.to_u256()).collect::<Vec<_>>();
            assert_eq!(actual_stack, *expected_stack, "stack mismatch");
            assert_eq!(
                MemDisplay(ecx.memory.context_memory()),
                MemDisplay(expected_memory),
                "interpreter memory mismatch"
            );
            assert_eq!(ecx.gas.spent(), expected_gas, "gas mismatch");
            if let Some(assert_host) = assert_host {
                assert_host(ecx.host.downcast_mut().unwrap());
            }
        });
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct MemDisplay<'a>(&'a [u8]);
    impl fmt::Debug for MemDisplay<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let chunks = self.0.chunks(32).map(revm_primitives::hex::encode_prefixed);
            f.debug_list().entries(chunks).finish()
        }
    }

    // ---

    mod fibonacci {
        use super::*;

        with_matrix!(run_fibonacci_tests);
    }

    fn run_fibonacci_tests<B: Backend>(jit: &mut JitEvm<B>) {
        jit.set_inspect_stack_length(true);

        for i in 0..=10 {
            run_fibonacci_test(jit, i);
        }
        run_fibonacci_test(jit, 100);

        fn run_fibonacci_test<B: Backend>(jit: &mut JitEvm<B>, input: u16) {
            println!("  Running fibonacci({input}) statically");
            run_fibonacci(jit, input, false);
            println!("  Running fibonacci({input}) dynamically");
            run_fibonacci(jit, input, true);
        }

        fn run_fibonacci<B: Backend>(jit: &mut JitEvm<B>, input: u16, dynamic: bool) {
            let code = mk_fibonacci_code(input, dynamic);

            unsafe { jit.free_all_functions() }.unwrap();
            let f = jit.compile(None, &code, DEF_SPEC).unwrap();

            let mut stack_buf = EvmStack::new_heap();
            let stack = EvmStack::from_mut_vec(&mut stack_buf);
            if dynamic {
                stack.as_mut_slice()[0] = U256::from(input).into();
            }
            let mut stack_len = 0;
            if dynamic {
                stack_len = 1;
            }
            with_evm_context(&code, |ecx| {
                let r = unsafe { f.call(Some(stack), Some(&mut stack_len), ecx) };
                assert_eq!(r, InstructionResult::Stop);
                // Apparently the code does `fibonacci(input - 1)`.
                assert_eq!(stack.as_slice()[0].to_u256(), fibonacci_rust(input + 1));
                assert_eq!(stack_len, 1);
            });
        }

        fn mk_fibonacci_code(input: u16, dynamic: bool) -> Vec<u8> {
            if dynamic {
                [&[op::JUMPDEST; 3][..], FIBONACCI_CODE].concat()
            } else {
                let input = input.to_be_bytes();
                [[op::PUSH2].as_slice(), input.as_slice(), FIBONACCI_CODE].concat()
            }
        }

        // Modified from jitevm: https://github.com/paradigmxyz/jitevm/blob/f82261fc8a1a6c1a3d40025a910ba0ce3fcaed71/src/test_data.rs#L3
        #[rustfmt::skip]
        const FIBONACCI_CODE: &[u8] = &[
            // 1st/2nd fib number
            op::PUSH1, 0,
            op::PUSH1, 1,
            // 7

            // MAINLOOP:
            op::JUMPDEST,
            op::DUP3,
            op::ISZERO,
            op::PUSH1, 28, // cleanup
            op::JUMPI,

            // fib step
            op::DUP2,
            op::DUP2,
            op::ADD,
            op::SWAP2,
            op::POP,
            op::SWAP1,
            // 19

            // decrement fib step counter
            op::SWAP2,
            op::PUSH1, 1,
            op::SWAP1,
            op::SUB,
            op::SWAP2,
            op::PUSH1, 7, // goto MAINLOOP
            op::JUMP,
            // 28

            // CLEANUP:
            op::JUMPDEST,
            op::SWAP2,
            op::POP,
            op::POP,
            // done: requested fib number is the only element on the stack!
            op::STOP,
        ];
    }

    fn fibonacci_rust(n: u16) -> U256 {
        let mut a = U256::from(0);
        let mut b = U256::from(1);
        for _ in 0..n {
            let tmp = a;
            a = b;
            b = b.wrapping_add(tmp);
        }
        a
    }

    #[test]
    fn test_fibonacci_rust() {
        uint! {
            assert_eq!(fibonacci_rust(0), 0_U256);
            assert_eq!(fibonacci_rust(1), 1_U256);
            assert_eq!(fibonacci_rust(2), 1_U256);
            assert_eq!(fibonacci_rust(3), 2_U256);
            assert_eq!(fibonacci_rust(4), 3_U256);
            assert_eq!(fibonacci_rust(5), 5_U256);
            assert_eq!(fibonacci_rust(6), 8_U256);
            assert_eq!(fibonacci_rust(7), 13_U256);
            assert_eq!(fibonacci_rust(8), 21_U256);
            assert_eq!(fibonacci_rust(9), 34_U256);
            assert_eq!(fibonacci_rust(10), 55_U256);
            assert_eq!(fibonacci_rust(100), 354224848179261915075_U256);
            assert_eq!(fibonacci_rust(1000), 0x2e3510283c1d60b00930b7e8803c312b4c8e6d5286805fc70b594dc75cc0604b_U256);
        }
    }

    // ---

    mod resume {
        use super::*;

        with_matrix!(run_resume_tests);
    }

    fn run_resume_tests<B: Backend>(jit: &mut JitEvm<B>) {
        #[rustfmt::skip]
        let code = &[
            // 0
            op::PUSH1, 0x42,
            TEST_SUSPEND,
            
            // 1
            op::PUSH1, 0x69,
            TEST_SUSPEND,
            
            // 2
            op::ADD,
            TEST_SUSPEND,

            // 3
        ][..];

        let f = jit.compile(None, code, DEF_SPEC).unwrap();

        let stack = &mut EvmStack::new();
        let mut stack_len = 0;
        with_evm_context(code, |ecx| {
            assert_eq!(ecx.resume_at, 0);

            // op::PUSH1, 0x42,
            let r = unsafe { f.call(Some(stack), Some(&mut stack_len), ecx) };
            assert_eq!(r, InstructionResult::CallOrCreate);
            assert_eq!(stack_len, 1);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42));
            assert_eq!(ecx.resume_at, 1);

            // op::PUSH1, 0x69,
            let r = unsafe { f.call(Some(stack), Some(&mut stack_len), ecx) };
            assert_eq!(r, InstructionResult::CallOrCreate);
            assert_eq!(stack_len, 2);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42));
            assert_eq!(stack.as_slice()[1].to_u256(), U256::from(0x69));
            assert_eq!(ecx.resume_at, 2);

            // op::ADD,
            let r = unsafe { f.call(Some(stack), Some(&mut stack_len), ecx) };
            assert_eq!(r, InstructionResult::CallOrCreate);
            assert_eq!(stack_len, 1);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69));
            assert_eq!(ecx.resume_at, 3);

            // stop
            let r = unsafe { f.call(Some(stack), Some(&mut stack_len), ecx) };
            assert_eq!(r, InstructionResult::Stop);
            assert_eq!(stack_len, 1);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69));
            assert_eq!(ecx.resume_at, 3);

            // op::ADD,
            ecx.resume_at = 2;
            let r = unsafe { f.call(Some(stack), Some(&mut stack_len), ecx) };
            assert_eq!(r, InstructionResult::StackUnderflow);
            assert_eq!(stack_len, 1);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69));
            assert_eq!(ecx.resume_at, 2);

            stack.as_mut_slice()[stack_len] = U256::from(2).into();
            stack_len += 1;

            // op::ADD,
            ecx.resume_at = 2;
            let r = unsafe { f.call(Some(stack), Some(&mut stack_len), ecx) };
            assert_eq!(r, InstructionResult::CallOrCreate);
            assert_eq!(stack_len, 1);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2));
            assert_eq!(ecx.resume_at, 3);

            // op::PUSH1, 0x69,
            ecx.resume_at = 1;
            let r = unsafe { f.call(Some(stack), Some(&mut stack_len), ecx) };
            assert_eq!(r, InstructionResult::CallOrCreate);
            assert_eq!(stack_len, 2);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2));
            assert_eq!(stack.as_slice()[1].to_u256(), U256::from(0x69));
            assert_eq!(ecx.resume_at, 2);

            // op::ADD,
            let r = unsafe { f.call(Some(stack), Some(&mut stack_len), ecx) };
            assert_eq!(r, InstructionResult::CallOrCreate);
            assert_eq!(stack_len, 1);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2 + 0x69));
            assert_eq!(ecx.resume_at, 3);

            // stop
            let r = unsafe { f.call(Some(stack), Some(&mut stack_len), ecx) };
            assert_eq!(r, InstructionResult::Stop);
            assert_eq!(stack_len, 1);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2 + 0x69));
            assert_eq!(ecx.resume_at, 3);

            // stop
            let r = unsafe { f.call(Some(stack), Some(&mut stack_len), ecx) };
            assert_eq!(r, InstructionResult::Stop);
            assert_eq!(stack_len, 1);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2 + 0x69));
            assert_eq!(ecx.resume_at, 3);
        });
    }
}

//! JIT compiler implementation.

use crate::{
    callbacks::Callback, Backend, Builder, Bytecode, EvmContext, EvmStack, Instr, InstrData,
    InstrFlags, IntCC, JitEvmFn, Result, I256_MIN,
};
use revm_interpreter::{opcode as op, Contract, Gas, InstructionResult};
use revm_jit_backend::{Attribute, FunctionAttributeLocation, OptimizationLevel, TypeMethods};
use revm_primitives::{BlockEnv, CfgEnv, Env, SpecId, TxEnv, BLOCK_HASH_HISTORY, U256};
use std::{mem, path::PathBuf, sync::atomic::AtomicPtr};

const STACK_CAP: usize = 1024;
// const WORD_SIZE: usize = 32;

// TODO: Cannot find function if `compile` is called a second time.

// TODO: Generate less redundant stack length manip code, e.g. pop + push
// TODO: We can emit the length check by adding a params in/out instr flag; can be re-used for EOF

// TODO: Unify `callback` instructions after above.

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

    /// Sets whether to pass the stack length through the arguments.
    ///
    /// If this is set to `true`, the EVM stack will be passed in the arguments rather than
    /// allocated in the function locally.
    ///
    /// This is required if the executing with in an Evm and the bytecode contains `CALL` or
    /// `CREATE`-like instructions, as execution will need to be restored after the call.
    ///
    /// This is useful to inspect the stack after the function has been executed, but it does
    /// incur a performance penalty as the pointer might not be able to be fully optimized.
    ///
    /// Defaults to `true`.
    pub fn set_pass_stack_through_args(&mut self, yes: bool) {
        self.config.stack_through_args = yes;
    }

    /// Sets whether to pass the stack length through the arguments.
    ///
    /// If this is set to `true`, the EVM stack length will be passed in the arguments rather than
    /// allocated in the function locally.
    ///
    /// This is required if the executing with in an Evm and the bytecode contains `CALL` or
    /// `CREATE`-like instructions, as execution will need to be restored after the call.
    ///
    /// This is useful to inspect the stack length after the function has been executed, but it does
    /// incur a performance penalty as the pointer might not be able to be fully optimized.
    ///
    /// Defaults to `true`.
    pub fn set_pass_stack_len_through_args(&mut self, yes: bool) {
        self.config.stack_len_through_args = yes;
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

    /// Sets the static gas limit.
    ///
    /// Improves performance by being able to skip most gas checks.
    ///
    /// Has no effect if `disable_gas` is set, and will make the compiled function ignore the gas
    /// limit argument.
    ///
    /// Defaults to `None`.
    pub fn set_static_gas_limit(&mut self, static_gas_limit: Option<u64>) {
        self.config.static_gas_limit = static_gas_limit;
    }

    /// Compiles the given EVM bytecode into a JIT function.
    pub fn compile(&mut self, bytecode: &[u8], spec_id: SpecId) -> Result<JitEvmFn> {
        let bytecode = debug_time!("parse", || self.parse_bytecode(bytecode, spec_id))?;
        debug_time!("compile", || self.compile_bytecode(&bytecode))
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
        // trace!(bytecode = revm_primitives::hex::encode(bytecode));
        let mut bytecode = trace_time!("new bytecode", || Bytecode::new(bytecode, spec_id));
        trace_time!("analyze", || bytecode.analyze())?;
        Ok(bytecode)
    }

    fn compile_bytecode(&mut self, bytecode: &Bytecode<'_>) -> Result<JitEvmFn> {
        fn align_size<T>(i: usize) -> (usize, usize, usize) {
            (i, mem::align_of::<T>(), mem::size_of::<T>())
        }

        let name = &self.new_name()[..];

        let i8 = self.backend.type_int(8);
        let ptr = self.backend.type_ptr();
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
        let mut bcx = self.backend.build_function(name, ret, params, param_names)?;

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
        .chain(self.config.frame_pointers.then_some(Attribute::AllFramePointers))
        // We can unwind in panics, which are present only in debug assertions.
        .chain((!self.config.debug_assertions).then_some(Attribute::NoUnwind));
        for attr in function_attributes {
            bcx.add_function_attribute(None, attr, FunctionAttributeLocation::Function);
        }

        // Pointer argument attributes.
        if !self.config.debug_assertions {
            for &(i, align, dereferenceable) in ptr_attrs {
                for attr in [
                    Attribute::NoAlias,
                    Attribute::NoCapture,
                    Attribute::NoUndef,
                    Attribute::Align(align as u64),
                    Attribute::Dereferenceable(dereferenceable as u64),
                ] {
                    let loc = FunctionAttributeLocation::Param(i as _);
                    bcx.add_function_attribute(None, attr, loc);
                }
            }
        }

        trace_time!("translate", || FunctionCx::translate(
            bcx,
            &self.config,
            &mut self.callbacks,
            bytecode
        ))?;

        let verify = |b: &mut B| trace_time!("verify", || b.verify_function(name));
        if let Some(dir) = &self.out_dir {
            trace_time!("dump unopt IR", || {
                let filename = format!("{name}.unopt.{}", self.backend.ir_extension());
                self.backend.dump_ir(&dir.join(filename))
            })?;

            // Dump IR before verifying for better debugging.
            verify(&mut self.backend)?;

            trace_time!("dump unopt disasm", || {
                let filename = format!("{name}.unopt.s");
                self.backend.dump_disasm(&dir.join(filename))
            })?;
        } else {
            verify(&mut self.backend)?;
        }

        trace_time!("optimize", || self.backend.optimize_function(name))?;

        if let Some(dir) = &self.out_dir {
            trace_time!("dump opt IR", || {
                let filename = format!("{name}.opt.{}", self.backend.ir_extension());
                self.backend.dump_ir(&dir.join(filename))
            })?;

            trace_time!("dump opt disasm", || {
                let filename = format!("{name}.opt.s");
                self.backend.dump_disasm(&dir.join(filename))
            })?;
        }

        let addr = trace_time!("finalize", || self.backend.get_function(name))?;
        Ok(JitEvmFn::new(unsafe { std::mem::transmute(addr) }))
    }

    fn new_name(&mut self) -> String {
        let name = format!("evm_bytecode_{}", self.function_counter);
        self.function_counter += 1;
        name
    }
}

#[derive(Clone, Debug)]
struct FcxConfig {
    comments_enabled: bool,
    debug_assertions: bool,
    frame_pointers: bool,

    stack_through_args: bool,
    stack_len_through_args: bool,
    gas_disabled: bool,
    static_gas_limit: Option<u64>,
}

impl Default for FcxConfig {
    fn default() -> Self {
        Self {
            debug_assertions: cfg!(debug_assertions),
            comments_enabled: false,
            frame_pointers: cfg!(debug_assertions),
            stack_through_args: true,
            stack_len_through_args: true,
            gas_disabled: false,
            static_gas_limit: None,
        }
    }
}

struct FunctionCx<'a, B: Backend> {
    comments_enabled: bool,
    #[allow(dead_code)]
    frame_pointers: bool,
    disable_gas: bool,

    /// The backend's function builder.
    bcx: B::Builder<'a>,

    // Common types.
    isize_type: B::Type,
    word_type: B::Type,
    address_type: B::Type,
    i8_type: B::Type,

    /// The stack length. Either passed in the arguments as a pointer or allocated locally.
    stack_len: Pointer<B>,
    /// The stack value. Constant throughout the function, either passed in the arguments as a
    /// pointer or allocated locally.
    stack: Pointer<B>,
    /// The amount of gas used. `isize`. Either passed in the arguments as a pointer or allocated
    /// locally.
    gas_remaining: Pointer<B>,
    gas_remaining_nomem: Pointer<B>,
    /// The gas limit. Constant throughout the function, passed in the arguments or set statically.
    gas_limit: Option<B::Value>,
    /// The environment. Constant throughout the function.
    env: B::Value,
    /// The contract. Constant throughout the function.
    contract: B::Value,
    /// The EVM context. Opaque pointer, only passed to callbacks.
    ecx: B::Value,

    /// The bytecode being translated.
    bytecode: &'a Bytecode<'a>,
    /// All entry blocks for each instruction.
    instr_blocks: Vec<B::BasicBlock>,
    /// The current instruction being translated.
    ///
    /// Note that `self.op_blocks[current_opcode]` does not necessarily equal the builder's current
    /// block.
    current_opcode: Instr,

    /// Callbacks.
    callbacks: &'a mut Callbacks<B>,
}

impl<'a, B: Backend> FunctionCx<'a, B> {
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

        let zero = bcx.iconst(isize_type, 0);

        // Set up entry block.
        let gas_ptr = bcx.fn_param(0);
        let gas_remaining = {
            let offset = bcx.iconst(isize_type, mem::offset_of!(pf::Gas, remaining) as i64);
            let base = PointerBase::Address(bcx.gep(i8_type, gas_ptr, &[offset]));
            Pointer { ty: isize_type, base }
        };
        let gas_remaining_nomem = {
            let offset = bcx.iconst(isize_type, mem::offset_of!(pf::Gas, remaining_nomem) as i64);
            let base = PointerBase::Address(bcx.gep(i8_type, gas_ptr, &[offset]));
            Pointer { ty: i64_type, base }
        };

        let sp_arg = bcx.fn_param(1);
        let stack = {
            let base = if config.stack_through_args {
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
            let base = if config.stack_len_through_args {
                PointerBase::Address(stack_len_arg)
            } else {
                let stack_len = bcx.new_stack_slot(isize_type, "len.addr");
                bcx.stack_store(zero, stack_len);
                PointerBase::StackSlot(stack_len)
            };
            Pointer { ty: isize_type, base }
        };

        let env = bcx.fn_param(3);
        let contract = bcx.fn_param(4);
        let ecx = bcx.fn_param(5);

        // Create all instruction entry blocks.
        let op_blocks: Vec<_> = bytecode
            .iter_instrs()
            .map(|(i, data)| bcx.create_block(&op_block_name_with(i, data, "")))
            .collect();
        assert!(!op_blocks.is_empty(), "translating empty bytecode");

        let mut fx = FunctionCx {
            comments_enabled: config.comments_enabled,
            frame_pointers: config.frame_pointers,
            disable_gas: config.gas_disabled,

            bcx,
            isize_type,
            address_type,
            word_type,
            i8_type,
            stack_len,
            stack,
            gas_remaining,
            gas_remaining_nomem,
            gas_limit: None,
            env,
            contract,
            ecx,

            bytecode,
            instr_blocks: op_blocks,
            current_opcode: usize::MAX,

            callbacks,
        };

        // Add debug assertions for the parameters.
        if config.debug_assertions {
            fx.pointer_panic_with_bool(!config.gas_disabled, gas_ptr, "gas pointer");
            fx.pointer_panic_with_bool(config.stack_through_args, sp_arg, "stack pointer");
            fx.pointer_panic_with_bool(
                config.stack_len_through_args,
                stack_len_arg,
                "stack length pointer",
            );
            fx.pointer_panic_with_bool(true, env, "env pointer");
            fx.pointer_panic_with_bool(true, contract, "contract pointer");
            fx.pointer_panic_with_bool(true, ecx, "EVM context pointer");
        }

        // Load the gas limit after generating debug assertions.
        fx.gas_limit = Some(if let Some(static_gas_limit) = config.static_gas_limit {
            fx.bcx.iconst(i64_type, static_gas_limit as i64)
        } else {
            fx.bcx.load(i64_type, gas_ptr, "gas_limit")
        });

        // Branch to the first instruction.
        // The bytecode is guaranteed to have at least one instruction.
        fx.bcx.br(fx.instr_blocks[0]);

        // Translate individual instructions into their respective blocks.
        for (instr, _) in bytecode.iter_instrs() {
            fx.translate_instr(instr)?;
        }

        Ok(())
    }

    fn translate_instr(&mut self, instr: Instr) -> Result<()> {
        self.current_opcode = instr;
        let data = self.bytecode.instr(instr);
        let instr_block = self.instr_blocks[instr];
        self.bcx.switch_to_block(instr_block);

        let opcode = data.opcode;

        let branch_to_next_opcode = |this: &mut Self| {
            debug_assert!(
                !this.bytecode.is_instr_diverging(instr),
                "attempted to branch to next instruction in a diverging instruction: {}",
                this.bytecode.raw_opcode(instr),
            );
            if let Some(next) = this.instr_blocks.get(instr + 1) {
                this.bcx.br(*next);
            }
        };
        let epilogue = |this: &mut Self| {
            this.bcx.seal_block(instr_block);
        };

        // Make sure to run the epilogue before returning.
        macro_rules! goto_return {
            ($comment:expr) => {
                branch_to_next_opcode(self);
                goto_return!(no_branch $comment);
            };
            (no_branch $comment:expr) => {
                if self.comments_enabled {
                    self.add_comment($comment);
                }
                epilogue(self);
                return Ok(());
            };
            (build $ret:expr) => {{
                self.build_return($ret);
                goto_return!(no_branch "");
            }};
        }

        // Assert that we already skipped the block.
        debug_assert!(!data.flags.contains(InstrFlags::DEAD_CODE));

        // Disabled instructions don't pay gas.
        if data.flags.contains(InstrFlags::DISABLED) {
            goto_return!(build InstructionResult::NotActivated);
        }
        if data.flags.contains(InstrFlags::UNKNOWN) {
            goto_return!(build InstructionResult::OpcodeNotFound);
        }

        // Pay static gas.
        if !self.disable_gas {
            if let Some(static_gas) = data.static_gas() {
                self.gas_cost_imm(static_gas as u64);
            }
        }

        if data.flags.contains(InstrFlags::SKIP_LOGIC) {
            goto_return!("skipped");
        }

        // TODO: Stack length manip go here.

        macro_rules! unop {
            ($op:ident) => {{
                let mut a = self.pop(true);
                a = self.bcx.$op(a);
                self.push_unchecked(a);
            }};
        }

        macro_rules! binop {
            ($op:ident) => {{
                let [a, b] = self.popn(true);
                let r = self.bcx.$op(a, b);
                self.push_unchecked(r);
            }};
            (@rev $op:ident) => {{
                let [a, b] = self.popn(true);
                let r = self.bcx.$op(b, a);
                self.push_unchecked(r);
            }};
            (@if_not_zero $op:ident) => {{
                // TODO: `select` might not have the same semantics in all backends.
                let [a, b] = self.popn(true);
                let b_is_zero = self.bcx.icmp_imm(IntCC::Equal, b, 0);
                let zero = self.bcx.iconst_256(U256::ZERO);
                let op_result = self.bcx.$op(a, b);
                let r = self.bcx.select(b_is_zero, zero, op_result);
                self.push_unchecked(r);
            }};
        }

        macro_rules! field {
            // Gets the pointer to a field.
            ($field:ident; @get $($paths:path),*; $($spec:tt).*) => {
                self.get_field(self.$field, 0 $(+ mem::offset_of!($paths, $spec))*)
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
                let [a, b] = self.popn(true);
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
                self.push_unchecked(r);
            }
            op::MOD => binop!(@if_not_zero urem),
            op::SMOD => binop!(@if_not_zero srem),
            op::ADDMOD => {
                let sp = self.pop_top_sp(3);
                let _ = self.callback(Callback::AddMod, &[sp]);
            }
            op::MULMOD => {
                let sp = self.pop_top_sp(3);
                let _ = self.callback(Callback::MulMod, &[sp]);
            }
            op::EXP => {
                let sp = self.pop_top_sp(2);
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

                let [ext, x] = self.popn(true);
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
                self.push_unchecked(r);
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

                let [a, b] = self.popn(true);
                let r = self.bcx.icmp(cond, a, b);
                let r = self.bcx.zext(self.word_type, r);
                self.push_unchecked(r);
            }
            op::ISZERO => {
                let a = self.pop(true);
                let r = self.bcx.icmp_imm(IntCC::Equal, a, 0);
                let r = self.bcx.zext(self.word_type, r);
                self.push_unchecked(r);
            }
            op::AND => binop!(bitand),
            op::OR => binop!(bitor),
            op::XOR => binop!(bitxor),
            op::NOT => unop!(bitnot),
            op::BYTE => {
                let [index, value] = self.popn(true);
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
                self.push_unchecked(r);
            }
            op::SHL => binop!(@rev ishl),
            op::SHR => binop!(@rev ushr),
            op::SAR => binop!(@rev sshr),

            op::KECCAK256 => {
                let sp = self.pop_top_sp(2);
                self.callback_ir(Callback::Keccak256, &[self.ecx, sp]);
            }

            op::ADDRESS => {
                contract_field!(@push @[endian = "big"] self.address_type, Contract; address)
            }
            op::BALANCE => {
                let sp = self.pop_top_sp(1);
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::Balance, &[self.ecx, spec_id, sp]);
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
                let index = self.pop(true);
                let len_ptr = contract_field!(@get Contract, pf::Bytes; input.len);
                let len = self.bcx.load(self.isize_type, len_ptr, "input.len");
                let len = self.bcx.zext(self.word_type, len);
                let in_bounds = self.bcx.icmp(IntCC::UnsignedLessThan, index, len);

                let ptr = contract_field!(@get Contract, pf::Bytes; input.ptr);
                let value = self.lazy_select(
                    in_bounds,
                    self.word_type,
                    |this, _| {
                        let ptr = this.bcx.load(this.bcx.type_ptr(), ptr, "contract.input.ptr");
                        let calldata = this.bcx.gep(this.i8_type, ptr, &[index]);

                        let max = this.bcx.iconst(this.word_type, 32);
                        let slice_len = this.bcx.umin(index, max);
                        let slice_len = this.bcx.ireduce(this.isize_type, slice_len);

                        let tmp = this.bcx.new_stack_slot(this.word_type, "calldata.addr");
                        let tmp_addr = this.bcx.stack_addr(tmp);
                        this.bcx.memcpy(tmp_addr, calldata, slice_len);
                        let mut value = this.bcx.stack_load(this.word_type, tmp, "calldata.i256");
                        if cfg!(target_endian = "little") {
                            value = this.bcx.bswap(value);
                        }
                        value
                    },
                    |this, _| this.bcx.iconst_256(U256::ZERO),
                );
                self.push_unchecked(value);
            }
            op::CALLDATASIZE => {
                contract_field!(@push self.isize_type, Contract, pf::Bytes; input.len)
            }
            op::CALLDATACOPY => {
                let sp = self.pop_top_sp(4);
                self.callback_ir(Callback::CallDataCopy, &[self.ecx, sp]);
            }
            op::CODESIZE => {
                contract_field!(@push self.isize_type, Contract, pf::BytecodeLocked; bytecode.original_len)
            }
            op::CODECOPY => {
                let sp = self.pop_top_sp(3);
                self.callback_ir(Callback::CodeCopy, &[self.ecx, sp]);
            }

            op::GASPRICE => {
                env_field!(@push @[endian = "little"] self.word_type, Env, TxEnv; tx.gas_price)
            }
            op::EXTCODESIZE => {
                let sp = self.pop_top_sp(1);
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::ExtCodeSize, &[self.ecx, spec_id, sp]);
            }
            op::EXTCODECOPY => {
                let sp = self.pop_top_sp(4);
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::ExtCodeCopy, &[self.ecx, spec_id, sp]);
            }
            op::RETURNDATASIZE => {
                field!(ecx; @push self.isize_type, EvmContext<'_>, pf::Slice; return_data.len)
            }
            op::RETURNDATACOPY => {
                let sp = self.pop_top_sp(3);
                self.callback_ir(Callback::ReturnDataCopy, &[self.ecx, sp]);
            }
            op::EXTCODEHASH => {
                let sp = self.pop_top_sp(1);
                let spec_id = self.const_spec_id();
                self.callback_ir(Callback::ExtCodeHash, &[self.ecx, spec_id, sp]);
            }
            op::BLOCKHASH => {
                let requested_number = self.pop(true);
                let actual_number = env_field!(
                    @load @[endian = "little"] self.word_type, Env, BlockEnv; block.number
                );

                let block_hash = self.bcx.new_stack_slot(self.word_type, "block_hash.addr");

                // Only call host if `1 <= diff <= 256`. CFG:
                // current -> in_range -> call_host -> contd
                //         \           |            /
                //          \------ default -------/
                let in_range = self.create_block_after_current("in_range");
                let call_host = self.create_block_after(in_range, "call_host");
                let default = self.create_block_after(call_host, "default");
                let target = self.create_block_after(default, "contd");

                // Not GE to skip `diff == 0`.
                let is_valid =
                    self.bcx.icmp(IntCC::UnsignedGreaterThan, actual_number, requested_number);
                self.bcx.brif(is_valid, in_range, default);

                self.bcx.switch_to_block(in_range);
                let diff = self.bcx.isub(actual_number, requested_number);
                let diff_in_range = self.bcx.icmp_imm(
                    IntCC::UnsignedLessThanOrEqual,
                    diff,
                    BLOCK_HASH_HISTORY as i64,
                );
                self.bcx.brif(diff_in_range, call_host, default);

                self.bcx.switch_to_block(call_host);
                let diff = self.bcx.ireduce(self.isize_type, diff);
                let block_hash_ptr = self.bcx.stack_addr(block_hash);
                self.callback_ir(Callback::BlockHash, &[self.ecx, diff, block_hash_ptr]);
                self.bcx.br(target);

                self.bcx.switch_to_block(default);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.bcx.stack_store(zero, block_hash);
                self.bcx.br(target);

                self.bcx.switch_to_block(target);
                let block_hash = self.bcx.stack_load(self.word_type, block_hash, "block_hash");
                self.push_unchecked(block_hash);
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
                        let ptr = self.bcx.gep(self.i8_type, opt, &[one]);
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
                let sp = self.pop_top_sp(1);
                self.callback_ir(Callback::SelfBalance, &[self.ecx, sp]);
            }
            op::BASEFEE => {
                env_field!(@push @[endian = "little"] self.word_type, Env, BlockEnv; block.basefee)
            }
            op::BLOBHASH => {
                let index = self.pop(true);
                let blob_hashes = env_field!(@get Env, TxEnv; tx.blob_hashes);
                // Manual `<[_]>::get` :/
                // Vec<T> == { ptr, len: u64, capacity: u64 }
                let type_ptr = self.bcx.type_ptr();
                let len = {
                    let one = self.bcx.iconst(self.isize_type, 1);
                    let len = self.bcx.gep(self.isize_type, blob_hashes, &[one]);
                    let len = self.bcx.load(self.isize_type, len, "blob_hashes.len");
                    self.bcx.zext(self.word_type, len)
                };
                let in_bounds = self.bcx.icmp(IntCC::UnsignedLessThan, index, len);
                let word_type = self.word_type;
                let result = self.bcx.lazy_select(
                    in_bounds,
                    word_type,
                    |bcx, _block| {
                        let ptr = bcx.load(type_ptr, blob_hashes, "blob_hashes.ptr");
                        let ptr = bcx.gep(word_type, ptr, &[index]);
                        let mut hash = bcx.load(word_type, ptr, "blob_hashes[index]");
                        if !cfg!(target_endian = "big") {
                            hash = bcx.bswap(hash);
                        }
                        hash
                    },
                    |bcx, _block| bcx.iconst_256(U256::ZERO),
                );
                self.push_unchecked(result);
            }
            op::BLOBBASEFEE => {
                // TODO
                // Option<{ u128, u64 }> (48 bytes) =>
                // { bool, pad([u8; 15]), u128, u64, pad([u8; 8]) }
                let ptr = env_field!(@get Env, BlockEnv; block.blob_excess_gas_and_price);
                let _ = ptr;
            }

            op::POP => {
                let len = self.load_len_at_least(1);
                let len = self.bcx.isub_imm(len, 1);
                self.store_len(len);
            }
            op::MLOAD => {
                let sp = self.pop_top_sp(1);
                self.callback_ir(Callback::Mload, &[self.ecx, sp]);
            }
            op::MSTORE => {
                let sp = self.pop_sp(2);
                self.callback_ir(Callback::Mstore, &[self.ecx, sp]);
            }
            op::MSTORE8 => {
                let sp = self.pop_sp(2);
                self.callback_ir(Callback::Mstore8, &[self.ecx, sp]);
            }
            op::SLOAD => {
                let sp = self.pop_top_sp(1);
                self.callback_ir(Callback::Sload, &[self.ecx, sp]);
            }
            op::SSTORE => {
                self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
                let sp = self.pop_sp(2);
                self.callback_ir(Callback::Sstore, &[self.ecx, sp]);
            }
            op::JUMP | op::JUMPI => {
                if data.flags.contains(InstrFlags::INVALID_JUMP) {
                    self.build_return(InstructionResult::InvalidJump);
                } else if data.flags.contains(InstrFlags::STATIC_JUMP) {
                    let target_opcode = data.data as usize;
                    debug_assert_eq!(
                        self.bytecode.instr(target_opcode).opcode,
                        op::JUMPDEST,
                        "jumping to non-JUMPDEST: ic={target_opcode} -> {}",
                        self.bytecode.instr(target_opcode).to_op()
                    );

                    let target = self.instr_blocks[target_opcode];
                    if opcode == op::JUMPI {
                        let cond_word = self.pop(true);
                        let cond = self.bcx.icmp_imm(IntCC::NotEqual, cond_word, 0);
                        let next = self.instr_blocks[instr + 1];
                        self.bcx.brif(cond, target, next);
                    } else {
                        self.bcx.br(target);
                    }
                } else {
                    todo!("dynamic jumps");
                }

                goto_return!(no_branch "");
            }
            op::PC => {
                let pc = self.bcx.iconst_256(U256::from(data.data));
                self.push(pc);
            }
            op::MSIZE => {
                let msize = self.callback(Callback::Msize, &[self.ecx]).unwrap();
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
                let sp = self.pop_top_sp(1);
                self.callback(Callback::Tload, &[self.ecx, sp]).unwrap();
            }
            op::TSTORE => {
                self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
                let sp = self.pop_sp(2);
                self.callback(Callback::Tstore, &[self.ecx, sp]).unwrap();
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
                let _n = opcode - op::LOG0;
                // TODO: host
            }

            op::CREATE => {
                self.create_common(false);
                goto_return!(build InstructionResult::CallOrCreate);
            }
            op::CALL => {
                // TODO: host
            }
            op::CALLCODE => {
                // TODO: host
            }
            op::RETURN => {
                self.return_common();
                goto_return!(build InstructionResult::Return);
            }
            op::DELEGATECALL => {
                // TODO: host
            }
            op::CREATE2 => {
                self.create_common(true);
                goto_return!(build InstructionResult::CallOrCreate);
            }

            op::STATICCALL => {
                // TODO: host
            }

            op::REVERT => {
                self.return_common();
                goto_return!(build InstructionResult::Revert);
            }
            op::INVALID => goto_return!(build InstructionResult::InvalidFEOpcode),
            op::SELFDESTRUCT => {
                self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
                // TODO: host
            }

            _ => unreachable!("unimplemented instructions: {}", data.to_op_in(self.bytecode)),
        }

        goto_return!("normal exit");
    }

    /// Pushes a 256-bit value onto the stack, checking for stack overflow.
    fn push(&mut self, value: B::Value) {
        self.pushn(&[value]);
    }

    /// Pushes 256-bit values onto the stack, checking for stack overflow.
    fn pushn(&mut self, values: &[B::Value]) {
        debug_assert!(values.len() <= STACK_CAP);

        let len = self.load_len();
        let failure_cond =
            self.bcx.icmp_imm(IntCC::UnsignedGreaterThan, len, (STACK_CAP - values.len()) as i64);
        self.build_failure(failure_cond, InstructionResult::StackOverflow);

        self.pushn_unchecked(values);
    }

    /// Pushes a 256-bit value onto the stack, without checking for stack overflow.
    fn push_unchecked(&mut self, value: B::Value) {
        self.pushn_unchecked(&[value]);
    }

    /// Pushes 256-bit values onto the stack, without checking for stack overflow.
    fn pushn_unchecked(&mut self, values: &[B::Value]) {
        let mut len = self.load_len();
        for &value in values {
            let sp = self.sp_at(len);
            self.bcx.store(value, sp);
            len = self.bcx.iadd_imm(len, 1);
        }
        self.store_len(len);
    }

    /// Removes the topmost element from the stack and returns it.
    fn pop(&mut self, load: bool) -> B::Value {
        self.popn::<1>(load)[0]
    }

    /// Removes the topmost `N` elements from the stack and returns them.
    ///
    /// If `load` is `false`, returns just the pointers.
    fn popn<const N: usize>(&mut self, load: bool) -> [B::Value; N] {
        debug_assert_ne!(N, 0);
        debug_assert!(N < 26, "too many pops");

        let mut len = self.load_len_at_least(N);
        let ret = std::array::from_fn(|i| {
            len = self.bcx.isub_imm(len, 1);
            let sp = self.sp_at(len);
            if load {
                let name = b'a' + i as u8;
                self.load_word(sp, core::str::from_utf8(&[name]).unwrap())
            } else {
                sp
            }
        });
        self.store_len(len);
        ret
    }

    /// Check length for `n`, decrement `n`, and return the stack pointer at the initial length.
    fn pop_sp(&mut self, n: usize) -> B::Value {
        debug_assert_ne!(n, 0);
        let len = self.load_len_at_least(n);
        let subtracted = self.bcx.isub_imm(len, n as i64);
        self.store_len(subtracted);
        self.sp_at(len)
    }

    /// Check length for `n`, decrement `n - 1`, and return the stack pointer at the initial length.
    fn pop_top_sp(&mut self, n: usize) -> B::Value {
        debug_assert_ne!(n, 0);
        let len = self.load_len_at_least(n);
        if n > 1 {
            let subtracted = self.bcx.isub_imm(len, (n - 1) as i64);
            self.store_len(subtracted);
        }
        self.sp_at(len)
    }

    /// Checks if the stack has at least `n` elements and returns the stack length.
    fn load_len_at_least(&mut self, n: usize) -> B::Value {
        let len = self.load_len();
        let failure_cond = self.bcx.icmp_imm(IntCC::UnsignedLessThan, len, n as i64);
        self.build_failure(failure_cond, InstructionResult::StackUnderflow);
        len
    }

    /// Returns an error if the current context is a static call.
    fn fail_if_staticcall(&mut self, ret: InstructionResult) {
        let ptr = self.get_field(self.ecx, mem::offset_of!(EvmContext<'_>, is_static));
        let bool = self.bcx.type_int(1);
        let is_static = self.bcx.load(bool, ptr, "is_static");
        self.build_failure(is_static, ret)
    }

    /// Duplicates the `n`th value from the top of the stack.
    /// `n` cannot be `0`.
    fn dup(&mut self, n: u8) {
        debug_assert_ne!(n, 0);

        let len = self.load_len();
        let failure_cond = self.bcx.icmp_imm(IntCC::UnsignedLessThan, len, n as i64);
        self.build_failure(failure_cond, InstructionResult::StackUnderflow);

        let sp = self.sp_from_top(len, n as usize);
        let value = self.load_word(sp, &format!("dup{n}"));
        self.push(value);
    }

    /// Swaps the topmost value with the `n`th value from the top.
    fn swap(&mut self, n: u8) {
        debug_assert_ne!(n, 0);

        let len = self.load_len();
        let failure_cond = self.bcx.icmp_imm(IntCC::UnsignedLessThan, len, n as i64);
        self.build_failure(failure_cond, InstructionResult::StackUnderflow);

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
    fn return_common(&mut self) {
        let sp = self.pop_sp(2);
        self.callback_ir(Callback::DoReturn, &[self.ecx, sp]);
    }

    fn create_common(&mut self, is_create2: bool) {
        self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
        let sp = self.pop_sp(3 + is_create2 as usize);
        let is_create2 = self.bcx.iconst(self.bcx.type_int(1), is_create2 as i64);
        self.callback_ir(Callback::Create, &[self.ecx, sp, is_create2]);
        self.build_return(InstructionResult::CallOrCreate);
    }

    /// Loads the word at the given pointer.
    fn load_word(&mut self, ptr: B::Value, name: &str) -> B::Value {
        self.bcx.load(self.word_type, ptr, name)
    }

    /// Loads the stack length.
    fn load_len(&mut self) -> B::Value {
        self.stack_len.load(&mut self.bcx, "len")
    }

    /// Returns the spec ID as a value.
    fn const_spec_id(&mut self) -> B::Value {
        self.bcx.iconst(self.i8_type, self.bytecode.spec_id as i64)
    }

    /// Gets the environment field at the given offset.
    fn get_field(&mut self, ptr: B::Value, offset: usize) -> B::Value {
        let offset = self.bcx.iconst(self.isize_type, offset as i64);
        self.bcx.gep(self.i8_type, ptr, &[offset])
    }

    /// Stores the stack length.
    fn store_len(&mut self, value: B::Value) {
        self.stack_len.store(&mut self.bcx, value);
    }

    /// Loads the gas used.
    fn load_gas_remaining(&mut self) -> B::Value {
        self.gas_remaining.load(&mut self.bcx, "gas_remaining")
    }

    /// Stores the gas used.
    fn store_gas_remaining(&mut self, value: B::Value) {
        self.gas_remaining.store(&mut self.bcx, value);
    }

    /// Returns the stack pointer at `len` (`&stack[len]`).
    fn sp_at(&mut self, len: B::Value) -> B::Value {
        let ptr = self.stack.addr(&mut self.bcx);
        self.bcx.gep(self.word_type, ptr, &[len])
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
        self.bcx.ret(&[ret]);

        self.bcx.switch_to_block(target);
    }

    /// Builds `return ret`.
    fn build_return(&mut self, ret: InstructionResult) {
        let ret = self.bcx.iconst(self.i8_type, ret as i64);
        self.bcx.ret(&[ret]);
        if self.comments_enabled {
            self.add_comment(&format!("return {ret:?}"));
        }
    }

    // Pointer must not be null if `must_be_set` is true.
    fn pointer_panic_with_bool(&mut self, must_be_set: bool, ptr: B::Value, name: &str) {
        if !must_be_set {
            return;
        }
        let panic_cond = self.bcx.is_null(ptr);
        let msg = format!("revm_jit panic: {name} must not be null");
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
        let _callsite = self.bcx.call(function, &[ptr, len]);
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
        if self.current_opcode == usize::MAX {
            return format!("entry.{name}");
        }
        op_block_name_with(self.current_opcode, self.bytecode.instr(self.current_opcode), name)
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

    /// Gets the address of the pointer.
    fn addr(&self, bcx: &mut B::Builder<'_>) -> B::Value {
        match self.base {
            PointerBase::Address(ptr) => ptr,
            PointerBase::StackSlot(slot) => bcx.stack_addr(slot),
        }
    }
}

/// Callback cache.
struct Callbacks<B: Backend>([Option<B::Function>; Callback::COUNT]);

impl<B: Backend> Callbacks<B> {
    fn new() -> Self {
        Self([None; Callback::COUNT])
    }

    fn clear(&mut self) {
        *self = Self::new();
    }

    fn get(&mut self, cb: Callback, bcx: &mut B::Builder<'_>) -> B::Function {
        *self.0[cb as usize].get_or_insert_with(
            #[cold]
            || {
                let name = cb.name();
                let ret = cb.ret(bcx);
                let params = cb.params(bcx);
                let address = cb.addr();
                let f = bcx.add_callback_function(name, ret, &params, address);
                let default_attrs: &[Attribute] = if cb == Callback::Panic {
                    &[
                        Attribute::Cold,
                        Attribute::NoReturn,
                        Attribute::NoFree,
                        Attribute::NoRecurse,
                        Attribute::NoSync,
                    ]
                } else {
                    &[
                        Attribute::WillReturn,
                        Attribute::NoFree,
                        Attribute::NoRecurse,
                        Attribute::NoSync,
                        Attribute::NoUnwind,
                        Attribute::Speculatable,
                    ]
                };
                for attr in default_attrs.iter().chain(cb.attrs()).copied() {
                    bcx.add_function_attribute(Some(f), attr, FunctionAttributeLocation::Function);
                }
                f
            },
        )
    }
}

fn op_block_name_with(op: Instr, data: InstrData, with: &str) -> String {
    let data = data.to_op();
    if with.is_empty() {
        format!("op.{op}.{data}")
    } else {
        format!("op.{op}.{data}.{with}")
    }
}

// HACK: Need these structs' fields to be public for `offset_of!`.
// `private_fields`
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

#[cfg(test)]
#[allow(clippy::needless_update)]
mod tests {
    use super::*;
    use crate::*;
    use primitives::{address, keccak256, spec_to_generic, Address, Bytes, KECCAK_EMPTY};
    use revm_interpreter::{gas, opcode as op};
    use revm_primitives::ruint::uint;
    use std::fmt;

    #[cfg(feature = "llvm")]
    use llvm::inkwell::context::Context as LlvmContext;

    const I256_MAX: U256 = U256::from_limbs([
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x7FFFFFFFFFFFFFFF,
    ]);

    // Default values.
    const DEF_SPEC: SpecId = SpecId::CANCUN;
    const DEF_OPINFOS: &[OpcodeInfo; 256] = op_info_map(DEF_SPEC);

    const DEF_GAS_LIMIT: u64 = 100_000;
    const DEF_GAS_LIMIT_U256: U256 = U256::from_le_slice(&DEF_GAS_LIMIT.to_le_bytes());

    /// Default code address.
    const DEF_ADDR: Address = address!("babababababababababababababababababababa");
    const DEF_CALLER: Address = address!("cacacacacacacacacacacacacacacacacacacaca");
    static DEF_CD: &[u8] = &[0xaa; 64];
    static DEF_RD: &[u8] = &[0xbb; 64];
    const DEF_VALUE: U256 = uint!(123_456_789_U256);

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
        // interpreter.shared_memory = {
        //     let cap = 64;
        //     let mut mem = revm_interpreter::SharedMemory::with_capacity(cap);
        //     unsafe { mem.context_memory_mut().as_mut_ptr().write_bytes(0xcc, cap) };
        //     mem
        // };

        let mut host = revm_interpreter::DummyHost::default();

        f(&mut EvmContext::from_interpreter(&mut interpreter, &mut host))
    }

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

    macro_rules! tests {
        ($($group:ident { $($t:tt)* })*) => { uint! {
            $(
                mod $group {
                    #[allow(unused_imports)]
                    use super::*;

                    tests!(@cases $($t)*);
                }
            )*
        }};

        (@cases $( $name:ident($($t:tt)*) ),* $(,)?) => {
            $(
                #[test]
                fn $name() {
                    run_case(tests!(@case $($t)*));
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
                expected_stack: &[],
                expected_gas: 0,
            }),
            no_stop(@raw {
                bytecode: &[op::PUSH0],
                expected_stack: &[U256::ZERO],
                expected_gas: 2,
            }),
            stop(@raw {
                bytecode: &[op::STOP],
                expected_stack: &[],
                expected_gas: 0,
            }),
            invalid(@raw {
                bytecode: &[op::INVALID],
                expected_return: InstructionResult::InvalidFEOpcode,
                expected_stack: &[],
                expected_gas: 0,
            }),
            unknown(@raw {
                bytecode: &[0x21],
                expected_return: InstructionResult::OpcodeNotFound,
                expected_stack: &[],
                expected_gas: 0,
            }),
            underflow1(@raw {
                bytecode: &[op::ADD],
                expected_return: InstructionResult::StackUnderflow,
                expected_stack: &[],
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
                expected_stack: &[],
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
                expected_stack: &[],
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
                expected_stack: &[],
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
                expected_stack: &[],
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
                expected_gas: 3 + 2 + (gas::keccak256_cost(32).unwrap() + 3),
            }),
            keccak256_2(@raw {
                bytecode: &[op::PUSH2, 0x69, 0x42, op::PUSH0, op::MSTORE, op::PUSH1, 0x20, op::PUSH0, op::KECCAK256],
                expected_stack: &[keccak256(0x6942_U256.to_be_bytes::<32>()).into()],
                expected_gas: 3 + 2 + (3 + 3) + 3 + 2 + gas::keccak256_cost(32).unwrap(),
            }),
        }

        host {
            gas_price(@raw {
                bytecode: &[op::GASPRICE],
                expected_stack: &[U256::ZERO],
                expected_gas: 2,
            }),
        }
    }

    struct TestCase<'a> {
        bytecode: &'a [u8],
        spec_id: SpecId,

        expected_return: InstructionResult,
        expected_stack: &'a [U256],
        expected_gas: u64,
    }

    impl Default for TestCase<'_> {
        fn default() -> Self {
            Self {
                bytecode: &[],
                spec_id: DEF_SPEC,
                expected_return: InstructionResult::Stop,
                expected_stack: &[],
                expected_gas: 0,
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
                .field("expected_gas", &self.expected_gas)
                .finish()
        }
    }

    #[cfg(feature = "llvm")]
    fn with_llvm_context(f: impl FnOnce(&LlvmContext)) {
        thread_local! {
            static TLS_LLVM_CONTEXT: LlvmContext = LlvmContext::create();
        }

        TLS_LLVM_CONTEXT.with(f);
    }

    fn run_case(test_case: &TestCase<'_>) {
        #[cfg(feature = "llvm")]
        run_case_llvm(test_case);
        #[cfg(not(feature = "llvm"))]
        let _ = test_case;
    }

    #[cfg(feature = "llvm")]
    fn run_case_llvm(test_case: &TestCase<'_>) {
        with_llvm_context(|context| {
            let make_backend = |opt_level| JitEvmLlvmBackend::new(context, opt_level).unwrap();
            run_case_generic(test_case, make_backend);
        });
    }

    fn run_case_generic<B: Backend>(
        test_case: &TestCase<'_>,
        make_backend: impl Fn(OptimizationLevel) -> B,
    ) {
        println!("Running {test_case:#?}\n");

        println!("--- not optimized ---");
        run_case_built(test_case, &mut JitEvm::new(make_backend(OptimizationLevel::None)));

        println!("--- optimized ---");
        run_case_built(test_case, &mut JitEvm::new(make_backend(OptimizationLevel::Aggressive)));
    }

    fn run_case_built<B: Backend>(test_case: &TestCase<'_>, jit: &mut JitEvm<B>) {
        let TestCase { bytecode, spec_id, expected_return, expected_stack, expected_gas } =
            *test_case;
        jit.set_disable_gas(false);
        let f = jit.compile(bytecode, spec_id).unwrap();

        let mut stack = EvmStack::new();
        let mut stack_len = 0;
        with_evm_context(bytecode, |ecx| {
            let table =
                spec_to_generic!(test_case.spec_id, op::make_instruction_table::<_, SPEC>());
            let mut interpreter = ecx.to_interpreter(Default::default());
            let memory = interpreter.take_memory();
            interpreter.run(memory, &table, &mut interpreter::DummyHost::default());
            assert_eq!(
                interpreter.instruction_result, expected_return,
                "interpreter return value mismatch"
            );
            assert_eq!(interpreter.stack.data(), expected_stack, "interpreter stack mismatch");
            assert_eq!(interpreter.gas.spent(), expected_gas, "interpreter gas mismatch");

            let actual_return = unsafe { f.call(Some(&mut stack), Some(&mut stack_len), ecx) };
            assert_eq!(actual_return, expected_return, "return value mismatch");
            let actual_stack =
                stack.as_slice().iter().take(stack_len).map(|x| x.to_u256()).collect::<Vec<_>>();
            assert_eq!(actual_stack, *expected_stack, "stack mismatch");
            assert_eq!(ecx.gas.spent(), expected_gas, "gas mismatch");
        });
    }

    // ---

    #[test]
    fn fibonacci() {
        #[cfg(feature = "llvm")]
        with_llvm_context(|context| {
            let make_backend = |opt_level| JitEvmLlvmBackend::new(context, opt_level).unwrap();
            fibonacci_generic(make_backend);
        });
    }

    fn fibonacci_generic<B: Backend>(make_backend: impl Fn(OptimizationLevel) -> B) {
        println!("--- not optimized ---");
        let mut jit = JitEvm::new(make_backend(OptimizationLevel::None));
        run_fibonacci_tests(&mut jit);

        println!("--- optimized ---");
        let mut jit = JitEvm::new(make_backend(OptimizationLevel::Aggressive));
        run_fibonacci_tests(&mut jit);
    }

    fn run_fibonacci_tests<B: Backend>(jit: &mut JitEvm<B>) {
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
            let f = jit.compile(&code, DEF_SPEC).unwrap();

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
}

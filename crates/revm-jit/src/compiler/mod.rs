//! JIT compiler implementation.

use crate::{Backend, Builder, Bytecode, EvmContext, EvmStack, JitEvmFn, Result};
use revm_interpreter::{Contract, Gas};
use revm_jit_backend::{eyre::ensure, Attribute, FunctionAttributeLocation, OptimizationLevel};
use revm_jit_builtins::Builtins;
use revm_jit_context::RawJitEvmFn;
use revm_primitives::{Env, SpecId};
use std::{
    io::Write,
    mem,
    path::{Path, PathBuf},
};

// TODO: Add `nuw`/`nsw` flags to stack length arithmetic.

// TODO: Somehow have a config to tell the backend to assume that stack stores are unobservable,
// making it eliminate redundant stores for values outside the stack length when optimized away.
// E.g. `PUSH0 POP` gets fully optimized away, but the `store i256 0, ptr %stack` will still get
// emitted.
// Use this when `stack` is passed in arguments.

// TODO: Test on big-endian hardware.
// It probably doesn't work when loading Rust U256 into native endianness.

// TODO: Add FileCheck codegen tests.

mod translate;
use translate::{FcxConfig, FunctionCx};

/// EVM bytecode compiler.
///
/// This currently represents one single-threaded IR context and module, which can be used to
/// compile multiple functions.
#[allow(missing_debug_implementations)]
pub struct JitEvm<B: Backend> {
    name: Option<String>,
    backend: B,
    out_dir: Option<PathBuf>,
    config: FcxConfig,
    builtins: Builtins<B>,

    dump_assembly: bool,
    dump_unopt_assembly: bool,

    function_counter: u32,
    finalized: bool,
}

impl<B: Backend> JitEvm<B> {
    /// Creates a new instance of the JIT compiler with the given backend.
    pub fn new(backend: B) -> Self {
        Self {
            name: None,
            backend,
            out_dir: None,
            config: FcxConfig::default(),
            function_counter: 0,
            builtins: Builtins::new(),
            dump_assembly: true,
            dump_unopt_assembly: false,
            finalized: false,
        }
    }

    /// Sets the name of the module.
    pub fn set_name(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.backend.set_module_name(&name);
        self.name = Some(name);
    }

    fn with_name<T>(&mut self, name: impl FnOnce() -> String, f: impl FnOnce(&mut Self) -> T) -> T {
        let none = self.name.is_none();
        if none {
            self.set_name(name());
        }
        let r = f(self);
        if none {
            self.name = None;
        }
        r
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
    /// Note that this does not disable gas usage in certain instructions, mainly the ones that
    /// are implemented as builtins.
    ///
    /// Use with care, as executing a function with gas disabled may result in an infinite loop.
    ///
    /// Defaults to `false`.
    pub fn set_disable_gas(&mut self, disable_gas: bool) {
        self.config.gas_disabled = disable_gas;
    }

    /// Translates the given EVM bytecode into an internal function.
    pub fn translate(
        &mut self,
        name: Option<&str>,
        bytecode: &[u8],
        spec_id: SpecId,
    ) -> Result<B::FuncId> {
        ensure!(!self.finalized, "cannot compile more functions after finalizing the module");
        let bytecode = debug_time!("parse", || self.parse(bytecode, spec_id))?;
        debug_time!("translate", || self.translate_inner(name, &bytecode))
    }

    /// Compiles the given EVM bytecode into a JIT function.
    pub fn compile(
        &mut self,
        name: Option<&str>,
        bytecode: &[u8],
        spec_id: SpecId,
    ) -> Result<JitEvmFn> {
        self.with_name(
            || name.unwrap_or("evm").to_string(),
            |this| {
                let id = this.translate(name, bytecode, spec_id)?;
                this.jit_function(id)
            },
        )
    }

    /// Finalizes the module and JITs the given function.
    pub fn jit_function(&mut self, id: B::FuncId) -> Result<JitEvmFn> {
        if !self.finalized {
            trace_time!("finalize", || self.finalize())?;
            self.finalized = true;
        }
        let addr = trace_time!("get_function", || self.backend.jit_function(id))?;
        Ok(JitEvmFn::new(unsafe { std::mem::transmute::<usize, RawJitEvmFn>(addr) }))
    }

    /// Frees a single function.
    ///
    /// Note that this will not reset the state of the internal module even if all functions are
    /// freed with this function. Use `free_all_functions` to reset the module.
    ///
    /// # Safety
    ///
    /// Because this function invalidates any pointers retrived from the corresponding module, it
    /// should only be used when none of the functions from that module are currently executing and
    /// none of the `fn` pointers are called afterwards.
    pub unsafe fn free_function(&mut self, id: B::FuncId) -> Result<()> {
        self.backend.free_function(id)
    }

    /// Frees all functions and resets the state of the internal module, allowing for new functions
    /// to be compiled.
    ///
    /// # Safety
    ///
    /// Because this function invalidates any pointers retrived from the corresponding module, it
    /// should only be used when none of the functions from that module are currently executing and
    /// none of the `fn` pointers are called afterwards.
    pub unsafe fn free_all_functions(&mut self) -> Result<()> {
        self.builtins.clear();
        self.function_counter = 0;
        self.finalized = false;
        self.backend.free_all_functions()
    }

    fn parse<'a>(&mut self, bytecode: &'a [u8], spec_id: SpecId) -> Result<Bytecode<'a>> {
        let mut bytecode = trace_time!("new bytecode", || Bytecode::new(bytecode, spec_id));
        trace_time!("analyze", || bytecode.analyze())?;
        Ok(bytecode)
    }

    fn translate_inner(
        &mut self,
        name: Option<&str>,
        bytecode: &Bytecode<'_>,
    ) -> Result<B::FuncId> {
        let name = name.unwrap_or("evm_bytecode");
        let mname = self.mangle_name(name, bytecode.spec_id);

        if let Some(dump_dir) = &self.dump_dir() {
            trace_time!("dump bytecode", || Self::dump_bytecode(dump_dir, bytecode))?;
        }

        let (bcx, id) = trace_time!("make builder", || Self::make_builder(
            &mut self.backend,
            &self.config,
            &mname,
        ))?;
        trace_time!("translate", || FunctionCx::translate(
            bcx,
            &self.config,
            &mut self.builtins,
            bytecode,
        ))?;

        Ok(id)
    }

    fn finalize(&mut self) -> Result<()> {
        let verify = |b: &mut B| trace_time!("verify", || b.verify_module());
        if let Some(dump_dir) = &self.dump_dir() {
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

        trace_time!("optimize", || self.backend.optimize_module())?;

        if let Some(dump_dir) = &self.dump_dir() {
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

        Ok(())
    }

    fn make_builder<'a>(
        backend: &'a mut B,
        config: &FcxConfig,
        name: &str,
    ) -> Result<(B::Builder<'a>, B::FuncId)> {
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
        let (mut bcx, id) = backend.build_function(name, ret, params, param_names, linkage)?;

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

        Ok((bcx, id))
    }

    fn dump_bytecode(dump_dir: &Path, bytecode: &Bytecode<'_>) -> Result<()> {
        fn extra_ext(p: &Path, ext: &str) -> PathBuf {
            p.with_file_name(format!("{}.{ext}", p.file_name().unwrap().to_str().unwrap()))
        }

        let fname = dump_dir.join("evm").with_extension(format!("{:?}", bytecode.spec_id));
        std::fs::write(extra_ext(&fname, "bin"), bytecode.code)?;

        std::fs::write(extra_ext(&fname, "hex"), revm_primitives::hex::encode(bytecode.code))?;

        let file = std::fs::File::create(extra_ext(&fname, "txt"))?;
        let mut file = std::io::BufWriter::new(file);

        let header = format!("{:^6} | {:^6} | {:^80} | {}", "ic", "pc", "opcode", "instruction");
        writeln!(file, "{header}")?;
        writeln!(file, "{}", "-".repeat(header.len()))?;
        for (inst, (pc, opcode)) in bytecode.opcodes().with_pc().enumerate() {
            let data = bytecode.inst(inst);
            let opcode = opcode.to_string();
            writeln!(file, "{inst:>6} | {pc:>6} | {opcode:<80} | {data:?}")?;
        }

        file.flush()?;

        Ok(())
    }

    fn mangle_name(&mut self, base: &str, spec_id: SpecId) -> String {
        let name = format!("{base}_{spec_id:?}_{}", self.function_counter);
        self.function_counter += 1;
        name
    }

    fn dump_dir(&self) -> Option<PathBuf> {
        let mut dump_dir = self.out_dir.clone()?;
        if let Some(name) = &self.name {
            dump_dir.push(name.replace(char::is_whitespace, "_"));
        }
        if !dump_dir.exists() {
            let _ = std::fs::create_dir_all(&dump_dir);
        }
        Some(dump_dir)
    }
}

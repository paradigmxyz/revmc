//! EVM bytecode compiler implementation.

use crate::{Backend, Builder, Bytecode, EvmCompilerFn, EvmContext, EvmStack, Result};
use revm_interpreter::{Contract, Gas};
use revm_primitives::{Bytes, Env, Eof, SpecId, EOF_MAGIC_BYTES};
use revmc_backend::{
    eyre::{ensure, eyre},
    Attribute, FunctionAttributeLocation, Linkage, OptimizationLevel,
};
use revmc_builtins::Builtins;
use revmc_context::RawEvmCompilerFn;
use std::{
    borrow::Cow,
    fs,
    io::{self, Write},
    mem,
    path::{Path, PathBuf},
};

// TODO: Somehow have a config to tell the backend to assume that stack stores are unobservable,
// making it eliminate redundant stores for values outside the stack length when optimized away.
// E.g. `PUSH0 POP` gets fully optimized away, but the `store i256 0, ptr %stack` will still get
// emitted.
// Use this when `stack` is passed in arguments.

// TODO: Get rid of `cfg!(target_endian)` calls.

// TODO: Test on big-endian hardware.
// It probably doesn't work when loading Rust U256 into native endianness.

mod translate;
use translate::{FcxConfig, FunctionCx};

/// EVM bytecode compiler.
///
/// This currently represents one single-threaded IR context and module, which can be used to
/// compile multiple functions as JIT or AOT.
///
/// Functions can be incrementally added with [`translate`], and then either written to an object
/// file with [`write_object`] when in AOT mode, or JIT-compiled with [`jit_function`].
///
/// Performing either of these operations finalizes the module, and no more functions can be added
/// afterwards until [`clear`] is called, which will reset the module to its initial state.
///
/// [`translate`]: EvmCompiler::translate
/// [`write_object`]: EvmCompiler::write_object
/// [`jit_function`]: EvmCompiler::jit_function
/// [`clear`]: EvmCompiler::clear
#[allow(missing_debug_implementations)]
pub struct EvmCompiler<B: Backend> {
    name: Option<String>,
    backend: B,
    out_dir: Option<PathBuf>,
    config: FcxConfig,
    builtins: Builtins<B>,

    dump_assembly: bool,
    dump_unopt_assembly: bool,

    finalized: bool,
}

impl<B: Backend> EvmCompiler<B> {
    /// Creates a new instance of the compiler with the given backend.
    pub fn new(backend: B) -> Self {
        Self {
            name: None,
            backend,
            out_dir: None,
            config: FcxConfig::default(),
            builtins: Builtins::new(),
            dump_assembly: true,
            dump_unopt_assembly: false,
            finalized: false,
        }
    }

    /// Sets the name of the module.
    pub fn set_module_name(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.backend.set_module_name(&name);
        self.name = Some(name);
    }

    fn is_aot(&self) -> bool {
        self.backend.is_aot()
    }

    fn is_jit(&self) -> bool {
        !self.is_aot()
    }

    /// Returns the output directory.
    pub fn out_dir(&self) -> Option<&Path> {
        self.out_dir.as_deref()
    }

    /// Dumps intermediate outputs and other debug info to the given directory after compilation.
    ///
    /// Disables dumping if `output_dir` is `None`.
    pub fn set_dump_to(&mut self, output_dir: Option<PathBuf>) {
        self.backend.set_is_dumping(output_dir.is_some());
        self.config.comments = output_dir.is_some();
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
    pub fn debug_assertions(&mut self, yes: bool) {
        self.backend.set_debug_assertions(yes);
        self.config.debug_assertions = yes;
    }

    /// Sets whether to enable frame pointers.
    ///
    /// This is useful for profiling and debugging, but it incurs a very slight performance penalty.
    ///
    /// Defaults to `cfg!(debug_assertions)`.
    pub fn frame_pointers(&mut self, yes: bool) {
        self.config.frame_pointers = yes;
    }

    /// Sets whether to validate input EOF containers.
    ///
    /// **An invalid EOF container will likely results in a panic.**
    ///
    /// Defaults to `true`.
    pub fn validate_eof(&mut self, yes: bool) {
        self.config.validate_eof = yes;
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
    pub fn local_stack(&mut self, yes: bool) {
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
    pub fn inspect_stack_length(&mut self, yes: bool) {
        self.config.inspect_stack_length = yes;
    }

    /// Sets whether to enable stack bound checks.
    ///
    /// Ignored for EOF bytecodes, as they are assumed to be correct.
    ///
    /// Defaults to `true`.
    ///
    /// # Safety
    ///
    /// Removing stack length checks may improve compilation speed and performance, but will result
    /// in **undefined behavior** if the stack length overflows at runtime, rather than a
    /// [`StackUnderflow`]/[`StackOverflow`] result.
    ///
    /// [`StackUnderflow`]: crate::interpreter::InstructionResult::StackUnderflow
    /// [`StackOverflow`]: crate::interpreter::InstructionResult::StackOverflow
    pub unsafe fn stack_bound_checks(&mut self, yes: bool) {
        self.config.stack_bound_checks = yes;
    }

    /// Sets whether to track gas costs.
    ///
    /// Disabling this will greatly improves compilation speed and performance, at the cost of not
    /// being able to check for gas exhaustion.
    ///
    /// Note that this does not disable gas usage in certain instructions, mainly the ones that
    /// are implemented as builtins.
    ///
    /// Use with care, as executing a function with gas disabled may result in an infinite loop.
    ///
    /// Defaults to `true`.
    pub fn gas_metering(&mut self, yes: bool) {
        self.config.gas_metering = yes;
    }

    /// Translates the given EVM bytecode into an internal function.
    ///
    /// NOTE: `name` must be unique for each function, as it is used as the name of the final
    /// symbol.
    pub fn translate<'a>(
        &mut self,
        name: &str,
        input: impl Into<EvmCompilerInput<'a>>,
        spec_id: SpecId,
    ) -> Result<B::FuncId> {
        ensure!(cfg!(target_endian = "little"), "only little-endian is supported");
        ensure!(!self.finalized, "cannot compile more functions after finalizing the module");
        let bytecode = self.parse(input.into(), spec_id)?;
        self.translate_inner(name, &bytecode)
    }

    /// (JIT) Compiles the given EVM bytecode into a JIT function.
    ///
    /// See [`translate`](Self::translate) for more information.
    ///
    /// # Safety
    ///
    /// The returned function pointer is owned by the module, and must not be called after the
    /// module is cleared or the function is freed.
    pub unsafe fn jit<'a>(
        &mut self,
        name: &str,
        bytecode: impl Into<EvmCompilerInput<'a>>,
        spec_id: SpecId,
    ) -> Result<EvmCompilerFn> {
        let id = self.translate(name, bytecode.into(), spec_id)?;
        unsafe { self.jit_function(id) }
    }

    /// (JIT) Finalizes the module and JITs the given function.
    ///
    /// # Safety
    ///
    /// The returned function pointer is owned by the module, and must not be called after the
    /// module is cleared or the function is freed.
    pub unsafe fn jit_function(&mut self, id: B::FuncId) -> Result<EvmCompilerFn> {
        ensure!(self.is_jit(), "cannot JIT functions during AOT compilation");
        self.finalize()?;
        let addr = self.backend.jit_function(id)?;
        debug_assert!(addr != 0);
        Ok(EvmCompilerFn::new(unsafe { std::mem::transmute::<usize, RawEvmCompilerFn>(addr) }))
    }

    /// (AOT) Writes the compiled object to the given file.
    pub fn write_object_to_file(&mut self, path: &Path) -> Result<()> {
        let file = fs::File::create(path)?;
        let mut writer = io::BufWriter::new(file);
        self.write_object(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    /// (AOT) Finalizes the module and writes the compiled object to the given writer.
    pub fn write_object<W: io::Write>(&mut self, w: W) -> Result<()> {
        ensure!(self.is_aot(), "cannot write AOT object during JIT compilation");
        self.finalize()?;
        self.backend.write_object(w)
    }

    /// (JIT) Frees the memory associated with a single function.
    ///
    /// Note that this will not reset the state of the internal module even if all functions are
    /// freed with this function. Use [`clear`] to reset the module.
    ///
    /// [`clear`]: EvmCompiler::clear
    ///
    /// # Safety
    ///
    /// Because this function invalidates any pointers retrieved from the corresponding module, it
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
    /// Because this function invalidates any pointers retrieved from the corresponding module, it
    /// should only be used when none of the functions from that module are currently executing and
    /// none of the `fn` pointers are called afterwards.
    pub unsafe fn clear(&mut self) -> Result<()> {
        self.builtins.clear();
        self.finalized = false;
        self.backend.free_all_functions()
    }

    /// Parses the given EVM bytecode. Not public API.
    #[doc(hidden)] // Not public API.
    pub fn parse<'a>(
        &mut self,
        input: EvmCompilerInput<'a>,
        spec_id: SpecId,
    ) -> Result<Bytecode<'a>> {
        let bytecode;
        let eof;
        match input {
            EvmCompilerInput::Code(code) => {
                bytecode = code;
                if spec_id.is_enabled_in(SpecId::PRAGUE_EOF) && code.starts_with(&EOF_MAGIC_BYTES) {
                    eof = Some(Cow::Owned(Eof::decode(Bytes::copy_from_slice(code))?));
                } else {
                    eof = None;
                }
            }
            EvmCompilerInput::Eof(e) => {
                bytecode = &e.raw[..];
                eof = Some(Cow::Borrowed(e));
            }
        }
        if let Some(eof) = &eof {
            self.do_validate_eof(eof)?;
        }

        let mut bytecode = Bytecode::new(bytecode, eof, spec_id);
        bytecode.analyze()?;
        if let Some(dump_dir) = &self.dump_dir() {
            Self::dump_bytecode(dump_dir, &bytecode)?;
        }
        Ok(bytecode)
    }

    fn do_validate_eof(&self, eof: &Eof) -> Result<()> {
        if !self.config.validate_eof {
            return Ok(());
        }
        revm_interpreter::analysis::validate_eof(eof).map_err(|e| match e {
            revm_interpreter::analysis::EofError::Decode(e) => e.into(),
            revm_interpreter::analysis::EofError::Validation(e) => {
                eyre!("validation error: {e:?}")
            }
        })
    }

    #[instrument(name = "translate", level = "debug", skip_all)]
    fn translate_inner(&mut self, name: &str, bytecode: &Bytecode<'_>) -> Result<B::FuncId> {
        ensure!(self.backend.function_name_is_unique(name), "function name `{name}` is not unique");
        let linkage = Linkage::Public;
        let (bcx, id) = Self::make_builder(&mut self.backend, &self.config, name, linkage)?;
        FunctionCx::translate(bcx, self.config, &mut self.builtins, bytecode)?;
        Ok(id)
    }

    #[instrument(level = "debug", skip_all)]
    fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;

        if let Some(dump_dir) = &self.dump_dir() {
            let path = dump_dir.join("unopt").with_extension(self.backend.ir_extension());
            self.dump_ir(&path)?;

            // Dump IR before verifying for better debugging.
            self.verify_module()?;

            if self.dump_assembly && self.dump_unopt_assembly {
                let path = dump_dir.join("unopt.s");
                self.dump_disasm(&path)?;
            }
        } else {
            self.verify_module()?;
        }

        self.optimize_module()?;

        if let Some(dump_dir) = &self.dump_dir() {
            let path = dump_dir.join("opt").with_extension(self.backend.ir_extension());
            self.dump_ir(&path)?;

            if self.dump_assembly {
                let path = dump_dir.join("opt.s");
                self.dump_disasm(&path)?;
            }
        }

        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    fn make_builder<'a>(
        backend: &'a mut B,
        config: &FcxConfig,
        name: &str,
        linkage: Linkage,
    ) -> Result<(B::Builder<'a>, B::FuncId)> {
        fn size_align<T>(i: usize) -> (usize, usize, usize) {
            (i, mem::size_of::<T>(), mem::align_of::<T>())
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
                size_align::<Gas>(0),
                size_align::<EvmStack>(1),
                size_align::<usize>(2),
                size_align::<Env>(3),
                size_align::<Contract>(4),
                size_align::<EvmContext<'_>>(5),
            ],
        );
        debug_assert_eq!(params.len(), param_names.len());
        let (mut bcx, id) = backend.build_function(name, ret, params, param_names, linkage)?;

        // Function attributes.
        let function_attributes = default_attrs::for_fn()
            .chain(config.frame_pointers.then_some(Attribute::AllFramePointers))
            // We can unwind in panics, which are present only in debug assertions.
            .chain((!config.debug_assertions).then_some(Attribute::NoUnwind));
        for attr in function_attributes {
            bcx.add_function_attribute(None, attr, FunctionAttributeLocation::Function);
        }

        // Pointer argument attributes.
        if !config.debug_assertions {
            for &(i, size, align) in ptr_attrs {
                let attrs = default_attrs::for_sized_ptr((size, align))
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

    #[instrument(level = "debug", skip_all)]
    fn dump_ir(&mut self, path: &Path) -> Result<()> {
        self.backend.dump_ir(path)
    }

    #[instrument(level = "debug", skip_all)]
    fn dump_disasm(&mut self, path: &Path) -> Result<()> {
        self.backend.dump_disasm(path)
    }

    #[instrument(level = "debug", skip_all)]
    fn verify_module(&mut self) -> Result<()> {
        self.backend.verify_module()
    }

    #[instrument(level = "debug", skip_all)]
    fn optimize_module(&mut self) -> Result<()> {
        self.backend.optimize_module()
    }

    #[instrument(level = "debug", skip_all)]
    fn dump_bytecode(dump_dir: &Path, bytecode: &Bytecode<'_>) -> Result<()> {
        {
            let file = fs::File::create(dump_dir.join("bytecode.txt"))?;
            let mut writer = io::BufWriter::new(file);
            write!(writer, "{bytecode}")?;
            writer.flush()?;
        }

        {
            let file = fs::File::create(dump_dir.join("bytecode.dbg.txt"))?;
            let mut writer = io::BufWriter::new(file);
            writeln!(writer, "{bytecode:#?}")?;
            writer.flush()?;
        }

        Ok(())
    }

    fn dump_dir(&self) -> Option<PathBuf> {
        let mut dump_dir = self.out_dir.clone()?;
        if let Some(name) = &self.name {
            dump_dir.push(name.replace(char::is_whitespace, "_"));
        }
        if !dump_dir.exists() {
            let _ = fs::create_dir_all(&dump_dir);
        }
        Some(dump_dir)
    }
}

/// [`EvmCompiler`] input.
#[allow(missing_debug_implementations)]
pub enum EvmCompilerInput<'a> {
    /// EVM bytecode. Can also be raw EOF code, which will be parsed.
    Code(&'a [u8]),
    /// Already-parsed EOF container.
    Eof(&'a Eof),
}

impl<'a> From<&'a [u8]> for EvmCompilerInput<'a> {
    fn from(code: &'a [u8]) -> Self {
        EvmCompilerInput::Code(code)
    }
}

impl<'a> From<&'a Vec<u8>> for EvmCompilerInput<'a> {
    fn from(code: &'a Vec<u8>) -> Self {
        EvmCompilerInput::Code(code)
    }
}

impl<'a> From<&'a Bytes> for EvmCompilerInput<'a> {
    fn from(code: &'a Bytes) -> Self {
        EvmCompilerInput::Code(code)
    }
}

impl<'a> From<&'a Eof> for EvmCompilerInput<'a> {
    fn from(eof: &'a Eof) -> Self {
        EvmCompilerInput::Eof(eof)
    }
}

#[allow(dead_code)]
mod default_attrs {
    use revmc_backend::Attribute;

    pub(crate) fn for_fn() -> impl Iterator<Item = Attribute> {
        [
            Attribute::WillReturn,      // Always returns.
            Attribute::NoSync,          // No thread synchronization.
            Attribute::NativeTargetCpu, // Optimization.
            Attribute::Speculatable,    // No undefined behavior.
            Attribute::NoRecurse,       // Revm is not recursive.
        ]
        .into_iter()
    }

    pub(crate) fn for_param() -> impl Iterator<Item = Attribute> {
        [Attribute::NoUndef].into_iter()
    }

    pub(crate) fn for_ptr() -> impl Iterator<Item = Attribute> {
        for_param().chain([Attribute::NoCapture])
    }

    pub(crate) fn for_sized_ptr((size, align): (usize, usize)) -> impl Iterator<Item = Attribute> {
        for_ptr().chain([Attribute::Dereferenceable(size as u64), Attribute::Align(align as u64)])
    }

    pub(crate) fn for_ptr_t<T>() -> impl Iterator<Item = Attribute> {
        for_sized_ptr(size_align::<T>())
    }

    pub(crate) fn for_ref() -> impl Iterator<Item = Attribute> {
        for_ptr().chain([Attribute::NonNull, Attribute::NoAlias])
    }

    pub(crate) fn for_sized_ref((size, align): (usize, usize)) -> impl Iterator<Item = Attribute> {
        for_ref().chain([Attribute::Dereferenceable(size as u64), Attribute::Align(align as u64)])
    }

    pub(crate) fn for_ref_t<T>() -> impl Iterator<Item = Attribute> {
        for_sized_ref(size_align::<T>())
    }

    pub(crate) fn size_align<T>() -> (usize, usize) {
        (std::mem::size_of::<T>(), std::mem::align_of::<T>())
    }
}

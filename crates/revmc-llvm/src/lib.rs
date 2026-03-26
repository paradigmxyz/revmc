#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[macro_use]
extern crate tracing;

use eyre::eyre;
use inkwell::{
    AddressSpace, IntPredicate, OptimizationLevel,
    attributes::{Attribute, AttributeLoc},
    basic_block::BasicBlock,
    debug_info::{
        AsDIScope, DICompileUnit, DIFlags, DIFlagsConstants, DISubprogram, DWARFEmissionKind,
        DWARFSourceLanguage, DebugInfoBuilder,
    },
    execution_engine::ExecutionEngine,
    module::{FlagBehavior, Module},
    passes::PassBuilderOptions,
    support::error_handling::install_fatal_error_handler,
    targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
    },
    types::{
        AnyType, AnyTypeEnum, BasicType, BasicTypeEnum, FunctionType, IntType, PointerType,
        StringRadix, VoidType,
    },
    values::{
        BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, InstructionValue,
        PointerValue,
    },
};
use object::{Object, ObjectSymbol};
use revmc_backend::{
    Backend, BackendTypes, Builder, IntCC, Result, TailCallKind, TypeMethods, U256, eyre,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    borrow::Cow,
    iter,
    path::{Path, PathBuf},
    sync::{Once, OnceLock},
};

pub use inkwell::{self, context::Context};

mod dh;
pub mod orc;

mod utils;
pub(crate) use utils::*;

const DEFAULT_WEIGHT: u32 = 20000;

/// The LLVM-based EVM bytecode compiler backend.
#[derive(Debug)]
#[must_use]
pub struct EvmLlvmBackend {
    cx: &'static Context,
    _dh: dh::DiagnosticHandlerGuard,
    bcx: inkwell::builder::Builder<'static>,
    module: Module<'static>,
    exec_engine: Option<ExecutionEngine<'static>>,
    machine: TargetMachine,

    ty_void: VoidType<'static>,
    ty_ptr: PointerType<'static>,
    ty_i1: IntType<'static>,
    ty_i8: IntType<'static>,
    ty_i32: IntType<'static>,
    ty_i64: IntType<'static>,
    ty_i256: IntType<'static>,
    ty_isize: IntType<'static>,

    aot: bool,
    debug_assertions: bool,
    is_dumping: bool,
    opt_level: OptimizationLevel,
    /// Separate from `functions` to have always increasing IDs.
    function_counter: u32,
    functions: FxHashMap<u32, (String, FunctionValue<'static>)>,
    /// Symbol names that have been registered via `add_global_mapping` in the MCJIT engine.
    /// Used to avoid re-registering builtins when a new module is created after `clear_ir`.
    mapped_symbols: FxHashSet<String>,

    /// Debug info source file path set by the compiler.
    debug_file: Option<PathBuf>,
    /// LLVM debug info builder and compile unit, created lazily when `debug_file` is set.
    di_state: Option<DiState>,
}

/// LLVM debug info state for a module.
struct DiState {
    dibuilder: DebugInfoBuilder<'static>,
    compile_unit: DICompileUnit<'static>,
    finalized: bool,
}

impl std::fmt::Debug for DiState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiState").field("finalized", &self.finalized).finish()
    }
}

unsafe impl Send for EvmLlvmBackend {}

impl EvmLlvmBackend {
    /// Creates a new LLVM backend for the host machine.
    ///
    /// Use [`new_for_target`](Self::new_for_target) to create a backend for a specific target.
    pub fn new(aot: bool, opt_level: revmc_backend::OptimizationLevel) -> Result<Self> {
        Self::new_for_target(aot, opt_level, &revmc_backend::Target::Native)
    }

    /// Creates a new LLVM backend for the given target.
    #[instrument(name = "new_llvm_backend", level = "debug", skip_all)]
    pub fn new_for_target(
        aot: bool,
        opt_level: revmc_backend::OptimizationLevel,
        target: &revmc_backend::Target,
    ) -> Result<Self> {
        init()?;

        let cx = get_context();

        let opt_level = convert_opt_level(opt_level);

        let target_info = TargetInfo::new(target)?;
        let target = &target_info.target;
        let machine = target
            .create_target_machine(
                &target_info.triple,
                &target_info.cpu,
                &target_info.features,
                opt_level,
                if aot { RelocMode::PIC } else { RelocMode::Static },
                if aot { CodeModel::Default } else { CodeModel::JITDefault },
            )
            .ok_or_else(|| eyre::eyre!("failed to create target machine"))?;

        let module = create_module(cx, &machine, aot)?;

        let exec_engine = if aot {
            None
        } else {
            if !target.has_jit() {
                return Err(eyre::eyre!("target {:?} does not support JIT", target.get_name()));
            }
            if !target.has_target_machine() {
                return Err(eyre::eyre!(
                    "target {:?} does not have target machine",
                    target.get_name()
                ));
            }
            Some(module.create_jit_execution_engine(opt_level).map_err(error_msg)?)
        };

        let bcx = cx.create_builder();

        let ty_void = cx.void_type();
        let ty_i1 = cx.bool_type();
        let ty_i8 = cx.i8_type();
        let ty_i32 = cx.i32_type();
        let ty_i64 = cx.i64_type();
        let ty_i256 =
            cx.custom_width_int_type(std::num::NonZeroU32::new(256).unwrap()).expect("i256");
        let ty_isize = cx.ptr_sized_int_type(&machine.get_target_data(), None);
        let ty_ptr = cx.ptr_type(AddressSpace::default());
        Ok(Self {
            cx,
            _dh: dh::DiagnosticHandlerGuard::new(cx),
            bcx,
            module,
            exec_engine,
            machine,
            ty_void,
            ty_i1,
            ty_i8,
            ty_i32,
            ty_i64,
            ty_i256,
            ty_isize,
            ty_ptr,
            aot,
            debug_assertions: cfg!(debug_assertions),
            is_dumping: false,
            opt_level,
            function_counter: 0,
            functions: FxHashMap::default(),
            mapped_symbols: FxHashSet::default(),
            debug_file: None,
            di_state: None,
        })
    }

    /// Returns the LLVM context.
    #[inline]
    pub fn cx(&self) -> &Context {
        self.cx
    }

    fn exec_engine(&self) -> &ExecutionEngine<'static> {
        assert!(!self.aot, "requested JIT execution engine on AOT");
        self.exec_engine.as_ref().expect("missing JIT execution engine")
    }

    fn fn_type(
        &self,
        ret: Option<BasicTypeEnum<'static>>,
        params: &[BasicTypeEnum<'static>],
    ) -> FunctionType<'static> {
        let params = params.iter().copied().map(Into::into).collect::<Vec<_>>();
        match ret {
            Some(ret) => ret.fn_type(&params, false),
            None => self.ty_void.fn_type(&params, false),
        }
    }

    /// Returns the given name if IR output is being dumped, otherwise an empty string.
    /// LLVM skips internal name processing for empty names, avoiding overhead when names
    /// are not needed for readability.
    #[inline]
    fn name<'a>(&self, name: &'a str) -> &'a str {
        if self.is_dumping { name } else { "" }
    }

    fn id_to_name(&self, id: u32) -> &str {
        &self.functions[&id].0
    }

    /// Lazily initializes the debug info builder and compile unit for the module.
    fn ensure_di_state(&mut self) {
        if self.di_state.is_some() {
            return;
        }
        let Some(debug_file) = &self.debug_file else { return };

        let filename =
            debug_file.file_name().map(|f| f.to_string_lossy()).unwrap_or_default().into_owned();
        let directory =
            debug_file.parent().map(|p| p.to_string_lossy()).unwrap_or_default().into_owned();

        // Add required module flags for debug info.
        self.module.add_basic_value_flag(
            "Debug Info Version",
            FlagBehavior::Warning,
            self.ty_i32.const_int(inkwell::debug_info::debug_metadata_version() as u64, false),
        );
        self.module.add_basic_value_flag(
            "Dwarf Version",
            FlagBehavior::Warning,
            self.ty_i32.const_int(5, false),
        );

        let is_optimized = self.opt_level != OptimizationLevel::None;
        let mut flags = Vec::new();
        flags.push(match self.opt_level {
            OptimizationLevel::None => "-O0",
            OptimizationLevel::Less => "-O1",
            OptimizationLevel::Default => "-O2",
            OptimizationLevel::Aggressive => "-O3",
        });
        flags.push(if self.aot { "--aot" } else { "--jit" });
        let flags = flags.join(" ");

        let (dibuilder, compile_unit) = self.module.create_debug_info_builder(
            true,
            DWARFSourceLanguage::C,
            &filename,
            &directory,
            "revmc",
            is_optimized,
            &flags,
            0,
            "",
            DWARFEmissionKind::Full,
            0,
            false,
            false,
            "",
            "",
        );

        self.di_state = Some(DiState { dibuilder, compile_unit, finalized: false });
    }

    // Delete IR and free JIT-compiled machine code.
    //
    // With MCJIT, machine code pages are owned by `RuntimeDyld` inside the `ExecutionEngine` and
    // are only freed when the engine is dropped. `free_fn_machine_code` is a no-op in modern LLVM.
    // So we drop the old engine entirely and create a fresh one.
    fn clear_module(&mut self) -> Result<()> {
        self.functions.clear();
        self.mapped_symbols.clear();

        // Drop the old DI state before replacing the module, since DIBuilder references the module.
        self.di_state = None;

        // Drop the old execution engine to free machine code memory, then create a fresh module
        // and a new engine.
        self.exec_engine = None;
        self.module = create_module(self.cx, &self.machine, self.aot)?;
        if !self.aot {
            self.exec_engine =
                Some(self.module.create_jit_execution_engine(self.opt_level).map_err(error_msg)?);
        }

        Ok(())
    }
}

impl BackendTypes for EvmLlvmBackend {
    type Type = BasicTypeEnum<'static>;
    type Value = BasicValueEnum<'static>;
    type StackSlot = PointerValue<'static>;
    type BasicBlock = BasicBlock<'static>;
    type Function = FunctionValue<'static>;
}

impl TypeMethods for EvmLlvmBackend {
    fn type_ptr(&self) -> Self::Type {
        self.ty_ptr.into()
    }

    fn type_ptr_sized_int(&self) -> Self::Type {
        self.ty_isize.into()
    }

    fn type_int(&self, bits: u32) -> Self::Type {
        match bits {
            1 => self.ty_i1,
            8 => self.ty_i8,
            16 => self.cx.i16_type(),
            32 => self.ty_i32,
            64 => self.ty_i64,
            128 => self.cx.i128_type(),
            256 => self.ty_i256,
            bits => self
                .cx
                .custom_width_int_type(std::num::NonZeroU32::new(bits).unwrap())
                .expect("custom int type"),
        }
        .into()
    }

    fn type_array(&self, ty: Self::Type, size: u32) -> Self::Type {
        ty.array_type(size).into()
    }

    fn type_bit_width(&self, ty: Self::Type) -> u32 {
        ty.into_int_type().get_bit_width()
    }
}

impl Backend for EvmLlvmBackend {
    type Builder<'a>
        = EvmLlvmBuilder<'a>
    where
        Self: 'a;
    type FuncId = u32;

    fn ir_extension(&self) -> &'static str {
        "ll"
    }

    fn set_module_name(&mut self, name: &str) {
        self.module.set_name(name);
    }

    fn set_is_dumping(&mut self, yes: bool) {
        self.is_dumping = yes;
        self.machine.set_asm_verbosity(yes);
    }

    fn set_debug_assertions(&mut self, yes: bool) {
        self.debug_assertions = yes;
    }

    fn set_debug_file(&mut self, path: Option<PathBuf>) {
        self.debug_file = path;
    }

    fn finalize_debug_info(&mut self) -> Result<()> {
        if let Some(di) = &mut self.di_state
            && !di.finalized
        {
            di.dibuilder.finalize();
            di.finalized = true;
        }
        Ok(())
    }

    fn opt_level(&self) -> revmc_backend::OptimizationLevel {
        convert_opt_level_rev(self.opt_level)
    }

    fn set_opt_level(&mut self, level: revmc_backend::OptimizationLevel) {
        self.opt_level = convert_opt_level(level);
    }

    fn is_aot(&self) -> bool {
        self.aot
    }

    fn function_name_is_unique(&self, name: &str) -> bool {
        self.module.get_function(name).is_none()
    }

    fn dump_ir(&mut self, path: &Path) -> Result<()> {
        self.module.print_to_file(path).map_err(error_msg)
    }

    fn dump_disasm(&mut self, path: &Path) -> Result<()> {
        self.machine.write_to_file(&self.module, FileType::Assembly, path).map_err(error_msg)
    }

    fn build_function(
        &mut self,
        name: &str,
        ret: Option<Self::Type>,
        params: &[Self::Type],
        param_names: &[&str],
        linkage: revmc_backend::Linkage,
    ) -> Result<(Self::Builder<'_>, Self::FuncId)> {
        let (id, function) = if let Some((&id, &(_, function))) =
            self.functions.iter().find(|(_k, (fname, _f))| fname == name)
            && let Some(function2) = self.module.get_function(name)
            && function == function2
        {
            self.bcx.position_at_end(function.get_first_basic_block().unwrap());
            (id, function)
        } else {
            let fn_type = self.fn_type(ret, params);
            let function = self.module.add_function(name, fn_type, Some(convert_linkage(linkage)));
            if self.is_dumping {
                for (i, &name) in param_names.iter().enumerate() {
                    function.get_nth_param(i as u32).expect(name).set_name(self.name(name));
                }
            }

            let entry = self.cx.append_basic_block(function, self.name("entry"));
            self.bcx.position_at_end(entry);

            let id = self.function_counter;
            self.function_counter += 1;
            self.functions.insert(id, (name.to_string(), function));
            (id, function)
        };

        // Attach debug info subprogram if debug is active.
        self.ensure_di_state();
        let debug_scope = if let Some(di) = &self.di_state {
            let file = di.compile_unit.get_file();
            let subroutine_type =
                di.dibuilder.create_subroutine_type(file, None, &[], DIFlags::PUBLIC);
            let subprogram = di.dibuilder.create_function(
                di.compile_unit.as_debug_info_scope(),
                name,
                None,
                file,
                0,
                subroutine_type,
                true,
                true,
                0,
                DIFlags::PUBLIC,
                self.opt_level != OptimizationLevel::None,
            );
            function.set_subprogram(subprogram);
            Some(subprogram)
        } else {
            None
        };

        let builder = EvmLlvmBuilder { backend: self, function, debug_scope };
        Ok((builder, id))
    }

    fn verify_module(&mut self) -> Result<()> {
        self.module.verify().map_err(error_msg)
    }

    fn optimize_module(&mut self) -> Result<()> {
        // We use a custom pipeline instead of `default<O3>` because GVN is extremely slow on
        // the huge single-function modules that EVM compilation produces. Replacing GVN with
        // `early-cse` + `sccp` achieves equivalent or better code size and runtime performance
        // at ~5x faster compile time.
        //
        // The standard `default<O1>` is also slow (~730ms on snailtracer) because the loop
        // analysis infrastructure (LoopInfo, DominatorTree, MemorySSA, LCSSA) is expensive to
        // compute on functions with thousands of basic blocks, even though the loop passes
        // themselves do nothing useful — EVM has no natural loops to optimize.
        //
        // LICM (Loop Invariant Code Motion) helps tight EVM loops by hoisting gas counter and
        // stack slot loads/stores into registers. However, the loop analysis infrastructure is
        // quadratic on large functions — e.g. +430ms on snailtracer (7770 BBs) vs +0ms on
        // fibonacci (45 BBs). We skip it for functions with >4000 basic blocks.
        //
        // Can be overridden with `REVMC_PASSES` env var for experimentation.
        // From `opt --help`, `-passes`.

        static PASSES_OVERRIDE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        static PASSES: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        static PASSES_WITH_LICM: std::sync::OnceLock<String> = std::sync::OnceLock::new();

        let passes_override = PASSES_OVERRIDE.get_or_init(|| std::env::var("REVMC_PASSES").ok());

        let passes = passes_override.as_deref().unwrap_or_else(|| match self.opt_level {
            OptimizationLevel::None => "default<O0>",
            OptimizationLevel::Less
            | OptimizationLevel::Default
            | OptimizationLevel::Aggressive => {
                let total_bbs: u32 =
                    self.module.get_functions().map(|f| f.count_basic_blocks()).sum();
                let passes = if total_bbs > 4000 { &PASSES } else { &PASSES_WITH_LICM };
                passes.get_or_init(|| build_pass_pipeline(total_bbs <= 4000))
            }
        });
        let opts = PassBuilderOptions::create();
        self.module.run_passes(passes, &self.machine, opts).map_err(error_msg)
    }

    fn write_object<W: std::io::Write>(&mut self, mut w: W) -> Result<()> {
        let buffer = self
            .machine
            .write_to_memory_buffer(&self.module, FileType::Object)
            .map_err(error_msg)?;
        w.write_all(buffer.as_slice())?;
        Ok(())
    }

    fn jit_function(&mut self, id: Self::FuncId) -> Result<usize> {
        let name = self.id_to_name(id);
        let addr = self.exec_engine().get_function_address(name)?;
        Ok(addr)
    }

    fn function_sizes(&self) -> Vec<(String, usize)> {
        let buffer = match self.machine.write_to_memory_buffer(&self.module, FileType::Object) {
            Ok(buf) => buf,
            Err(_) => return Vec::new(),
        };
        let data = buffer.as_slice();
        let Ok(obj) = object::File::parse(data) else { return Vec::new() };

        let mut result: Vec<_> = obj
            .symbols()
            .filter(|sym| sym.is_definition())
            .filter_map(|sym| {
                let name = sym.name().ok()?;
                self.functions.values().any(|(n, _)| n == name).then_some(())?;
                Some((name.to_string(), sym.size() as usize))
            })
            .collect();
        result.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
        result
    }

    fn clear_ir(&mut self) -> Result<()> {
        // Drop the old DI state before replacing the module, since DIBuilder references the module.
        self.di_state = None;

        // Remove the old module from the execution engine before replacing it.
        // Without this, each clear_ir() cycle leaks a module in the engine.
        if let Some(exec_engine) = &self.exec_engine {
            exec_engine
                .remove_module(&self.module)
                .map_err(|e| eyre!("failed to remove module: {e}"))?;
        }

        self.module = create_module(self.cx, &self.machine, self.aot)?;
        if let Some(exec_engine) = &self.exec_engine {
            exec_engine.add_module(&self.module).map_err(|()| eyre!("failed to add module"))?;
        }

        Ok(())
    }

    unsafe fn free_function(&mut self, id: Self::FuncId) -> Result<()> {
        let (_, function) = self.functions.remove(&id).unwrap();
        if let Some(exec_engine) = &self.exec_engine {
            exec_engine.free_fn_machine_code(function);
        }
        unsafe { function.delete() };
        Ok(())
    }

    unsafe fn free_all_functions(&mut self) -> Result<()> {
        self.clear_module()
    }
}

/// Cached target information for the host machine.
#[derive(Debug)]
struct TargetInfo {
    triple: TargetTriple,
    target: Target,
    cpu: String,
    features: String,
}

// SAFETY: No mutability is exposed and `TargetTriple` is an owned string.
unsafe impl std::marker::Send for TargetInfo {}
unsafe impl std::marker::Sync for TargetInfo {}

impl Clone for TargetInfo {
    fn clone(&self) -> Self {
        let triple = TargetTriple::create(self.triple.as_str().to_str().unwrap());
        Self {
            target: Target::from_triple(&triple).unwrap(),
            triple,
            cpu: self.cpu.clone(),
            features: self.features.clone(),
        }
    }
}

impl TargetInfo {
    fn new(target: &revmc_backend::Target) -> Result<Cow<'static, Self>> {
        match target {
            revmc_backend::Target::Native => {
                static HOST_TARGET_INFO: OnceLock<TargetInfo> = OnceLock::new();
                Ok(Cow::Borrowed(HOST_TARGET_INFO.get_or_init(|| {
                    let triple = TargetMachine::get_default_triple();
                    let target = Target::from_triple(&triple).unwrap();
                    let cpu = TargetMachine::get_host_cpu_name().to_string_lossy().into_owned();
                    let features =
                        TargetMachine::get_host_cpu_features().to_string_lossy().into_owned();
                    Self { target, triple, cpu, features }
                })))
            }
            revmc_backend::Target::Triple { triple, cpu, features } => {
                let triple = TargetTriple::create(triple);
                let target = Target::from_triple(&triple).map_err(error_msg)?;
                let cpu = cpu.as_ref().cloned().unwrap_or_default();
                let features = features.as_ref().cloned().unwrap_or_default();
                Ok(Cow::Owned(Self { target, triple, cpu, features }))
            }
        }
    }
}

/// The LLVM-based EVM bytecode compiler function builder.
#[derive(Debug)]
#[must_use]
pub struct EvmLlvmBuilder<'a> {
    backend: &'a mut EvmLlvmBackend,
    function: FunctionValue<'static>,
    debug_scope: Option<DISubprogram<'static>>,
}

impl std::ops::Deref for EvmLlvmBuilder<'_> {
    type Target = EvmLlvmBackend;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.backend
    }
}

impl std::ops::DerefMut for EvmLlvmBuilder<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.backend
    }
}

impl EvmLlvmBuilder<'_> {
    #[allow(dead_code)]
    fn extract_value(
        &mut self,
        value: BasicValueEnum<'static>,
        index: u32,
        name: &str,
    ) -> BasicValueEnum<'static> {
        self.bcx.build_extract_value(value.into_struct_value(), index, self.name(name)).unwrap()
    }

    fn memcpy_inner(
        &mut self,
        dst: BasicValueEnum<'static>,
        src: BasicValueEnum<'static>,
        len: BasicValueEnum<'static>,
        inline: bool,
    ) {
        let dst = dst.into_pointer_value();
        let src = src.into_pointer_value();
        let len = len.into_int_value();
        let volatile = self.bool_const(false);
        let name = format!(
            "llvm.memcpy{}.p0.p0.{}",
            if inline { ".inline" } else { "" },
            fmt_ty(len.get_type().into()),
        );
        let memcpy = self.get_or_add_function(&name, |this| {
            this.ty_void.fn_type(
                &[this.ty_ptr.into(), this.ty_ptr.into(), this.ty_i64.into(), this.ty_i1.into()],
                false,
            )
        });
        self.bcx
            .build_call(memcpy, &[dst.into(), src.into(), len.into(), volatile.into()], "")
            .unwrap();
    }

    #[allow(dead_code)]
    fn call_overflow_function(
        &mut self,
        name: &str,
        lhs: BasicValueEnum<'static>,
        rhs: BasicValueEnum<'static>,
    ) -> (BasicValueEnum<'static>, BasicValueEnum<'static>) {
        let f = self.get_overflow_function(name, lhs.get_type());
        let result = self.call(f, &[lhs, rhs]).unwrap();
        (self.extract_value(result, 0, "result"), self.extract_value(result, 1, "overflow"))
    }

    #[allow(dead_code)]
    fn get_overflow_function(
        &mut self,
        name: &str,
        ty: BasicTypeEnum<'static>,
    ) -> FunctionValue<'static> {
        let name = format!("llvm.{name}.with.overflow.{}", fmt_ty(ty));
        self.get_or_add_function(&name, |this| {
            this.fn_type(
                Some(this.cx.struct_type(&[ty, this.ty_i1.into()], false).into()),
                &[ty, ty],
            )
        })
    }

    fn get_sat_function(
        &mut self,
        name: &str,
        ty: BasicTypeEnum<'static>,
    ) -> FunctionValue<'static> {
        let name = format!("llvm.{name}.sat.{}", fmt_ty(ty));
        self.get_or_add_function(&name, |this| this.fn_type(Some(ty), &[ty, ty]))
    }

    fn get_or_add_function(
        &mut self,
        name: &str,
        mk_ty: impl FnOnce(&mut Self) -> FunctionType<'static>,
    ) -> FunctionValue<'static> {
        match self.module.get_function(name) {
            Some(function) => function,
            None => {
                let ty = mk_ty(self);
                self.module.add_function(name, ty, None)
            }
        }
    }

    fn set_branch_weights(
        &self,
        inst: InstructionValue<'static>,
        weights: impl IntoIterator<Item = u32>,
    ) {
        let weights = weights.into_iter();
        let mut values = Vec::<BasicMetadataValueEnum<'static>>::with_capacity(
            1 + weights.size_hint().1.unwrap(),
        );
        values.push(self.cx.metadata_string("branch_weights").into());
        for weight in weights {
            values.push(self.ty_i32.const_int(weight as u64, false).into());
        }
        let metadata = self.cx.metadata_node(&values);
        let kind_id = self.cx.get_kind_id("prof");
        inst.set_metadata(metadata, kind_id).unwrap();
    }
}

impl BackendTypes for EvmLlvmBuilder<'_> {
    type Type = <EvmLlvmBackend as BackendTypes>::Type;
    type Value = <EvmLlvmBackend as BackendTypes>::Value;
    type StackSlot = <EvmLlvmBackend as BackendTypes>::StackSlot;
    type BasicBlock = <EvmLlvmBackend as BackendTypes>::BasicBlock;
    type Function = <EvmLlvmBackend as BackendTypes>::Function;
}

impl TypeMethods for EvmLlvmBuilder<'_> {
    fn type_ptr(&self) -> Self::Type {
        self.backend.type_ptr()
    }

    fn type_ptr_sized_int(&self) -> Self::Type {
        self.backend.type_ptr_sized_int()
    }

    fn type_int(&self, bits: u32) -> Self::Type {
        self.backend.type_int(bits)
    }

    fn type_array(&self, ty: Self::Type, size: u32) -> Self::Type {
        self.backend.type_array(ty, size)
    }

    fn type_bit_width(&self, ty: Self::Type) -> u32 {
        self.backend.type_bit_width(ty)
    }
}

impl Builder for EvmLlvmBuilder<'_> {
    fn create_block(&mut self, name: &str) -> Self::BasicBlock {
        self.cx.append_basic_block(self.function, self.name(name))
    }

    fn create_block_after(&mut self, after: Self::BasicBlock, name: &str) -> Self::BasicBlock {
        self.cx.insert_basic_block_after(after, self.name(name))
    }

    fn switch_to_block(&mut self, block: Self::BasicBlock) {
        self.bcx.position_at_end(block);
    }

    fn seal_block(&mut self, block: Self::BasicBlock) {
        let _ = block;
        // Nothing to do.
    }

    fn seal_all_blocks(&mut self) {
        // Nothing to do.
    }

    fn set_current_block_cold(&mut self) {
        let function = self.get_or_add_function("llvm.assume", |this| {
            this.ty_void.fn_type(&[this.ty_i1.into()], false)
        });
        let true_ = self.bool_const(true);
        let callsite = self.bcx.build_call(function, &[true_.into()], "cold").unwrap();
        let cold = self.cx.create_enum_attribute(Attribute::get_named_enum_kind_id("cold"), 0);
        callsite.add_attribute(AttributeLoc::Function, cold);
    }

    fn current_block(&mut self) -> Option<Self::BasicBlock> {
        self.bcx.get_insert_block()
    }

    fn block_addr(&mut self, block: Self::BasicBlock) -> Option<Self::Value> {
        unsafe { block.get_address().map(Into::into) }
    }

    fn add_comment_to_current_inst(&mut self, comment: &str) {
        let Some(block) = self.current_block() else { return };
        let Some(ins) = block.get_last_instruction() else { return };
        let metadata = self.cx.metadata_string(comment);
        let metadata = self.cx.metadata_node(&[metadata.into()]);
        ins.set_metadata(metadata, self.cx.get_kind_id("annotation")).unwrap();
    }

    fn set_debug_location(&mut self, line: u32, col: u32) {
        let Some(scope) = self.debug_scope else { return };
        let Some(di) = &self.di_state else { return };
        let loc = di.dibuilder.create_debug_location(
            self.cx,
            line,
            col,
            scope.as_debug_info_scope(),
            None,
        );
        self.bcx.set_current_debug_location(loc);
    }

    fn clear_debug_location(&mut self) {
        if self.debug_scope.is_some() {
            self.bcx.unset_current_debug_location();
        }
    }

    fn fn_param(&mut self, index: usize) -> Self::Value {
        self.function.get_nth_param(index as _).unwrap()
    }

    fn num_fn_params(&self) -> usize {
        self.function.count_params() as usize
    }

    fn bool_const(&mut self, value: bool) -> Self::Value {
        self.ty_i1.const_int(value as u64, false).into()
    }

    fn iconst(&mut self, ty: Self::Type, value: i64) -> Self::Value {
        ty.into_int_type().const_int(value as u64, value.is_negative()).into()
    }

    fn uconst(&mut self, ty: Self::Type, value: u64) -> Self::Value {
        ty.into_int_type().const_int(value, false).into()
    }

    fn iconst_256(&mut self, value: U256) -> Self::Value {
        if value == U256::ZERO {
            return self.ty_i256.const_zero().into();
        }

        self.ty_i256.const_int_from_string(&value.to_string(), StringRadix::Decimal).unwrap().into()
    }

    fn str_const(&mut self, value: &str) -> Self::Value {
        self.bcx.build_global_string_ptr(value, "").unwrap().as_pointer_value().into()
    }

    fn nullptr(&mut self) -> Self::Value {
        self.ty_ptr.const_null().into()
    }

    fn new_stack_slot_raw(&mut self, ty: Self::Type, name: &str) -> Self::StackSlot {
        // let ty = self.ty_i8.array_type(size);
        // let ptr = self.bcx.build_alloca(ty, self.name(name)).unwrap();
        // ptr.as_instruction().unwrap().set_alignment(align).unwrap();
        // ptr
        self.bcx.build_alloca(ty, self.name(name)).unwrap()
    }

    fn stack_load(&mut self, ty: Self::Type, slot: Self::StackSlot, name: &str) -> Self::Value {
        self.load(ty, slot.into(), name)
    }

    fn stack_store(&mut self, value: Self::Value, slot: Self::StackSlot) {
        self.store(value, slot.into())
    }

    fn stack_addr(&mut self, ty: Self::Type, slot: Self::StackSlot) -> Self::Value {
        let _ = ty;
        slot.into()
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, name: &str) -> Self::Value {
        let value = self.bcx.build_load(ty, ptr.into_pointer_value(), self.name(name)).unwrap();
        if ty == self.ty_i256.into() {
            self.current_block().unwrap().get_last_instruction().unwrap().set_alignment(8).unwrap();
        }
        value
    }

    fn load_aligned(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        align: usize,
        name: &str,
    ) -> Self::Value {
        let value = self.bcx.build_load(ty, ptr.into_pointer_value(), self.name(name)).unwrap();
        self.current_block()
            .unwrap()
            .get_last_instruction()
            .unwrap()
            .set_alignment(align as u32)
            .unwrap();
        value
    }

    fn store(&mut self, value: Self::Value, ptr: Self::Value) {
        let inst = self.bcx.build_store(ptr.into_pointer_value(), value).unwrap();
        if value.get_type() == self.ty_i256.into() {
            inst.set_alignment(8).unwrap();
        }
    }

    fn store_aligned(&mut self, value: Self::Value, ptr: Self::Value, align: usize) {
        let inst = self.bcx.build_store(ptr.into_pointer_value(), value).unwrap();
        inst.set_alignment(align as u32).unwrap();
    }

    fn nop(&mut self) {
        // LLVM doesn't have a NOP instruction.
    }

    fn ret(&mut self, values: &[Self::Value]) {
        match values {
            [] => self.bcx.build_return(None),
            [value] => self.bcx.build_return(Some(value)),
            values => self.bcx.build_aggregate_return(values),
        }
        .unwrap();
    }

    fn icmp(&mut self, cond: IntCC, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx
            .build_int_compare(convert_intcc(cond), lhs.into_int_value(), rhs.into_int_value(), "")
            .unwrap()
            .into()
    }

    fn icmp_imm(&mut self, cond: IntCC, lhs: Self::Value, rhs: i64) -> Self::Value {
        let rhs = self.iconst(lhs.get_type(), rhs);
        self.icmp(cond, lhs, rhs)
    }

    fn is_null(&mut self, ptr: Self::Value) -> Self::Value {
        self.bcx.build_is_null(ptr.into_pointer_value(), "").unwrap().into()
    }

    fn is_not_null(&mut self, ptr: Self::Value) -> Self::Value {
        self.bcx.build_is_not_null(ptr.into_pointer_value(), "").unwrap().into()
    }

    fn br(&mut self, dest: Self::BasicBlock) {
        self.bcx.build_unconditional_branch(dest).unwrap();
    }

    fn brif(
        &mut self,
        cond: Self::Value,
        then_block: Self::BasicBlock,
        else_block: Self::BasicBlock,
    ) {
        self.bcx.build_conditional_branch(cond.into_int_value(), then_block, else_block).unwrap();
    }

    fn brif_cold(
        &mut self,
        cond: Self::Value,
        then_block: Self::BasicBlock,
        else_block: Self::BasicBlock,
        then_is_cold: bool,
    ) {
        let inst = self
            .bcx
            .build_conditional_branch(cond.into_int_value(), then_block, else_block)
            .unwrap();
        let weights = if then_is_cold { [1, DEFAULT_WEIGHT] } else { [DEFAULT_WEIGHT, 1] };
        self.set_branch_weights(inst, weights);
    }

    fn switch(
        &mut self,
        index: Self::Value,
        default: Self::BasicBlock,
        targets: &[(u64, Self::BasicBlock)],
        default_is_cold: bool,
    ) {
        let ty = index.get_type().into_int_type();
        let targets =
            targets.iter().map(|(v, b)| (ty.const_int(*v, false), *b)).collect::<Vec<_>>();
        let inst = self.bcx.build_switch(index.into_int_value(), default, &targets).unwrap();
        if default_is_cold {
            let weights = iter::once(1).chain(iter::repeat_n(DEFAULT_WEIGHT, targets.len()));
            self.set_branch_weights(inst, weights);
        }
    }

    fn br_indirect(&mut self, address: Self::Value, destinations: &[Self::BasicBlock]) {
        let _ = self.bcx.build_indirect_branch(address, destinations).unwrap();
    }

    fn phi(&mut self, ty: Self::Type, incoming: &[(Self::Value, Self::BasicBlock)]) -> Self::Value {
        let incoming = incoming
            .iter()
            .map(|(value, block)| (value as &dyn BasicValue<'_>, *block))
            .collect::<Vec<_>>();
        let phi = self.bcx.build_phi(ty, "").unwrap();
        phi.add_incoming(&incoming);
        phi.as_basic_value()
    }

    fn select(
        &mut self,
        cond: Self::Value,
        then_value: Self::Value,
        else_value: Self::Value,
    ) -> Self::Value {
        self.bcx.build_select(cond.into_int_value(), then_value, else_value, "").unwrap()
    }

    fn lazy_select(
        &mut self,
        cond: Self::Value,
        ty: Self::Type,
        then_value: impl FnOnce(&mut Self) -> Self::Value,
        else_value: impl FnOnce(&mut Self) -> Self::Value,
    ) -> Self::Value {
        let then_block = if let Some(current) = self.current_block() {
            self.create_block_after(current, "then")
        } else {
            self.create_block("then")
        };
        let else_block = self.create_block_after(then_block, "else");
        let done_block = self.create_block_after(else_block, "contd");

        self.brif(cond, then_block, else_block);

        self.switch_to_block(then_block);
        let then_value = then_value(self);
        self.br(done_block);

        self.switch_to_block(else_block);
        let else_value = else_value(self);
        self.br(done_block);

        self.switch_to_block(done_block);
        let phi = self.bcx.build_phi(ty, "").unwrap();
        phi.add_incoming(&[(&then_value, then_block), (&else_value, else_block)]);
        phi.as_basic_value()
    }

    fn iadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.build_int_add(lhs.into_int_value(), rhs.into_int_value(), "").unwrap().into()
    }

    fn isub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "").unwrap().into()
    }

    fn imul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "").unwrap().into()
    }

    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx
            .build_int_unsigned_div(lhs.into_int_value(), rhs.into_int_value(), "")
            .unwrap()
            .into()
    }

    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx
            .build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "")
            .unwrap()
            .into()
    }

    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx
            .build_int_unsigned_rem(lhs.into_int_value(), rhs.into_int_value(), "")
            .unwrap()
            .into()
    }

    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx
            .build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "")
            .unwrap()
            .into()
    }

    fn iadd_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        let rhs = self.iconst(lhs.get_type(), rhs);
        self.iadd(lhs, rhs)
    }

    fn isub_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        let rhs = self.iconst(lhs.get_type(), rhs);
        self.isub(lhs, rhs)
    }

    fn imul_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        let rhs = self.iconst(lhs.get_type(), rhs);
        self.imul(lhs, rhs)
    }

    // - [Avoid using arithmetic intrinsics](https://llvm.org/docs/Frontend/PerformanceTips.html)
    // - [Don't use usub.with.overflow intrinsic](https://github.com/rust-lang/rust/pull/103299)
    // - [for unsigned add overflow the recommended pattern is x + y < x](https://github.com/rust-lang/rust/pull/124114#issuecomment-2066173305)
    fn uadd_overflow(&mut self, lhs: Self::Value, rhs: Self::Value) -> (Self::Value, Self::Value) {
        let result = self.iadd(lhs, rhs);
        let overflow = self.icmp(IntCC::UnsignedLessThan, result, rhs);
        (result, overflow)
    }

    fn usub_overflow(&mut self, lhs: Self::Value, rhs: Self::Value) -> (Self::Value, Self::Value) {
        let result = self.isub(lhs, rhs);
        let overflow = self.icmp(IntCC::UnsignedLessThan, lhs, rhs);
        (result, overflow)
    }

    fn uadd_sat(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let f = self.get_sat_function("uadd", lhs.get_type());
        self.call(f, &[lhs, rhs]).unwrap()
    }

    fn umax(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let ty = lhs.get_type();
        let name = format!("llvm.umin.{}", fmt_ty(ty));
        let max = self.get_or_add_function(&name, |this| this.fn_type(Some(ty), &[ty, ty]));
        self.call(max, &[lhs, rhs]).unwrap()
    }

    fn umin(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let ty = lhs.get_type();
        let name = format!("llvm.umin.{}", fmt_ty(ty));
        let max = self.get_or_add_function(&name, |this| this.fn_type(Some(ty), &[ty, ty]));
        self.call(max, &[lhs, rhs]).unwrap()
    }

    fn bswap(&mut self, value: Self::Value) -> Self::Value {
        let ty = value.get_type();
        let name = format!("llvm.bswap.{}", fmt_ty(ty));
        let bswap = self.get_or_add_function(&name, |this| this.fn_type(Some(ty), &[ty]));
        self.call(bswap, &[value]).unwrap()
    }

    fn bitor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.build_or(lhs.into_int_value(), rhs.into_int_value(), "").unwrap().into()
    }

    fn bitand(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.build_and(lhs.into_int_value(), rhs.into_int_value(), "").unwrap().into()
    }

    fn bitxor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.build_xor(lhs.into_int_value(), rhs.into_int_value(), "").unwrap().into()
    }

    fn bitnot(&mut self, value: Self::Value) -> Self::Value {
        self.bcx.build_not(value.into_int_value(), "").unwrap().into()
    }

    fn clz(&mut self, value: Self::Value) -> Self::Value {
        let ty = value.get_type();
        let i1_ty = self.type_int(1);
        let name = format!("llvm.ctlz.{}", fmt_ty(ty));
        let ctlz = self.get_or_add_function(&name, |this| this.fn_type(Some(ty), &[ty, i1_ty]));
        let is_poison_on_zero = self.bool_const(false);
        self.call(ctlz, &[value, is_poison_on_zero]).unwrap()
    }

    fn bitor_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        let rhs = self.iconst(lhs.get_type(), rhs);
        self.bitor(lhs, rhs)
    }

    fn bitand_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        let rhs = self.iconst(lhs.get_type(), rhs);
        self.bitand(lhs, rhs)
    }

    fn bitxor_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        let rhs = self.iconst(lhs.get_type(), rhs);
        self.bitxor(lhs, rhs)
    }

    fn ishl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.build_left_shift(lhs.into_int_value(), rhs.into_int_value(), "").unwrap().into()
    }

    fn ushr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx
            .build_right_shift(lhs.into_int_value(), rhs.into_int_value(), false, "")
            .unwrap()
            .into()
    }

    fn sshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx
            .build_right_shift(lhs.into_int_value(), rhs.into_int_value(), true, "")
            .unwrap()
            .into()
    }

    fn zext(&mut self, ty: Self::Type, value: Self::Value) -> Self::Value {
        self.bcx.build_int_z_extend(value.into_int_value(), ty.into_int_type(), "").unwrap().into()
    }

    fn sext(&mut self, ty: Self::Type, value: Self::Value) -> Self::Value {
        self.bcx.build_int_s_extend(value.into_int_value(), ty.into_int_type(), "").unwrap().into()
    }

    fn ireduce(&mut self, to: Self::Type, value: Self::Value) -> Self::Value {
        self.bcx.build_int_truncate(value.into_int_value(), to.into_int_type(), "").unwrap().into()
    }

    fn inttoptr(&mut self, value: Self::Value, ty: Self::Type) -> Self::Value {
        self.bcx
            .build_int_to_ptr(value.into_int_value(), ty.into_pointer_type(), "")
            .unwrap()
            .into()
    }

    fn gep(
        &mut self,
        elem_ty: Self::Type,
        ptr: Self::Value,
        indexes: &[Self::Value],
        name: &str,
    ) -> Self::Value {
        let indexes = indexes.iter().map(|idx| idx.into_int_value()).collect::<Vec<_>>();
        unsafe {
            self.bcx.build_in_bounds_gep(
                elem_ty,
                ptr.into_pointer_value(),
                &indexes,
                self.name(name),
            )
        }
        .unwrap()
        .into()
    }

    fn tail_call(
        &mut self,
        function: Self::Function,
        args: &[Self::Value],
        tail_call: TailCallKind,
    ) -> Option<Self::Value> {
        let args = args.iter().copied().map(Into::into).collect::<Vec<_>>();
        let callsite = self.bcx.build_call(function, &args, "").unwrap();
        if tail_call != TailCallKind::None {
            callsite.set_tail_call_kind(convert_tail_call_kind(tail_call));
        }
        callsite.try_as_basic_value().basic()
    }

    fn is_compile_time_known(&mut self, value: Self::Value) -> Option<Self::Value> {
        let ty = value.get_type();
        let name = format!("llvm.is.constant.{}", fmt_ty(ty));
        let f =
            self.get_or_add_function(&name, |this| this.fn_type(Some(this.ty_i1.into()), &[ty]));
        Some(self.call(f, &[value]).unwrap())
    }

    fn memcpy(&mut self, dst: Self::Value, src: Self::Value, len: Self::Value) {
        self.memcpy_inner(dst, src, len, false);
    }

    fn memcpy_inline(&mut self, dst: Self::Value, src: Self::Value, len: i64) {
        let len = self.iconst(self.ty_i64.into(), len);
        self.memcpy_inner(dst, src, len, true);
    }

    fn unreachable(&mut self) {
        self.bcx.build_unreachable().unwrap();
    }

    fn get_or_build_function(
        &mut self,
        name: &str,
        params: &[Self::Type],
        ret: Option<Self::Type>,
        linkage: revmc_backend::Linkage,
        build: impl FnOnce(&mut Self),
    ) -> Self::Function {
        if let Some(function) = self.module.get_function(name) {
            return function;
        }

        let before = self.current_block();

        let func_ty = self.fn_type(ret, params);
        let function = self.module.add_function(name, func_ty, Some(convert_linkage(linkage)));
        let prev_function = std::mem::replace(&mut self.function, function);

        let entry = self.cx.append_basic_block(function, self.name("entry"));
        self.bcx.position_at_end(entry);
        build(self);
        if let Some(before) = before {
            self.bcx.position_at_end(before);
        }

        self.function = prev_function;

        function
    }

    fn get_function(&mut self, name: &str) -> Option<Self::Function> {
        self.module.get_function(name)
    }

    fn get_printf_function(&mut self) -> Self::Function {
        let name = "printf";
        if let Some(function) = self.module.get_function(name) {
            return function;
        }

        let ty = self.cx.void_type().fn_type(&[self.ty_ptr.into()], true);
        self.module.add_function(name, ty, Some(inkwell::module::Linkage::External))
    }

    fn add_function(
        &mut self,
        name: &str,
        params: &[Self::Type],
        ret: Option<Self::Type>,
        address: Option<usize>,
        linkage: revmc_backend::Linkage,
    ) -> Self::Function {
        let func_ty = self.fn_type(ret, params);
        let function = self.module.add_function(name, func_ty, Some(convert_linkage(linkage)));
        if let Some(address) = address
            && let Some(exec_engine) = &self.exec_engine
            && !self.mapped_symbols.contains(name)
        {
            exec_engine.add_global_mapping(&function, address);
            self.mapped_symbols.insert(name.to_string());
        }
        function
    }

    fn add_function_attribute(
        &mut self,
        function: Option<Self::Function>,
        attribute: revmc_backend::Attribute,
        loc: revmc_backend::FunctionAttributeLocation,
    ) {
        let loc = convert_attribute_loc(loc);
        let attr = convert_attribute(self, attribute);
        function.unwrap_or(self.function).add_attribute(loc, attr);
    }
}

/// Builds the LLVM pass pipeline string. See [`EvmLlvmBackend::optimize_module`].
fn build_pass_pipeline(with_licm: bool) -> String {
    let mut passes = String::from("function(");
    let function_passes: &[&str] = &[
        "simplifycfg",
        "sroa",
        "early-cse",
        "jump-threading",
        "correlated-propagation",
        "simplifycfg",
        "instcombine<no-verify-fixpoint>",
    ];
    let licm_passes: &[&str] =
        &["loop-mssa(licm,loop-rotate,licm)", "simplifycfg", "instcombine<no-verify-fixpoint>"];
    let post_passes: &[&str] = &[
        "sroa",
        "early-cse",
        "sccp",
        "instcombine<no-verify-fixpoint>",
        "adce",
        "dse",
        "simplifycfg",
    ];

    let iter =
        function_passes.iter().chain(if with_licm { licm_passes } else { &[] }).chain(post_passes);
    for (i, pass) in iter.enumerate() {
        if i > 0 {
            passes.push(',');
        }
        passes.push_str(pass);
    }
    passes.push_str("),globaldce");
    passes
}

fn init() -> Result<()> {
    let mut init_result = Ok(());
    static INIT: Once = Once::new();
    INIT.call_once(|| init_result = init_());
    init_result
}

fn init_() -> Result<()> {
    // TODO: This also reports "PLEASE submit a bug report to..." when the segfault is
    // outside of LLVM.
    // enable_llvm_pretty_stack_trace();

    extern "C" fn report_fatal_error(msg: *const std::ffi::c_char) {
        let msg_cstr = unsafe { std::ffi::CStr::from_ptr(msg) };
        let msg = msg_cstr.to_string_lossy();
        error!(target: "llvm", "LLVM fatal error: {msg}");
    }

    unsafe {
        install_fatal_error_handler(report_fatal_error);
    }

    // The first arg is only used in `-help` output AFAICT.
    let args = [c"revmc-llvm".as_ptr(), c"-x86-asm-syntax=intel".as_ptr()];
    unsafe {
        inkwell::llvm_sys::support::LLVMParseCommandLineOptions(
            args.len() as i32,
            args.as_ptr(),
            std::ptr::null(),
        )
    }

    let config = InitializationConfig {
        asm_parser: false,
        asm_printer: true,
        base: true,
        disassembler: true,
        info: true,
        machine_code: true,
    };
    Target::initialize_all(&config);

    // Ensure MCJIT is linked in. Without this, LTO may strip the MCJIT
    // registration code, causing `create_jit_execution_engine` to fail with
    // "JIT has not been linked in" followed by a SIGSEGV in destructors.
    // See: https://github.com/TheDan64/inkwell/issues/320
    inkwell::execution_engine::ExecutionEngine::link_in_mc_jit();

    Ok(())
}

fn get_context() -> &'static Context {
    thread_local! {
        static TLS_LLVM_CONTEXT: Context = Context::create();
    }
    // SAFETY: It can't be shared across threads anyway.
    TLS_LLVM_CONTEXT.with(|cx| unsafe { core::mem::transmute(cx) })
}

fn create_module<'ctx>(
    cx: &'ctx Context,
    machine: &TargetMachine,
    aot: bool,
) -> Result<Module<'ctx>> {
    let module_name = "evm";
    let module = cx.create_module(module_name);
    module.set_source_file_name(module_name);
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.set_triple(&machine.get_triple());
    if aot {
        module.add_basic_value_flag(
            "PIC Level",
            FlagBehavior::Error, // TODO: Min
            cx.i32_type().const_int(2, false),
        );
        module.add_basic_value_flag(
            "RtLibUseGOT",
            FlagBehavior::Warning,
            cx.i32_type().const_int(1, false),
        );
    }
    Ok(module)
}

fn convert_intcc(cond: IntCC) -> IntPredicate {
    match cond {
        IntCC::Equal => IntPredicate::EQ,
        IntCC::NotEqual => IntPredicate::NE,
        IntCC::SignedLessThan => IntPredicate::SLT,
        IntCC::SignedGreaterThanOrEqual => IntPredicate::SGE,
        IntCC::SignedGreaterThan => IntPredicate::SGT,
        IntCC::SignedLessThanOrEqual => IntPredicate::SLE,
        IntCC::UnsignedLessThan => IntPredicate::ULT,
        IntCC::UnsignedGreaterThanOrEqual => IntPredicate::UGE,
        IntCC::UnsignedGreaterThan => IntPredicate::UGT,
        IntCC::UnsignedLessThanOrEqual => IntPredicate::ULE,
    }
}

fn convert_opt_level(level: revmc_backend::OptimizationLevel) -> OptimizationLevel {
    match level {
        revmc_backend::OptimizationLevel::None => OptimizationLevel::None,
        revmc_backend::OptimizationLevel::Less => OptimizationLevel::Less,
        revmc_backend::OptimizationLevel::Default => OptimizationLevel::Default,
        revmc_backend::OptimizationLevel::Aggressive => OptimizationLevel::Aggressive,
    }
}

fn convert_opt_level_rev(level: OptimizationLevel) -> revmc_backend::OptimizationLevel {
    match level {
        OptimizationLevel::None => revmc_backend::OptimizationLevel::None,
        OptimizationLevel::Less => revmc_backend::OptimizationLevel::Less,
        OptimizationLevel::Default => revmc_backend::OptimizationLevel::Default,
        OptimizationLevel::Aggressive => revmc_backend::OptimizationLevel::Aggressive,
    }
}

fn convert_attribute(bcx: &EvmLlvmBuilder<'_>, attr: revmc_backend::Attribute) -> Attribute {
    use revmc_backend::Attribute as OurAttr;

    enum AttrValue<'a> {
        String(&'a str),
        Enum(u64),
        Type(AnyTypeEnum<'a>),
    }

    let cpu;
    let (key, value) = match attr {
        OurAttr::WillReturn => ("willreturn", AttrValue::Enum(0)),
        OurAttr::NoReturn => ("noreturn", AttrValue::Enum(0)),
        OurAttr::NoFree => ("nofree", AttrValue::Enum(0)),
        OurAttr::NoRecurse => ("norecurse", AttrValue::Enum(0)),
        OurAttr::NoSync => ("nosync", AttrValue::Enum(0)),
        OurAttr::NoUnwind => ("nounwind", AttrValue::Enum(0)),
        OurAttr::AllFramePointers => ("frame-pointer", AttrValue::String("all")),
        OurAttr::NativeTargetCpu => (
            "target-cpu",
            AttrValue::String({
                cpu = bcx.machine.get_cpu();
                cpu.to_str().unwrap()
            }),
        ),
        OurAttr::Cold => ("cold", AttrValue::Enum(0)),
        OurAttr::Hot => ("hot", AttrValue::Enum(0)),
        OurAttr::HintInline => ("inlinehint", AttrValue::Enum(0)),
        OurAttr::AlwaysInline => ("alwaysinline", AttrValue::Enum(0)),
        OurAttr::NoInline => ("noinline", AttrValue::Enum(0)),
        OurAttr::Speculatable => ("speculatable", AttrValue::Enum(0)),

        OurAttr::NoAlias => ("noalias", AttrValue::Enum(0)),
        OurAttr::NoCapture => ("captures", AttrValue::Enum(0)), // captures(none) - no capture
        OurAttr::NoUndef => ("noundef", AttrValue::Enum(0)),
        OurAttr::Align(n) => ("align", AttrValue::Enum(n)),
        OurAttr::NonNull => ("nonnull", AttrValue::Enum(0)),
        OurAttr::Dereferenceable(n) => ("dereferenceable", AttrValue::Enum(n)),
        OurAttr::SRet(n) => {
            ("sret", AttrValue::Type(bcx.type_array(bcx.ty_i8.into(), n as _).as_any_type_enum()))
        }
        OurAttr::ReadNone => ("readnone", AttrValue::Enum(0)),
        OurAttr::ReadOnly => ("readonly", AttrValue::Enum(0)),
        OurAttr::WriteOnly => ("writeonly", AttrValue::Enum(0)),
        OurAttr::Writable => ("writable", AttrValue::Enum(0)),
        // memory(argmem: readwrite) = ModRef(3) << ArgMem(0) = 3.
        OurAttr::ArgMemOnly => ("memory", AttrValue::Enum(3)),

        attr => unimplemented!("llvm attribute conversion: {attr:?}"),
    };
    match value {
        AttrValue::String(value) => bcx.cx.create_string_attribute(key, value),
        AttrValue::Enum(value) => {
            let id = Attribute::get_named_enum_kind_id(key);
            bcx.cx.create_enum_attribute(id, value)
        }
        AttrValue::Type(ty) => {
            let id = Attribute::get_named_enum_kind_id(key);
            bcx.cx.create_type_attribute(id, ty)
        }
    }
}

fn convert_attribute_loc(loc: revmc_backend::FunctionAttributeLocation) -> AttributeLoc {
    match loc {
        revmc_backend::FunctionAttributeLocation::Return => AttributeLoc::Return,
        revmc_backend::FunctionAttributeLocation::Param(i) => AttributeLoc::Param(i),
        revmc_backend::FunctionAttributeLocation::Function => AttributeLoc::Function,
    }
}

fn convert_linkage(linkage: revmc_backend::Linkage) -> inkwell::module::Linkage {
    match linkage {
        revmc_backend::Linkage::Public => inkwell::module::Linkage::External,
        revmc_backend::Linkage::Import => inkwell::module::Linkage::External,
        revmc_backend::Linkage::Private => inkwell::module::Linkage::Private,
    }
}

fn convert_tail_call_kind(kind: TailCallKind) -> inkwell::llvm_sys::LLVMTailCallKind {
    match kind {
        TailCallKind::None => inkwell::llvm_sys::LLVMTailCallKind::LLVMTailCallKindNone,
        TailCallKind::Tail => inkwell::llvm_sys::LLVMTailCallKind::LLVMTailCallKindTail,
        TailCallKind::MustTail => inkwell::llvm_sys::LLVMTailCallKind::LLVMTailCallKindMustTail,
        TailCallKind::NoTail => inkwell::llvm_sys::LLVMTailCallKind::LLVMTailCallKindNoTail,
    }
}

// No `#[track_caller]` because `map_err` doesn't propagate it.
fn error_msg(msg: inkwell::support::LLVMString) -> revmc_backend::Error {
    revmc_backend::Error::msg(msg.to_string_lossy().trim_end().to_string())
}

fn fmt_ty(ty: BasicTypeEnum<'_>) -> impl std::fmt::Display {
    ty.print_to_string().to_str().unwrap().trim_matches('"').to_string()
}

#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

use melior::{
    dialect::{llvm::r#type::pointer, DialectRegistry},
    ir::{
        attribute::StringAttribute,
        operation::{OperationBuilder, OperationPrintingFlags},
        r#type::IntegerType,
        BlockRef, Identifier, Location, Module, Region, RegionRef, Type, Value,
    },
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
    Context,
};
use revm_jit_backend::{debug_time, Backend, BackendTypes, Result, TypeMethods};
use revm_jit_llvm as llvm;
use std::path::Path;

pub fn new_context() -> Context {
    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();
    register_all_passes();
    register_all_llvm_translations(&context);
    context
}

/// The MLIR-based EVM bytecode compiler backend.
#[derive(Debug)]
#[must_use]
pub struct EvmMlirBackend<'ctx> {
    #[allow(dead_code)]
    cx: &'ctx Context,
    module: Module<'ctx>,

    ty_ptr: Type<'ctx>,
    ty_isize: Type<'ctx>,

    aot: bool,
    opt_level: llvm::inkwell::OptimizationLevel,
}

impl<'ctx> EvmMlirBackend<'ctx> {
    /// Creates a new MLIR backend.
    pub fn new(
        cx: &'ctx Context,
        aot: bool,
        opt_level: revm_jit_backend::OptimizationLevel,
    ) -> Result<Self> {
        debug_time!("new MLIR backend", || Self::new_inner(cx, aot, opt_level))
    }

    fn new_inner(
        cx: &'ctx Context,
        aot: bool,
        opt_level: revm_jit_backend::OptimizationLevel,
    ) -> Result<Self> {
        llvm::init()?;

        let opt_level = llvm::convert_opt_level(opt_level);

        let target_defaults = llvm::TargetDefaults::get();
        let target_triple = target_defaults.triple.as_str().to_str().unwrap();

        let target_machine = target_defaults.create_target_machine(aot, opt_level)?;
        let target_data = target_machine.get_target_data();
        let data_layout = target_data.get_data_layout();
        let data_layout_str = data_layout.as_str().to_str().unwrap();
        let ptr_size = target_data.get_pointer_byte_size(None);

        let op = OperationBuilder::new("evm", Location::unknown(cx))
            .add_attributes(&[
                (
                    Identifier::new(cx, "llvm.target_triple"),
                    StringAttribute::new(cx, target_triple).into(),
                ),
                (
                    Identifier::new(cx, "llvm.data_layout"),
                    StringAttribute::new(cx, data_layout_str).into(),
                ),
            ])
            .build()?;
        assert!(op.verify());

        let module = Module::from_operation(op).unwrap();

        let ty_ptr = pointer(cx, 0);
        let ty_isize = IntegerType::new(cx, ptr_size * 8).into();

        Ok(Self { cx, module, ty_ptr, ty_isize, aot, opt_level })
    }
}

impl<'ctx> BackendTypes for EvmMlirBackend<'ctx> {
    type Type = Type<'ctx>;
    type Value = Value<'ctx, 'ctx>;
    type StackSlot = Value<'ctx, 'ctx>;
    type BasicBlock = BlockRef<'ctx, 'ctx>;
    type Function = RegionRef<'ctx, 'ctx>;
}

impl<'ctx> TypeMethods for EvmMlirBackend<'ctx> {
    fn type_ptr(&self) -> Self::Type {
        self.ty_ptr
    }

    fn type_ptr_sized_int(&self) -> Self::Type {
        self.ty_isize
    }

    fn type_int(&self, bits: u32) -> Self::Type {
        IntegerType::new(&self.cx, bits).into()
    }

    fn type_array(&self, ty: Self::Type, size: u32) -> Self::Type {
        Type::vector(&[size as u64], ty)
    }

    fn type_bit_width(&self, ty: Self::Type) -> u32 {
        IntegerType::try_from(ty).unwrap().width()
    }
}

impl<'ctx> Backend for EvmMlirBackend<'ctx> {
    type Builder<'a> = EvmMlirBuilder<'a, 'ctx> where Self: 'a;
    type FuncId = u32;

    fn ir_extension(&self) -> &'static str {
        "mlir"
    }

    fn set_module_name(&mut self, name: &str) {
        let _ = name;
        // self.module.as_operation_mut().name()
        // self.module.set_name(name);
    }

    fn set_is_dumping(&mut self, yes: bool) {
        let _ = yes;
        // self.machine.set_asm_verbosity(yes);
    }

    fn set_debug_assertions(&mut self, yes: bool) {
        let _ = yes;
        // self.debug_assertions = yes;
    }

    fn opt_level(&self) -> revm_jit_backend::OptimizationLevel {
        llvm::convert_opt_level_rev(self.opt_level)
    }

    fn set_opt_level(&mut self, level: revm_jit_backend::OptimizationLevel) {
        self.opt_level = llvm::convert_opt_level(level);
    }

    fn is_aot(&self) -> bool {
        self.aot
    }

    fn dump_ir(&mut self, path: &Path) -> Result<()> {
        let flags = OperationPrintingFlags::new();
        let s = self.module.as_operation().to_string_with_flags(flags)?;
        std::fs::write(path, s)?;
        Ok(())
    }

    fn dump_disasm(&mut self, path: &Path) -> Result<()> {
        // TODO
        let _ = path;
        Ok(())
    }
}

/// The MLIR-based EVM bytecode compiler function builder.
#[derive(Debug)]
#[must_use]
pub struct EvmMlirBuilder<'a, 'ctx> {
    backend: &'a mut EvmMlirBackend<'ctx>,
    function: Region<'ctx>,
}

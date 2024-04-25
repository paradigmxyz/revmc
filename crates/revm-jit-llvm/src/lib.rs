#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

#[macro_use]
extern crate tracing;

#[macro_use]
extern crate revm_jit_backend;

use inkwell::{
    attributes::{Attribute, AttributeLoc},
    basic_block::BasicBlock,
    execution_engine::ExecutionEngine,
    module::Module,
    passes::PassBuilderOptions,
    support::error_handling::install_fatal_error_handler,
    targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
    },
    types::{BasicType, BasicTypeEnum, FunctionType, IntType, PointerType, StringRadix, VoidType},
    values::{BasicValue, BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};
use revm_jit_backend::{
    eyre, Backend, BackendTypes, Builder, Error, IntCC, Result, TypeMethods, U256,
};
use rustc_hash::FxHashMap;
use std::{
    path::Path,
    sync::{Once, OnceLock},
};

pub use inkwell::{self, context::Context};

pub mod orc;

/// Executes the given closure with a thread-local LLVM context.
#[inline]
pub fn with_llvm_context<R>(f: impl FnOnce(&Context) -> R) -> R {
    thread_local! {
        static TLS_LLVM_CONTEXT: Context = Context::create();
    }
    TLS_LLVM_CONTEXT.with(f)
}

/// The LLVM-based EVM bytecode compiler backend.
#[derive(Debug)]
#[must_use]
pub struct EvmLlvmBackend<'ctx> {
    cx: &'ctx Context,
    bcx: inkwell::builder::Builder<'ctx>,
    module: Module<'ctx>,
    exec_engine: Option<ExecutionEngine<'ctx>>,
    machine: TargetMachine,

    ty_void: VoidType<'ctx>,
    ty_ptr: PointerType<'ctx>,
    ty_i1: IntType<'ctx>,
    ty_i8: IntType<'ctx>,
    ty_i32: IntType<'ctx>,
    ty_i64: IntType<'ctx>,
    ty_i256: IntType<'ctx>,
    ty_isize: IntType<'ctx>,

    aot: bool,
    debug_assertions: bool,
    opt_level: OptimizationLevel,
    /// Separate from `function_names` to have always increasing IDs.
    function_counter: u32,
    function_names: FxHashMap<u32, String>,
}

impl<'ctx> EvmLlvmBackend<'ctx> {
    /// Creates a new LLVM backend.
    #[inline]
    pub fn new(
        cx: &'ctx Context,
        aot: bool,
        opt_level: revm_jit_backend::OptimizationLevel,
    ) -> Result<Self> {
        debug_time!("new LLVM backend", || Self::new_inner(cx, aot, opt_level))
    }

    fn new_inner(
        cx: &'ctx Context,
        aot: bool,
        opt_level: revm_jit_backend::OptimizationLevel,
    ) -> Result<Self> {
        init()?;

        let opt_level = convert_opt_level(opt_level);

        let target_defaults = TargetDefaults::get();
        let target = Target::from_triple(&target_defaults.triple).map_err(error_msg)?;
        let machine = target
            .create_target_machine(
                &target_defaults.triple,
                &target_defaults.cpu,
                &target_defaults.features,
                opt_level,
                if aot { RelocMode::DynamicNoPic } else { RelocMode::PIC },
                if aot { CodeModel::Default } else { CodeModel::JITDefault },
            )
            .ok_or_else(|| eyre::eyre!("failed to create target machine"))?;

        let module = create_module(cx, &machine)?;

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
        let ty_i256 = cx.custom_width_int_type(256);
        let ty_isize = cx.ptr_sized_int_type(&machine.get_target_data(), None);
        let ty_ptr = ty_i8.ptr_type(AddressSpace::default());
        Ok(Self {
            cx,
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
            opt_level,
            function_counter: 0,
            function_names: FxHashMap::default(),
        })
    }

    /// Returns the LLVM context.
    #[inline]
    pub fn cx(&self) -> &'ctx Context {
        self.cx
    }

    fn exec_engine(&self) -> &ExecutionEngine<'ctx> {
        assert!(!self.aot, "requested JIT execution engine on AOT");
        self.exec_engine.as_ref().expect("missing JIT execution engine")
    }

    fn fn_type(
        &self,
        ret: Option<BasicTypeEnum<'ctx>>,
        params: &[BasicTypeEnum<'ctx>],
    ) -> FunctionType<'ctx> {
        let params = params.iter().copied().map(Into::into).collect::<Vec<_>>();
        match ret {
            Some(ret) => ret.fn_type(&params, false),
            None => self.ty_void.fn_type(&params, false),
        }
    }

    fn id_to_name(&self, id: u32) -> &str {
        &self.function_names[&id]
    }
}

impl<'ctx> BackendTypes for EvmLlvmBackend<'ctx> {
    type Type = BasicTypeEnum<'ctx>;
    type Value = BasicValueEnum<'ctx>;
    type StackSlot = PointerValue<'ctx>;
    type BasicBlock = BasicBlock<'ctx>;
    type Function = FunctionValue<'ctx>;
}

impl<'ctx> TypeMethods for EvmLlvmBackend<'ctx> {
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
            bits => self.cx.custom_width_int_type(bits),
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

impl<'ctx> Backend for EvmLlvmBackend<'ctx> {
    type Builder<'a> = EvmLlvmBuilder<'a, 'ctx> where Self: 'a;
    type FuncId = u32;

    fn ir_extension(&self) -> &'static str {
        "ll"
    }

    fn set_module_name(&mut self, name: &str) {
        self.module.set_name(name);
    }

    fn set_is_dumping(&mut self, yes: bool) {
        self.machine.set_asm_verbosity(yes);
    }

    fn set_debug_assertions(&mut self, yes: bool) {
        self.debug_assertions = yes;
    }

    fn opt_level(&self) -> revm_jit_backend::OptimizationLevel {
        convert_opt_level_rev(self.opt_level)
    }

    fn set_opt_level(&mut self, level: revm_jit_backend::OptimizationLevel) {
        self.opt_level = convert_opt_level(level);
    }

    fn is_aot(&self) -> bool {
        self.aot
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
        linkage: revm_jit_backend::Linkage,
    ) -> Result<(Self::Builder<'_>, Self::FuncId)> {
        let fn_type = self.fn_type(ret, params);
        let function = self.module.add_function(name, fn_type, Some(convert_linkage(linkage)));
        for (i, &name) in param_names.iter().enumerate() {
            function.get_nth_param(i as u32).expect(name).set_name(name);
        }

        let entry = self.cx.append_basic_block(function, "entry");
        self.bcx.position_at_end(entry);

        let id = self.function_counter;
        self.function_counter += 1;
        self.function_names.insert(id, name.to_string());
        let builder = EvmLlvmBuilder { backend: self, function };
        Ok((builder, id))
    }

    fn verify_module(&mut self) -> Result<()> {
        self.module.verify().map_err(error_msg)
    }

    fn optimize_module(&mut self) -> Result<()> {
        // From `opt --help`, `-passes`.
        let passes = match self.opt_level {
            OptimizationLevel::None => "default<O0>",
            OptimizationLevel::Less => "default<O1>",
            OptimizationLevel::Default => "default<O2>",
            OptimizationLevel::Aggressive => "default<O3>",
        };
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
        self.exec_engine().get_function_address(name).map_err(Into::into)
    }

    unsafe fn free_function(&mut self, id: Self::FuncId) -> Result<()> {
        let name = self.id_to_name(id);
        let function = self.exec_engine().get_function_value(name)?;
        self.exec_engine().free_fn_machine_code(function);
        self.function_names.clear();
        Ok(())
    }

    unsafe fn free_all_functions(&mut self) -> Result<()> {
        if let Some(exec_engine) = &self.exec_engine {
            exec_engine.remove_module(&self.module).map_err(|e| Error::msg(e.to_string()))?;
        }
        self.module = create_module(self.cx, &self.machine)?;
        if self.exec_engine.is_some() {
            self.exec_engine =
                Some(self.module.create_jit_execution_engine(self.opt_level).map_err(error_msg)?);
        }
        Ok(())
    }
}

/// Cached target information.
struct TargetDefaults {
    triple: TargetTriple,
    cpu: String,
    features: String,
    // target: Target,
}

// SAFETY: No mutability is exposed and `TargetTriple` is an owned string.
unsafe impl std::marker::Send for TargetDefaults {}
unsafe impl std::marker::Sync for TargetDefaults {}

impl TargetDefaults {
    fn get() -> &'static Self {
        static TARGET_DEFAULTS: OnceLock<TargetDefaults> = OnceLock::new();
        TARGET_DEFAULTS.get_or_init(|| {
            let triple = TargetMachine::get_default_triple();
            let cpu = TargetMachine::get_host_cpu_name().to_string_lossy().into_owned();
            let features = TargetMachine::get_host_cpu_features().to_string_lossy().into_owned();
            TargetDefaults { triple, cpu, features }
        })
    }
}

/// The LLVM-based EVM bytecode compiler function builder.
#[derive(Debug)]
#[must_use]
pub struct EvmLlvmBuilder<'a, 'ctx> {
    backend: &'a mut EvmLlvmBackend<'ctx>,
    function: FunctionValue<'ctx>,
}

impl<'a, 'ctx> std::ops::Deref for EvmLlvmBuilder<'a, 'ctx> {
    type Target = EvmLlvmBackend<'ctx>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.backend
    }
}

impl<'a, 'ctx> std::ops::DerefMut for EvmLlvmBuilder<'a, 'ctx> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.backend
    }
}

impl<'a, 'ctx> EvmLlvmBuilder<'a, 'ctx> {
    fn assume_function(&mut self) -> FunctionValue<'ctx> {
        self.get_or_add_function("llvm.assume", |this| {
            this.ty_void.fn_type(&[this.ty_i1.into()], false)
        })
    }

    fn memcpy_inner(
        &mut self,
        dst: BasicValueEnum<'ctx>,
        src: BasicValueEnum<'ctx>,
        len: BasicValueEnum<'ctx>,
        inline: bool,
    ) {
        let dst = dst.into_pointer_value();
        let src = src.into_pointer_value();
        let len = len.into_int_value();
        let volatile = self.bool_const(false);
        let len_bits = len.get_type().get_bit_width();
        let name = format!("llvm.memcpy{}.p0.p0.i{len_bits}", if inline { ".inline" } else { "" });
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

    fn call_overflow_function(
        &mut self,
        name: &str,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> (BasicValueEnum<'ctx>, BasicValueEnum<'ctx>) {
        let f = self.get_overflow_function(name, lhs.get_type());
        let result = self.call(f, &[lhs, rhs]).unwrap();
        (self.extract_value(result, 0, "result"), self.extract_value(result, 1, "overflow"))
    }

    fn get_overflow_function(
        &mut self,
        name: &str,
        ty: BasicTypeEnum<'ctx>,
    ) -> FunctionValue<'ctx> {
        let bits = ty.into_int_type().get_bit_width();
        let name = format!("llvm.{name}.with.overflow.i{bits}");
        self.get_or_add_function(&name, |this| {
            this.fn_type(
                Some(this.cx.struct_type(&[ty, this.ty_i1.into()], false).into()),
                &[ty, ty],
            )
        })
    }

    fn get_or_add_function(
        &mut self,
        name: &str,
        mk_ty: impl FnOnce(&mut Self) -> FunctionType<'ctx>,
    ) -> FunctionValue<'ctx> {
        match self.module.get_function(name) {
            Some(function) => function,
            None => {
                let ty = mk_ty(self);
                self.module.add_function(name, ty, None)
            }
        }
    }
}

impl<'a, 'ctx> BackendTypes for EvmLlvmBuilder<'a, 'ctx> {
    type Type = BasicTypeEnum<'ctx>;
    type Value = BasicValueEnum<'ctx>;
    type StackSlot = PointerValue<'ctx>;
    type BasicBlock = BasicBlock<'ctx>;
    type Function = FunctionValue<'ctx>;
}

impl<'a, 'ctx> TypeMethods for EvmLlvmBuilder<'a, 'ctx> {
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

impl<'a, 'ctx> Builder for EvmLlvmBuilder<'a, 'ctx> {
    fn create_block(&mut self, name: &str) -> Self::BasicBlock {
        self.cx.append_basic_block(self.function, name)
    }

    fn create_block_after(&mut self, after: Self::BasicBlock, name: &str) -> Self::BasicBlock {
        self.cx.insert_basic_block_after(after, name)
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

    fn set_cold_block(&mut self, block: Self::BasicBlock) {
        let prev = self.current_block();
        if prev != Some(block) {
            self.switch_to_block(block);
        }

        let function = self.assume_function();
        let true_ = self.bool_const(true);
        let callsite = self.bcx.build_call(function, &[true_.into()], "cold").unwrap();
        let cold = self.cx.create_enum_attribute(Attribute::get_named_enum_kind_id("cold"), 1);
        callsite.add_attribute(AttributeLoc::Function, cold);

        if let Some(prev) = prev {
            if prev != block {
                self.switch_to_block(prev);
            }
        }
    }

    fn current_block(&mut self) -> Option<Self::BasicBlock> {
        self.bcx.get_insert_block()
    }

    fn add_comment_to_current_inst(&mut self, comment: &str) {
        let Some(block) = self.current_block() else { return };
        let Some(ins) = block.get_last_instruction() else { return };
        let metadata = self.cx.metadata_string(comment);
        let metadata = self.cx.metadata_node(&[metadata.into()]);
        ins.set_metadata(metadata, self.cx.get_kind_id("annotation")).unwrap();
    }

    fn fn_param(&mut self, index: usize) -> Self::Value {
        self.function.get_nth_param(index as _).unwrap()
    }

    fn bool_const(&mut self, value: bool) -> Self::Value {
        self.ty_i1.const_int(value as u64, false).into()
    }

    fn iconst(&mut self, ty: Self::Type, value: i64) -> Self::Value {
        // TODO: sign extend?
        ty.into_int_type().const_int(value as u64, value.is_negative()).into()
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

    fn new_stack_slot(&mut self, ty: Self::Type, name: &str) -> Self::StackSlot {
        self.bcx.build_alloca(ty, name).unwrap()
    }

    fn stack_load(&mut self, ty: Self::Type, slot: Self::StackSlot, name: &str) -> Self::Value {
        self.load(ty, slot.into(), name)
    }

    fn stack_store(&mut self, value: Self::Value, slot: Self::StackSlot) {
        self.store(value, slot.into())
    }

    fn stack_addr(&mut self, stack_slot: Self::StackSlot) -> Self::Value {
        stack_slot.into()
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, name: &str) -> Self::Value {
        self.bcx.build_load(ty, ptr.into_pointer_value(), name).unwrap()
    }

    fn store(&mut self, value: Self::Value, ptr: Self::Value) {
        self.bcx.build_store(ptr.into_pointer_value(), value).unwrap();
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

    fn switch(
        &mut self,
        index: Self::Value,
        default: Self::BasicBlock,
        targets: &[(Self::Value, Self::BasicBlock)],
    ) {
        let targets = targets.iter().map(|(v, b)| (v.into_int_value(), *b)).collect::<Vec<_>>();
        self.bcx.build_switch(index.into_int_value(), default, &targets).unwrap();
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
        then_value: impl FnOnce(&mut Self, Self::BasicBlock) -> Self::Value,
        else_value: impl FnOnce(&mut Self, Self::BasicBlock) -> Self::Value,
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
        let then_value = then_value(self, then_block);
        self.br(done_block);

        self.switch_to_block(else_block);
        let else_value = else_value(self, else_block);
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

    fn uadd_overflow(&mut self, lhs: Self::Value, rhs: Self::Value) -> (Self::Value, Self::Value) {
        self.call_overflow_function("uadd", lhs, rhs)
    }

    fn usub_overflow(&mut self, lhs: Self::Value, rhs: Self::Value) -> (Self::Value, Self::Value) {
        self.call_overflow_function("usub", lhs, rhs)

        // https://llvm.org/docs/Frontend/PerformanceTips.html
        // > Avoid using arithmetic intrinsics
        // Rustc also codegens this sequence for `usize::overflowing_sub`.

        // let result = self.isub(lhs, rhs);
        // let overflow = self.icmp(IntCC::UnsignedLessThan, lhs, rhs);
        // (result, overflow)
    }

    fn umax(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let ty = lhs.get_type();
        let bits = ty.into_int_type().get_bit_width();
        let name = format!("llvm.umax.i{bits}");
        let max = self.get_or_add_function(&name, |this| this.fn_type(Some(ty), &[ty, ty]));
        self.call(max, &[lhs, rhs]).unwrap()
    }

    fn umin(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let ty = lhs.get_type();
        let bits = ty.into_int_type().get_bit_width();
        let name = format!("llvm.umin.i{bits}");
        let max = self.get_or_add_function(&name, |this| this.fn_type(Some(ty), &[ty, ty]));
        self.call(max, &[lhs, rhs]).unwrap()
    }

    fn bswap(&mut self, value: Self::Value) -> Self::Value {
        let ty = value.get_type();
        let bits = ty.into_int_type().get_bit_width();
        assert!(bits % 16 == 0);
        let name = format!("llvm.bswap.i{bits}");
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

    fn gep(
        &mut self,
        elem_ty: Self::Type,
        ptr: Self::Value,
        indexes: &[Self::Value],
        name: &str,
    ) -> Self::Value {
        let indexes = indexes.iter().map(|idx| idx.into_int_value()).collect::<Vec<_>>();
        unsafe { self.bcx.build_in_bounds_gep(elem_ty, ptr.into_pointer_value(), &indexes, name) }
            .unwrap()
            .into()
    }

    fn extract_value(&mut self, value: Self::Value, index: u32, name: &str) -> Self::Value {
        self.bcx.build_extract_value(value.into_struct_value(), index, name).unwrap()
    }

    fn call(&mut self, function: Self::Function, args: &[Self::Value]) -> Option<Self::Value> {
        let args = args.iter().copied().map(Into::into).collect::<Vec<_>>();
        let callsite = self.bcx.build_call(function, &args, "").unwrap();
        callsite.try_as_basic_value().left()
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

    fn get_function(&mut self, name: &str) -> Option<Self::Function> {
        self.module.get_function(name)
    }

    fn add_function(
        &mut self,
        name: &str,
        ret: Option<Self::Type>,
        params: &[Self::Type],
        address: usize,
        linkage: revm_jit_backend::Linkage,
    ) -> Self::Function {
        let func_ty = self.fn_type(ret, params);
        let function = self.module.add_function(name, func_ty, Some(convert_linkage(linkage)));
        if let Some(exec_engine) = &self.exec_engine {
            exec_engine.add_global_mapping(&function, address);
        }
        function
    }

    fn add_function_attribute(
        &mut self,
        function: Option<Self::Function>,
        attribute: revm_jit_backend::Attribute,
        loc: revm_jit_backend::FunctionAttributeLocation,
    ) {
        let loc = convert_attribute_loc(loc);
        let attr = convert_attribute(self, attribute);
        function.unwrap_or(self.function).add_attribute(loc, attr);
    }
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
        let msg = unsafe { std::ffi::CStr::from_ptr(msg) };
        error!(msg = %msg.to_string_lossy(), "LLVM fatal error");
    }

    unsafe {
        install_fatal_error_handler(report_fatal_error);
    }

    let config = InitializationConfig {
        asm_parser: false,
        asm_printer: true,
        base: true,
        disassembler: true,
        info: true,
        machine_code: true,
    };
    Target::initialize_native(&config).map_err(Error::msg)
}

fn create_module<'ctx>(cx: &'ctx Context, machine: &TargetMachine) -> Result<Module<'ctx>> {
    let module_name = "evm";
    let module = cx.create_module(module_name);
    module.set_source_file_name(module_name);
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.set_triple(&machine.get_triple());
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

fn convert_opt_level(level: revm_jit_backend::OptimizationLevel) -> OptimizationLevel {
    match level {
        revm_jit_backend::OptimizationLevel::None => OptimizationLevel::None,
        revm_jit_backend::OptimizationLevel::Less => OptimizationLevel::Less,
        revm_jit_backend::OptimizationLevel::Default => OptimizationLevel::Default,
        revm_jit_backend::OptimizationLevel::Aggressive => OptimizationLevel::Aggressive,
    }
}

fn convert_opt_level_rev(level: OptimizationLevel) -> revm_jit_backend::OptimizationLevel {
    match level {
        OptimizationLevel::None => revm_jit_backend::OptimizationLevel::None,
        OptimizationLevel::Less => revm_jit_backend::OptimizationLevel::Less,
        OptimizationLevel::Default => revm_jit_backend::OptimizationLevel::Default,
        OptimizationLevel::Aggressive => revm_jit_backend::OptimizationLevel::Aggressive,
    }
}

fn convert_attribute(bcx: &EvmLlvmBuilder<'_, '_>, attr: revm_jit_backend::Attribute) -> Attribute {
    use revm_jit_backend::Attribute as OurAttr;

    enum AttrValue<'a> {
        String(&'a str),
        Enum(u64),
    }

    let cpu;
    let (key, value) = match attr {
        OurAttr::WillReturn => ("willreturn", AttrValue::Enum(1)),
        OurAttr::NoReturn => ("noreturn", AttrValue::Enum(1)),
        OurAttr::NoFree => ("nofree", AttrValue::Enum(1)),
        OurAttr::NoRecurse => ("norecurse", AttrValue::Enum(1)),
        OurAttr::NoSync => ("nosync", AttrValue::Enum(1)),
        OurAttr::NoUnwind => ("nounwind", AttrValue::Enum(1)),
        OurAttr::AllFramePointers => ("frame-pointer", AttrValue::String("all")),
        OurAttr::NativeTargetCpu => (
            "target-cpu",
            AttrValue::String({
                cpu = bcx.machine.get_cpu();
                cpu.to_str().unwrap()
            }),
        ),
        OurAttr::Cold => ("cold", AttrValue::Enum(1)),
        OurAttr::Hot => ("hot", AttrValue::Enum(1)),
        OurAttr::HintInline => ("inlinehint", AttrValue::Enum(1)),
        OurAttr::AlwaysInline => ("alwaysinline", AttrValue::Enum(1)),
        OurAttr::NoInline => ("noinline", AttrValue::Enum(1)),
        OurAttr::Speculatable => ("speculatable", AttrValue::Enum(1)),

        OurAttr::NoAlias => ("noalias", AttrValue::Enum(1)),
        OurAttr::NoCapture => ("nocapture", AttrValue::Enum(1)),
        OurAttr::NoUndef => ("noundef", AttrValue::Enum(1)),
        OurAttr::Align(n) => ("align", AttrValue::Enum(n)),
        OurAttr::NonNull => ("nonnull", AttrValue::Enum(1)),
        OurAttr::Dereferenceable(n) => ("dereferenceable", AttrValue::Enum(n)),
        OurAttr::ReadNone => ("readnone", AttrValue::Enum(1)),
        OurAttr::ReadOnly => ("readonly", AttrValue::Enum(1)),
        OurAttr::WriteOnly => ("writeonly", AttrValue::Enum(1)),
        OurAttr::Writable => ("writable", AttrValue::Enum(1)),

        attr => todo!("{attr:?}"),
    };
    match value {
        AttrValue::String(value) => bcx.cx.create_string_attribute(key, value),
        AttrValue::Enum(value) => {
            let id = Attribute::get_named_enum_kind_id(key);
            bcx.cx.create_enum_attribute(id, value)
        }
    }
}

fn convert_attribute_loc(loc: revm_jit_backend::FunctionAttributeLocation) -> AttributeLoc {
    match loc {
        revm_jit_backend::FunctionAttributeLocation::Return => AttributeLoc::Return,
        revm_jit_backend::FunctionAttributeLocation::Param(i) => AttributeLoc::Param(i),
        revm_jit_backend::FunctionAttributeLocation::Function => AttributeLoc::Function,
    }
}

fn convert_linkage(linkage: revm_jit_backend::Linkage) -> inkwell::module::Linkage {
    match linkage {
        revm_jit_backend::Linkage::Public => inkwell::module::Linkage::External,
        revm_jit_backend::Linkage::Import => inkwell::module::Linkage::External,
        revm_jit_backend::Linkage::Private => inkwell::module::Linkage::Private,
    }
}

fn error_msg(msg: inkwell::support::LLVMString) -> revm_jit_backend::Error {
    revm_jit_backend::Error::msg(msg.to_string_lossy().trim_end().to_string())
}

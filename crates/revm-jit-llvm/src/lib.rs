//! TODO
// #![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]

use inkwell::{
    attributes::{Attribute, AttributeLoc},
    context::Context,
    execution_engine::ExecutionEngine,
    module::Module,
    passes::PassBuilderOptions,
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine},
    types::{BasicType, IntType, PointerType, StringRadix},
    values::FunctionValue,
    AddressSpace, IntPredicate, OptimizationLevel,
};
use revm_jit_core::{Backend, Builder, ContextStack, Error, IntCC, JitEvmFn, Result};
use revm_primitives::U256;
use std::path::Path;

pub use inkwell;

/// The LLVM-based EVM JIT backend.
#[derive(Debug)]
#[must_use]
pub struct JitEvmLlvmBackend<'ctx> {
    cx: &'ctx Context,
    bcx: inkwell::builder::Builder<'ctx>,
    module: Module<'ctx>,
    exec_engine: ExecutionEngine<'ctx>,
    machine: TargetMachine,

    ty_ptr: PointerType<'ctx>,
    ty_i8: IntType<'ctx>,
    ty_i32: IntType<'ctx>,
    ty_i256: IntType<'ctx>,
    ty_isize: IntType<'ctx>,

    no_opt: bool,
}

impl<'ctx> JitEvmLlvmBackend<'ctx> {
    /// Creates a new LLVM-based EVM JIT backend.
    #[inline]
    pub fn new(cx: &'ctx Context) -> Result<Self> {
        let config = InitializationConfig {
            asm_parser: false,
            asm_printer: true,
            base: true,
            disassembler: true,
            info: true,
            machine_code: true,
        };
        Target::initialize_native(&config).map_err(Error::msg)?;

        let triple = TargetMachine::get_default_triple();
        let cpu = TargetMachine::get_host_cpu_name();
        let features = TargetMachine::get_host_cpu_features();
        let target = Target::from_triple(&triple).unwrap();
        let machine = target
            .create_target_machine(
                &triple,
                &cpu.to_string_lossy(),
                &features.to_string_lossy(),
                OptimizationLevel::Aggressive,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .unwrap();
        machine.set_asm_verbosity(true);

        let module = cx.create_module("evm");
        module.set_data_layout(&machine.get_target_data().get_data_layout());
        module.set_triple(&machine.get_triple());

        let exec_engine =
            module.create_jit_execution_engine(OptimizationLevel::Aggressive).map_err(error_msg)?;

        let bcx = cx.create_builder();

        let ty_i8 = cx.i8_type();
        let ty_i32 = cx.i32_type();
        let ty_i256 = cx.custom_width_int_type(256);
        let ty_isize = cx.ptr_sized_int_type(&machine.get_target_data(), None);
        let ty_ptr = ty_i8.ptr_type(AddressSpace::default());
        Ok(Self {
            cx,
            bcx,
            module,
            exec_engine,
            machine,
            ty_i8,
            ty_i32,
            ty_i256,
            ty_isize,
            ty_ptr,
            no_opt: false,
        })
    }

    /// Returns the LLVM context.
    #[inline]
    pub fn cx(&self) -> &'ctx Context {
        self.cx
    }
}

impl<'ctx> Backend for JitEvmLlvmBackend<'ctx> {
    type Builder<'a> = JitEvmLlvmBuilder<'a, 'ctx> where Self:'a;

    fn ir_extension(&self) -> &'static str {
        "ll"
    }

    fn set_is_dumping(&mut self, yes: bool) {
        let _ = yes;
    }

    fn no_optimize(&mut self) {
        self.no_opt = true;
    }

    fn dump_ir(&mut self, path: &Path) -> Result<()> {
        self.module.print_to_file(path).map_err(error_msg)
    }

    fn dump_disasm(&mut self, path: &Path) -> Result<()> {
        self.machine.write_to_file(&self.module, FileType::Assembly, path).map_err(error_msg)
    }

    fn build_function(&mut self, name: &str) -> Result<Self::Builder<'_>> {
        let params = &[self.ty_ptr.into()];
        let fn_type = self.ty_i32.fn_type(params, false);
        let function = self.module.add_function(name, fn_type, None);

        // Function attributes.
        for attr in [
            "mustprogress",
            "nofree",
            "norecurse",
            "nosync",
            "nounwind",
            "nonlazybind",
            "willreturn",
        ] {
            let id = Attribute::get_named_enum_kind_id(attr);
            let attr = self.cx.create_enum_attribute(id, 1);
            function.add_attribute(AttributeLoc::Function, attr);
        }
        {
            let cpu = self.machine.get_cpu();
            let attr = self.cx.create_string_attribute("target-cpu", cpu.to_str().unwrap());
            function.add_attribute(AttributeLoc::Function, attr);
        }

        // Pointer argument attributes.
        for (i, align, dereferenceable) in [(0u32, 32u64, ContextStack::SIZE as u64)] {
            for (name, value) in [
                ("noalias", 1),
                ("nocapture", 1),
                ("noundef", 1),
                ("align", align),
                ("dereferenceable", dereferenceable),
            ] {
                let id = Attribute::get_named_enum_kind_id(name);
                let attr = self.cx.create_enum_attribute(id, value);
                function.add_attribute(AttributeLoc::Param(i as _), attr);
            }
        }

        let entry = self.cx.append_basic_block(function, "entry");
        self.bcx.position_at_end(entry);

        Ok(JitEvmLlvmBuilder { backend: self, function })
    }

    fn optimize_function(&mut self, _name: &str) -> Result<()> {
        // TODO: Segfaults in tests
        if self.no_opt {
            return Ok(());
        }

        let passes = if self.no_opt { "default<O0>" } else { "default<O3>" };
        let opts = PassBuilderOptions::create();
        self.module.run_passes(passes, &self.machine, opts).map_err(error_msg)
    }

    fn get_function(&mut self, name: &str) -> Result<JitEvmFn> {
        unsafe { self.exec_engine.get_function(name) }
            .map(|f| unsafe { f.into_raw() })
            .map_err(Into::into)
    }
}

/// The LLVM-based EVM JIT builder.
#[derive(Debug)]
#[must_use]
pub struct JitEvmLlvmBuilder<'a, 'ctx> {
    backend: &'a mut JitEvmLlvmBackend<'ctx>,
    function: FunctionValue<'ctx>,
}

impl<'a, 'ctx> std::ops::Deref for JitEvmLlvmBuilder<'a, 'ctx> {
    type Target = JitEvmLlvmBackend<'ctx>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.backend
    }
}

impl<'a, 'ctx> std::ops::DerefMut for JitEvmLlvmBuilder<'a, 'ctx> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.backend
    }
}

impl<'a, 'ctx> Builder for JitEvmLlvmBuilder<'a, 'ctx> {
    type Type = inkwell::types::BasicTypeEnum<'ctx>;
    type Value = inkwell::values::BasicValueEnum<'ctx>;
    type StackSlot = inkwell::values::PointerValue<'ctx>;
    type BasicBlock = inkwell::basic_block::BasicBlock<'ctx>;

    fn type_ptr(&self) -> Self::Type {
        self.ty_ptr.into()
    }

    fn type_ptr_sized_int(&self) -> Self::Type {
        self.ty_isize.into()
    }

    fn type_int(&self, bits: u32) -> Self::Type {
        match bits {
            1 => self.cx.bool_type(),
            8 => self.ty_i8,
            16 => self.cx.i16_type(),
            32 => self.ty_i32,
            64 => self.cx.i64_type(),
            128 => self.cx.i128_type(),
            256 => self.ty_i256,
            bits => self.cx.custom_width_int_type(bits),
        }
        .into()
    }

    fn type_array(&self, ty: Self::Type, size: u32) -> Self::Type {
        ty.array_type(size).into()
    }

    fn create_block(&mut self) -> Self::BasicBlock {
        self.cx.append_basic_block(self.function, "")
    }

    fn switch_to_block(&mut self, block: Self::BasicBlock) {
        self.bcx.position_at_end(block);
    }

    fn seal_block(&mut self, block: Self::BasicBlock) {
        let _ = block;
        // Nothing to do.
    }

    fn set_cold_block(&mut self, block: Self::BasicBlock) {
        let _ = block;
        // TODO: call llvm assume with cold attribute
    }

    fn current_block(&mut self) -> Option<Self::BasicBlock> {
        self.bcx.get_insert_block()
    }

    fn fn_param(&mut self, index: usize) -> Self::Value {
        self.function.get_nth_param(index as _).unwrap()
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

    fn new_stack_slot(&mut self, ty: Self::Type, name: &str) -> Self::StackSlot {
        self.bcx.build_alloca(ty, name).unwrap()
    }

    fn stack_load(&mut self, ty: Self::Type, slot: Self::StackSlot, name: &str) -> Self::Value {
        self.load(ty, slot.into(), name)
    }

    fn stack_store(&mut self, value: Self::Value, slot: Self::StackSlot) {
        self.store(value, slot.into())
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, name: &str) -> Self::Value {
        self.bcx.build_load(ty, ptr.into_pointer_value(), name).unwrap()
    }

    fn store(&mut self, value: Self::Value, ptr: Self::Value) {
        self.bcx.build_store(ptr.into_pointer_value(), value).unwrap();
    }

    fn nop(&mut self) {
        // Nothing to do.
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
        let then_block = self.create_block();
        let else_block = self.create_block();
        let done_block = self.create_block();

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

    fn ipow(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let _ = lhs;
        let _ = rhs;
        todo!("ipow")
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

    fn gep_add(
        &mut self,
        elem_ty: Self::Type,
        ptr: Self::Value,
        offset: Self::Value,
    ) -> Self::Value {
        let offset = offset.into_int_value();
        unsafe { self.bcx.build_in_bounds_gep(elem_ty, ptr.into_pointer_value(), &[offset], "") }
            .unwrap()
            .into()
    }
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

fn error_msg(msg: inkwell::support::LLVMString) -> revm_jit_core::Error {
    revm_jit_core::Error::msg(msg.to_string())
}

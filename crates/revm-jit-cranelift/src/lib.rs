#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]

use cranelift::{codegen::ir::StackSlot, prelude::*};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, FuncOrDataId, Linkage, Module};
use revm_jit_core::{Backend, Builder, Error, Result};
use revm_primitives::U256;
use std::{io::Write, path::Path};

mod pretty_clif;

pub use cranelift;
pub use cranelift_jit;
pub use cranelift_module;
pub use cranelift_native;

/// The Cranelift-based EVM JIT backend.
#[allow(missing_debug_implementations)]
#[must_use]
pub struct JitEvmCraneliftBackend {
    /// The function builder context, which is reused across multiple FunctionBuilder instances.
    builder_context: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: codegen::Context,

    /// The module, with the jit backend, which manages the JIT'd functions.
    module: JITModule,
}

#[allow(clippy::new_without_default)]
impl JitEvmCraneliftBackend {
    /// Returns `Ok(())` if the current architecture is supported, or `Err(())` if the host machine
    /// is not supported in the current configuration.
    pub fn is_supported() -> Result<(), &'static str> {
        cranelift_native::builder().map(drop)
    }

    /// Creates a new instance of the JIT compiler.
    ///
    /// # Panics
    ///
    /// Panics if the current architecture is not supported. See
    /// [`is_supported`](Self::is_supported).
    #[track_caller]
    pub fn new() -> Self {
        let builder = JITBuilder::with_flags(
            &[("opt_level", "speed")],
            cranelift_module::default_libcall_names(),
        )
        .unwrap();
        let module = JITModule::new(builder);
        Self { builder_context: FunctionBuilderContext::new(), ctx: module.make_context(), module }
    }

    fn name_to_id(&mut self, name: &str) -> Result<FuncId> {
        let Some(FuncOrDataId::Func(id)) = self.module.get_name(name) else {
            return Err(Error::msg("function not found"));
        };
        Ok(id)
    }
}

impl Backend for JitEvmCraneliftBackend {
    type Builder<'a> = JitEvmCraneliftBuilder<'a>;

    fn ir_extension(&self) -> &'static str {
        "clif"
    }

    fn set_is_dumping(&mut self, yes: bool) {
        self.ctx.set_disasm(yes);
    }

    fn no_optimize(&mut self) {}

    fn dump_ir(&mut self, path: &Path) -> Result<()> {
        crate::pretty_clif::write_clif_file(
            path,
            self.module.isa(),
            &self.ctx.func,
            // &clif_comments,
            &Default::default(),
        );
        Ok(())
    }

    fn dump_disasm(&mut self, path: &Path) -> Result<()> {
        if let Some(disasm) = &self.ctx.compiled_code().unwrap().vcode {
            crate::pretty_clif::write_ir_file(path, |file| file.write_all(disasm.as_bytes()))
        }
        Ok(())
    }

    fn build_function(&mut self, name: &str) -> Result<Self::Builder<'_>> {
        let ptr_type = self.module.target_config().pointer_type();
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.returns.push(AbiParam::new(types::I32));
        let _id = self.module.declare_function(name, Linkage::Export, &self.ctx.func.signature)?;
        let bcx = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
        Ok(JitEvmCraneliftBuilder { bcx, ptr_type })
    }

    fn optimize_function(&mut self, name: &str) -> Result<()> {
        let id = self.name_to_id(name)?;

        // Define the function to jit. This finishes compilation, although
        // there may be outstanding relocations to perform. Currently, jit
        // cannot finish relocations until all functions to be called are
        // defined. For this toy demo for now, we'll just finalize the
        // function below.
        self.module.define_function(id, &mut self.ctx)?;

        // Now that compilation is finished, we can clear out the context state.
        self.module.clear_context(&mut self.ctx);

        // Finalize the functions which we just defined, which resolves any outstanding relocations
        // (patching in addresses, now that they're available).
        self.module.finalize_definitions()?;

        Ok(())
    }

    fn get_function(&mut self, name: &str) -> Result<revm_jit_core::JitEvmFn> {
        let id = self.name_to_id(name)?;
        let ptr = self.module.get_finalized_function(id);
        Ok(unsafe { std::mem::transmute(ptr) })
    }
}

/// The Cranelift-based EVM JIT function builder.
#[allow(missing_debug_implementations)]
pub struct JitEvmCraneliftBuilder<'a> {
    bcx: FunctionBuilder<'a>,
    ptr_type: Type,
}

impl<'a> Builder for JitEvmCraneliftBuilder<'a> {
    type Type = Type;
    type Value = Value;
    type StackSlot = StackSlot;
    type BasicBlock = Block;

    fn type_ptr(&self) -> Self::Type {
        self.ptr_type
    }

    fn type_ptr_sized_int(&self) -> Self::Type {
        self.ptr_type
    }

    fn type_int(&self, bits: u32) -> Self::Type {
        bits.try_into()
            .ok()
            .and_then(Type::int)
            .unwrap_or_else(|| panic!("unsupported int type with {bits} bits"))
    }

    fn type_array(&self, ty: Self::Type, size: u32) -> Self::Type {
        unimplemented!("type: {size} x {ty}")
    }

    fn create_block(&mut self) -> Self::BasicBlock {
        self.bcx.create_block()
    }

    fn switch_to_block(&mut self, block: Self::BasicBlock) {
        self.bcx.switch_to_block(block);
    }

    fn seal_block(&mut self, block: Self::BasicBlock) {
        self.bcx.seal_block(block);
    }

    fn set_cold_block(&mut self, block: Self::BasicBlock) {
        self.bcx.set_cold_block(block);
    }

    fn current_block(&mut self) -> Option<Self::BasicBlock> {
        self.bcx.current_block()
    }

    fn fn_param(&mut self, index: usize) -> Self::Value {
        let block = self.current_block().unwrap();
        self.bcx.block_params(block)[index]
    }

    fn iconst(&mut self, ty: Self::Type, value: i64) -> Self::Value {
        self.bcx.ins().iconst(ty, value)
    }

    fn iconst_256(&mut self, value: U256) -> Self::Value {
        let _ = value;
        unimplemented!("no i256 :(")
    }

    fn new_stack_slot(&mut self, ty: Self::Type) -> Self::StackSlot {
        self.bcx.create_sized_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: ty.bytes(),
        })
    }

    fn stack_load(&mut self, ty: Self::Type, slot: Self::StackSlot, offset: i32) -> Self::Value {
        self.bcx.ins().stack_load(ty, slot, offset)
    }

    fn stack_store(&mut self, value: Self::Value, slot: Self::StackSlot, offset: i32) {
        self.bcx.ins().stack_store(value, slot, offset);
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, offset: i32) -> Self::Value {
        self.bcx.ins().load(ty, MemFlags::trusted(), ptr, offset)
    }

    fn store(&mut self, value: Self::Value, ptr: Self::Value, offset: i32) {
        self.bcx.ins().store(MemFlags::trusted(), value, ptr, offset);
    }

    fn nop(&mut self) {
        self.bcx.ins().nop();
    }

    fn ret(&mut self, values: &[Self::Value]) {
        self.bcx.ins().return_(values);
    }

    fn icmp(
        &mut self,
        cond: revm_jit_core::IntCC,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Self::Value {
        self.bcx.ins().icmp(convert_intcc(cond), lhs, rhs)
    }

    fn icmp_imm(&mut self, cond: revm_jit_core::IntCC, lhs: Self::Value, rhs: i64) -> Self::Value {
        self.bcx.ins().icmp_imm(convert_intcc(cond), lhs, rhs)
    }

    fn br(&mut self, dest: Self::BasicBlock) {
        self.bcx.ins().jump(dest, &[]);
    }

    fn brif(
        &mut self,
        cond: Self::Value,
        then_block: Self::BasicBlock,
        else_block: Self::BasicBlock,
    ) {
        self.bcx.ins().brif(cond, then_block, &[], else_block, &[]);
    }

    fn iadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().iadd(lhs, rhs)
    }

    fn isub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().isub(lhs, rhs)
    }

    fn imul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().imul(lhs, rhs)
    }

    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().udiv(lhs, rhs)
    }

    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().sdiv(lhs, rhs)
    }

    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().urem(lhs, rhs)
    }

    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().srem(lhs, rhs)
    }

    fn iadd_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        self.bcx.ins().iadd_imm(lhs, rhs)
    }

    fn imul_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        self.bcx.ins().imul_imm(lhs, rhs)
    }

    fn gep_add(&mut self, ty: Self::Type, ptr: Self::Value, offset: Self::Value) -> Self::Value {
        let offset = self.bcx.ins().imul_imm(offset, ty.bytes() as i64);
        self.bcx.ins().iadd(ptr, offset)
    }
}

fn convert_intcc(cond: revm_jit_core::IntCC) -> IntCC {
    match cond {
        revm_jit_core::IntCC::Equal => IntCC::Equal,
        revm_jit_core::IntCC::NotEqual => IntCC::NotEqual,
        revm_jit_core::IntCC::SignedLessThan => IntCC::SignedLessThan,
        revm_jit_core::IntCC::SignedGreaterThanOrEqual => IntCC::SignedGreaterThanOrEqual,
        revm_jit_core::IntCC::SignedGreaterThan => IntCC::SignedGreaterThan,
        revm_jit_core::IntCC::SignedLessThanOrEqual => IntCC::SignedLessThanOrEqual,
        revm_jit_core::IntCC::UnsignedLessThan => IntCC::UnsignedLessThan,
        revm_jit_core::IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
        revm_jit_core::IntCC::UnsignedGreaterThan => IntCC::UnsignedGreaterThan,
        revm_jit_core::IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedLessThanOrEqual,
    }
}

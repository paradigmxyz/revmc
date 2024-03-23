#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

use cranelift::{
    codegen::ir::{FuncRef, StackSlot},
    prelude::*,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, FuncOrDataId, Linkage, Module};
use pretty_clif::CommentWriter;
use revm_jit_core::{
    Backend, BackendTypes, Builder, Error, OptimizationLevel, Result, TypeMethods,
};
use revm_primitives::{HashMap, U256};
use std::{cell::RefCell, io::Write, path::Path, rc::Rc};

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

    symbols: Symbols,

    opt_level: OptimizationLevel,
    comments: CommentWriter,
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
    pub fn new(opt_level: OptimizationLevel) -> Self {
        let symbols = Symbols::new();
        let module = mk_jit_module(opt_level, symbols.clone());
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
            symbols,
            opt_level,
            comments: CommentWriter::new(),
        }
    }

    fn name_to_id(&mut self, name: &str) -> Result<FuncId> {
        let Some(FuncOrDataId::Func(id)) = self.module.get_name(name) else {
            return Err(Error::msg("function not found"));
        };
        Ok(id)
    }
}

impl BackendTypes for JitEvmCraneliftBackend {
    type Type = Type;
    type Value = Value;
    type StackSlot = StackSlot;
    type BasicBlock = Block;
    type Function = FuncRef;
}

impl TypeMethods for JitEvmCraneliftBackend {
    fn type_ptr(&self) -> Self::Type {
        self.module.target_config().pointer_type()
    }

    fn type_ptr_sized_int(&self) -> Self::Type {
        self.type_ptr()
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
}

impl Backend for JitEvmCraneliftBackend {
    type Builder<'a> = JitEvmCraneliftBuilder<'a>;

    fn ir_extension(&self) -> &'static str {
        "clif"
    }

    fn set_is_dumping(&mut self, yes: bool) {
        self.ctx.set_disasm(yes);
    }

    fn set_debug_assertions(&mut self, yes: bool) {
        let _ = yes;
    }

    fn set_opt_level(&mut self, level: OptimizationLevel) {
        // Note that this will only affect new functions after a new module is created in
        // `free_all_functions`.
        self.opt_level = level;
    }

    fn dump_ir(&mut self, path: &Path) -> Result<()> {
        crate::pretty_clif::write_clif_file(
            path,
            self.module.isa(),
            &self.ctx.func,
            &self.comments,
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
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.params.push(AbiParam::new(ptr_type));
        self.ctx.func.signature.returns.push(AbiParam::new(types::I8));
        let _id = self.module.declare_function(name, Linkage::Export, &self.ctx.func.signature)?;
        let bcx = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
        Ok(JitEvmCraneliftBuilder {
            module: &mut self.module,
            comments: &mut self.comments,
            bcx,
            ptr_type,
            symbols: self.symbols.clone(),
        })
    }

    fn verify_function(&mut self, name: &str) -> Result<()> {
        let _ = name;
        Ok(())
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

        self.comments.clear();

        Ok(())
    }

    fn get_function(&mut self, name: &str) -> Result<revm_jit_core::RawJitEvmFn> {
        let id = self.name_to_id(name)?;
        let ptr = self.module.get_finalized_function(id);
        Ok(unsafe { std::mem::transmute(ptr) })
    }

    unsafe fn free_function(&mut self, name: &str) -> Result<()> {
        // This doesn't exist yet.
        let _ = name;
        Ok(())
    }

    unsafe fn free_all_functions(&mut self) -> Result<()> {
        // TODO: Can `free_memory` take `&mut self` pls?
        let new = mk_jit_module(self.opt_level, self.symbols.clone());
        let old = std::mem::replace(&mut self.module, new);
        unsafe { old.free_memory() };
        self.ctx = self.module.make_context();
        Ok(())
    }
}

/// The Cranelift-based EVM JIT function builder.
#[allow(missing_debug_implementations)]
pub struct JitEvmCraneliftBuilder<'a> {
    module: &'a mut JITModule,
    comments: &'a mut CommentWriter,
    bcx: FunctionBuilder<'a>,
    ptr_type: Type,
    symbols: Symbols,
}

impl<'a> BackendTypes for JitEvmCraneliftBuilder<'a> {
    type Type = Type;
    type Value = Value;
    type StackSlot = StackSlot;
    type BasicBlock = Block;
    type Function = FuncRef;
}

impl<'a> TypeMethods for JitEvmCraneliftBuilder<'a> {
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
}

impl<'a> Builder for JitEvmCraneliftBuilder<'a> {
    fn create_block(&mut self, name: &str) -> Self::BasicBlock {
        let block = self.bcx.create_block();
        if !name.is_empty() && self.comments.enabled() {
            self.comments.add_comment(block, name);
        }
        block
    }

    fn create_block_after(&mut self, after: Self::BasicBlock, name: &str) -> Self::BasicBlock {
        let block = self.create_block(name);
        self.bcx.insert_block_after(block, after);
        block
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

    fn add_comment_to_current_inst(&mut self, comment: &str) {
        let Some(block) = self.bcx.current_block() else { return };
        let Some(inst) = self.bcx.func.layout.last_inst(block) else { return };
        self.comments.add_comment(inst, comment);
    }

    fn fn_param(&mut self, index: usize) -> Self::Value {
        let block = self.current_block().unwrap();
        self.bcx.block_params(block)[index]
    }

    fn bool_const(&mut self, value: bool) -> Self::Value {
        self.iconst(types::I8, value as i64)
    }

    fn iconst(&mut self, ty: Self::Type, value: i64) -> Self::Value {
        self.bcx.ins().iconst(ty, value)
    }

    fn iconst_256(&mut self, value: U256) -> Self::Value {
        let _ = value;
        unimplemented!("no i256 :(")
    }

    fn str_const(&mut self, value: &str) -> Self::Value {
        let _ = value;
        todo!("str_const")
    }

    fn new_stack_slot(&mut self, ty: Self::Type, name: &str) -> Self::StackSlot {
        let _ = name;
        self.bcx.create_sized_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: ty.bytes(),
        })
    }

    fn stack_load(&mut self, ty: Self::Type, slot: Self::StackSlot, name: &str) -> Self::Value {
        let value = self.bcx.ins().stack_load(ty, slot, 0);
        let _ = name;
        // if !name.is_empty() {
        //     self.declare_var(ty, value, name);
        // }
        value
    }

    fn stack_store(&mut self, value: Self::Value, slot: Self::StackSlot) {
        self.bcx.ins().stack_store(value, slot, 0);
    }

    fn stack_addr(&mut self, stack_slot: Self::StackSlot) -> Self::Value {
        // self.bcx.ins().stack_addr(self., stack_slot, 0)
        let _ = stack_slot;
        todo!("stack_addr")
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, name: &str) -> Self::Value {
        let value = self.bcx.ins().load(ty, MemFlags::trusted(), ptr, 0);
        let _ = name;
        // if !name.is_empty() {
        //     self.declare_var(ty, value, name);
        // }
        value
    }

    fn store(&mut self, value: Self::Value, ptr: Self::Value) {
        self.bcx.ins().store(MemFlags::trusted(), value, ptr, 0);
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

    fn is_null(&mut self, ptr: Self::Value) -> Self::Value {
        self.bcx.ins().icmp_imm(IntCC::Equal, ptr, 0)
    }

    fn is_not_null(&mut self, ptr: Self::Value) -> Self::Value {
        self.bcx.ins().icmp_imm(IntCC::NotEqual, ptr, 0)
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

    fn select(
        &mut self,
        cond: Self::Value,
        then_value: Self::Value,
        else_value: Self::Value,
    ) -> Self::Value {
        self.bcx.ins().select(cond, then_value, else_value)
    }

    fn lazy_select(
        &mut self,
        cond: Self::Value,
        ty: Self::Type,
        then_value: impl FnOnce(&mut Self, Self::BasicBlock) -> Self::Value,
        else_value: impl FnOnce(&mut Self, Self::BasicBlock) -> Self::Value,
    ) -> Self::Value {
        let then_block = self.bcx.create_block();
        let else_block = self.bcx.create_block();
        let done_block = self.bcx.create_block();
        let done_value = self.bcx.append_block_param(done_block, ty);

        self.brif(cond, then_block, else_block);

        self.seal_block(then_block);
        self.switch_to_block(then_block);
        let then_value = then_value(self, then_block);
        self.bcx.ins().jump(done_block, &[then_value]);

        self.seal_block(else_block);
        self.switch_to_block(else_block);
        let else_value = else_value(self, else_block);
        self.bcx.ins().jump(done_block, &[else_value]);

        self.seal_block(done_block);
        self.switch_to_block(done_block);
        done_value
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

    fn isub_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        self.iadd_imm(lhs, -rhs)
    }

    fn imul_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        self.bcx.ins().imul_imm(lhs, rhs)
    }

    fn bitor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().bor(lhs, rhs)
    }

    fn bitand(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().band(lhs, rhs)
    }

    fn bitxor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().bxor(lhs, rhs)
    }

    fn bitnot(&mut self, value: Self::Value) -> Self::Value {
        self.bcx.ins().bnot(value)
    }

    fn ishl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().ishl(lhs, rhs)
    }

    fn ushr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().ushr(lhs, rhs)
    }

    fn sshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().sshr(lhs, rhs)
    }

    fn zext(&mut self, ty: Self::Type, value: Self::Value) -> Self::Value {
        self.bcx.ins().uextend(ty, value)
    }

    fn sext(&mut self, ty: Self::Type, value: Self::Value) -> Self::Value {
        self.bcx.ins().sextend(ty, value)
    }

    fn gep(&mut self, ty: Self::Type, ptr: Self::Value, offset: Self::Value) -> Self::Value {
        let offset = self.bcx.ins().imul_imm(offset, ty.bytes() as i64);
        self.bcx.ins().iadd(ptr, offset)
    }

    fn call(&mut self, function: Self::Function, args: &[Self::Value]) -> Option<Self::Value> {
        let ins = self.bcx.ins().call(function, args);
        self.bcx.inst_results(ins).first().copied()
    }

    fn unreachable(&mut self) {
        self.bcx.ins().trap(TrapCode::UnreachableCodeReached);
    }

    fn add_callback_function(
        &mut self,
        name: &str,
        ret: Option<Self::Type>,
        params: &[Self::Type],
        address: usize,
    ) -> Self::Function {
        let mut sig = self.module.make_signature();
        for ret in &ret {
            sig.returns.push(AbiParam::new(*ret));
        }
        for param in params {
            sig.params.push(AbiParam::new(*param));
        }
        self.symbols.insert(name.to_string(), address as *const u8);
        let id = self.module.declare_function(name, Linkage::Import, &sig).unwrap();
        self.module.declare_func_in_func(id, self.bcx.func)
    }
}

#[derive(Clone, Debug, Default)]
struct Symbols(Rc<RefCell<HashMap<String, *const u8>>>);

impl Symbols {
    fn new() -> Self {
        Self::default()
    }

    fn get(&self, name: &str) -> Option<*const u8> {
        self.0.borrow().get(name).copied()
    }

    fn insert(&self, name: String, ptr: *const u8) -> Option<*const u8> {
        self.0.borrow_mut().insert(name, ptr)
    }
}

fn mk_jit_module(opt_level: OptimizationLevel, symbols: Symbols) -> JITModule {
    let opt_level: &str = match opt_level {
        OptimizationLevel::None => "none",
        OptimizationLevel::Less | OptimizationLevel::Default | OptimizationLevel::Aggressive => {
            "speed"
        }
    };
    let mut builder = JITBuilder::with_flags(
        &[("opt_level", opt_level)],
        cranelift_module::default_libcall_names(),
    )
    .unwrap();
    builder.symbol_lookup_fn(Box::new(move |s| symbols.get(s)));
    JITModule::new(builder)
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

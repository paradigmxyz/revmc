#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

use codegen::ir::Function;
use cranelift::{
    codegen::ir::{FuncRef, StackSlot},
    prelude::*,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, FuncOrDataId, Linkage, Module, ModuleError};
use cranelift_object::{ObjectBuilder, ObjectModule};
use pretty_clif::CommentWriter;
use revmc_backend::{
    eyre::eyre, Backend, BackendTypes, Builder, OptimizationLevel, Result, TailCallKind,
    TypeMethods, U256,
};
use std::{
    collections::HashMap,
    io::Write,
    path::Path,
    sync::{Arc, RwLock},
};

mod pretty_clif;

pub use cranelift;
pub use cranelift_jit;
pub use cranelift_module;
pub use cranelift_native;

/// The Cranelift-based EVM bytecode compiler backend.
#[allow(missing_debug_implementations)]
#[must_use]
pub struct EvmCraneliftBackend {
    /// The function builder context, which is reused across multiple FunctionBuilder instances.
    builder_context: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: codegen::Context,

    /// The module, with the jit backend, which manages the JIT'd functions.
    module: ModuleWrapper,

    symbols: Symbols,

    opt_level: OptimizationLevel,
    comments: CommentWriter,
    functions: Vec<FuncId>,
}

#[allow(clippy::new_without_default)]
impl EvmCraneliftBackend {
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
    pub fn new(aot: bool, opt_level: OptimizationLevel) -> Self {
        let symbols = Symbols::new();
        let module = ModuleWrapper::new(aot, opt_level, &symbols).unwrap();
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.get().make_context(),
            module,
            symbols,
            opt_level,
            comments: CommentWriter::new(),
            functions: Vec::new(),
        }
    }

    fn finish_module(&mut self) -> Result<Option<ObjectModule>> {
        let aot = match self.module {
            ModuleWrapper::Jit(_) => {
                // TODO: Can `free_memory` take `&mut self` pls?
                let new = ModuleWrapper::new_jit(self.opt_level, self.symbols.clone())?;
                let ModuleWrapper::Jit(old) = std::mem::replace(&mut self.module, new) else {
                    unreachable!()
                };
                unsafe { old.free_memory() };
                None
            }
            ModuleWrapper::Aot(_) => {
                let new = ModuleWrapper::new_aot(self.opt_level)?;
                let ModuleWrapper::Aot(old) = std::mem::replace(&mut self.module, new) else {
                    unreachable!()
                };
                Some(old)
            }
        };
        self.module.get().clear_context(&mut self.ctx);
        Ok(aot)
    }
}

impl BackendTypes for EvmCraneliftBackend {
    type Type = Type;
    type Value = Value;
    type StackSlot = StackSlot;
    type BasicBlock = Block;
    type Function = FuncRef;
}

impl TypeMethods for EvmCraneliftBackend {
    fn type_ptr(&self) -> Self::Type {
        self.module.get().target_config().pointer_type()
    }

    fn type_ptr_sized_int(&self) -> Self::Type {
        self.type_ptr()
    }

    fn type_int(&self, bits: u32) -> Self::Type {
        bits.try_into().ok().and_then(Type::int).unwrap_or_else(|| unimplemented!("type: i{bits}"))
    }

    fn type_array(&self, ty: Self::Type, size: u32) -> Self::Type {
        unimplemented!("type: [{size} x {ty}]")
    }

    fn type_bit_width(&self, ty: Self::Type) -> u32 {
        ty.bits()
    }
}

impl Backend for EvmCraneliftBackend {
    type Builder<'a> = EvmCraneliftBuilder<'a>;
    type FuncId = FuncId;

    fn ir_extension(&self) -> &'static str {
        "clif"
    }

    fn set_module_name(&mut self, name: &str) {
        let _ = name;
    }

    fn set_is_dumping(&mut self, yes: bool) {
        self.ctx.set_disasm(yes);
    }

    fn set_debug_assertions(&mut self, yes: bool) {
        let _ = yes;
    }

    fn opt_level(&self) -> OptimizationLevel {
        self.opt_level
    }

    fn set_opt_level(&mut self, level: OptimizationLevel) {
        // Note that this will only affect new functions after a new module is created in
        // `free_all_functions`.
        self.opt_level = level;
    }

    fn is_aot(&self) -> bool {
        self.module.is_aot()
    }

    fn function_name_is_unique(&self, name: &str) -> bool {
        self.module.get().get_name(name).is_none()
    }

    fn dump_ir(&mut self, path: &Path) -> Result<()> {
        crate::pretty_clif::write_clif_file(
            path,
            self.module.get().isa(),
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

    fn build_function(
        &mut self,
        name: &str,
        ret: Option<Self::Type>,
        params: &[Self::Type],
        param_names: &[&str],
        linkage: revmc_backend::Linkage,
    ) -> Result<(Self::Builder<'_>, FuncId)> {
        self.ctx.func.clear();
        if let Some(ret) = ret {
            self.ctx.func.signature.returns.push(AbiParam::new(ret));
        }
        for param in params {
            self.ctx.func.signature.params.push(AbiParam::new(*param));
        }
        let _ = param_names;
        let ptr_type = self.type_ptr();
        let id = self.module.get_mut().declare_function(
            name,
            convert_linkage(linkage),
            &self.ctx.func.signature,
        )?;
        self.functions.push(id);
        let bcx = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
        let mut builder = EvmCraneliftBuilder {
            module: &mut self.module,
            comments: &mut self.comments,
            bcx,
            ptr_type,
            symbols: self.symbols.clone(),
        };
        let entry = builder.bcx.create_block();
        builder.bcx.append_block_params_for_function_params(entry);
        Ok((builder, id))
    }

    fn verify_module(&mut self) -> Result<()> {
        Ok(())
    }

    fn optimize_module(&mut self) -> Result<()> {
        // Define the function to jit. This finishes compilation, although
        // there may be outstanding relocations to perform. Currently, jit
        // cannot finish relocations until all functions to be called are
        // defined. For this toy demo for now, we'll just finalize the
        // function below.
        for &id in &self.functions {
            self.module.get_mut().define_function(id, &mut self.ctx)?;
        }
        self.functions.clear();

        // Now that compilation is finished, we can clear out the context state.
        self.module.get().clear_context(&mut self.ctx);

        // Finalize the functions which we just defined, which resolves any outstanding relocations
        // (patching in addresses, now that they're available).
        self.module.finalize_definitions()?;

        self.comments.clear();

        Ok(())
    }

    fn write_object<W: std::io::Write>(&mut self, w: W) -> Result<()> {
        let module =
            self.finish_module()?.ok_or_else(|| eyre!("cannot write object in JIT mode"))?;
        let product = module.finish();
        product.object.write_stream(w).map_err(|e| eyre!("{e}"))?;
        Ok(())
    }

    fn jit_function(&mut self, id: Self::FuncId) -> Result<usize> {
        self.module.get_finalized_function(id).map(|ptr| ptr as usize)
    }

    unsafe fn free_function(&mut self, id: Self::FuncId) -> Result<()> {
        // This doesn't exist yet.
        let _ = id;
        Ok(())
    }

    unsafe fn free_all_functions(&mut self) -> Result<()> {
        self.finish_module().map(drop)
    }
}

/// The Cranelift-based EVM bytecode compiler function builder.
#[allow(missing_debug_implementations)]
pub struct EvmCraneliftBuilder<'a> {
    module: &'a mut ModuleWrapper,
    comments: &'a mut CommentWriter,
    bcx: FunctionBuilder<'a>,
    ptr_type: Type,
    symbols: Symbols,
}

impl<'a> BackendTypes for EvmCraneliftBuilder<'a> {
    type Type = <EvmCraneliftBackend as BackendTypes>::Type;
    type Value = <EvmCraneliftBackend as BackendTypes>::Value;
    type StackSlot = <EvmCraneliftBackend as BackendTypes>::StackSlot;
    type BasicBlock = <EvmCraneliftBackend as BackendTypes>::BasicBlock;
    type Function = <EvmCraneliftBackend as BackendTypes>::Function;
}

impl<'a> TypeMethods for EvmCraneliftBuilder<'a> {
    fn type_ptr(&self) -> Self::Type {
        self.ptr_type
    }

    fn type_ptr_sized_int(&self) -> Self::Type {
        self.ptr_type
    }

    fn type_int(&self, bits: u32) -> Self::Type {
        bits.try_into().ok().and_then(Type::int).unwrap_or_else(|| unimplemented!("type: i{bits}"))
    }

    fn type_array(&self, ty: Self::Type, size: u32) -> Self::Type {
        unimplemented!("type: [{size} x {ty}]")
    }

    fn type_bit_width(&self, ty: Self::Type) -> u32 {
        ty.bits()
    }
}

impl<'a> Builder for EvmCraneliftBuilder<'a> {
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

    fn seal_all_blocks(&mut self) {
        self.bcx.seal_all_blocks();
    }

    fn set_current_block_cold(&mut self) {
        self.bcx.set_cold_block(self.bcx.current_block().unwrap());
    }

    fn current_block(&mut self) -> Option<Self::BasicBlock> {
        self.bcx.current_block()
    }

    fn block_addr(&mut self, _block: Self::BasicBlock) -> Option<Self::Value> {
        None
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

    fn num_fn_params(&self) -> usize {
        self.bcx.func.signature.params.len()
    }

    fn bool_const(&mut self, value: bool) -> Self::Value {
        self.iconst(types::I8, value as i64)
    }

    fn iconst(&mut self, ty: Self::Type, value: i64) -> Self::Value {
        self.bcx.ins().iconst(ty, value)
    }

    fn uconst(&mut self, ty: Self::Type, value: u64) -> Self::Value {
        self.iconst(ty, value as i64)
    }

    fn iconst_256(&mut self, value: U256) -> Self::Value {
        let _ = value;
        todo!("no i256 :(")
    }

    fn str_const(&mut self, value: &str) -> Self::Value {
        // https://github.com/rust-lang/rustc_codegen_cranelift/blob/1122338eb88648ec36a2eb2b1c27031fa897964d/src/common.rs#L432

        let mut data = DataDescription::new();
        data.define(value.as_bytes().into());
        let msg_id = self.module.get_mut().declare_anonymous_data(false, false).unwrap();

        // Ignore DuplicateDefinition error, as the data will be the same
        let _ = self.module.get_mut().define_data(msg_id, &data);

        let local_msg_id = self.module.get().declare_data_in_func(msg_id, self.bcx.func);
        if self.comments.enabled() {
            self.comments.add_comment(local_msg_id, value);
        }
        self.bcx.ins().global_value(self.ptr_type, local_msg_id)
    }

    fn nullptr(&mut self) -> Self::Value {
        self.iconst(self.ptr_type, 0)
    }

    fn new_stack_slot_raw(&mut self, ty: Self::Type, name: &str) -> Self::StackSlot {
        // https://github.com/rust-lang/rustc_codegen_cranelift/blob/1122338eb88648ec36a2eb2b1c27031fa897964d/src/common.rs#L388

        /*
        let _ = name;
        let abi_align = 16;
        if align <= abi_align {
            self.bcx.create_sized_stack_slot(StackSlotData {
                kind: StackSlotKind::ExplicitSlot,
                // FIXME Don't force the size to a multiple of <abi_align> bytes once Cranelift gets
                // a way to specify stack slot alignment.
                size: (size + abi_align - 1) / abi_align * abi_align,
            })
        } else {
            unimplemented!("{align} > {abi_align}")
            /*
            // Alignment is too big to handle using the above hack. Dynamically realign a stack slot
            // instead. This wastes some space for the realignment.
            let stack_slot = self.bcx.create_sized_stack_slot(StackSlotData {
                kind: StackSlotKind::ExplicitSlot,
                // FIXME Don't force the size to a multiple of <abi_align> bytes once Cranelift gets
                // a way to specify stack slot alignment.
                size: (size + align) / abi_align * abi_align,
            });
            let base_ptr = self.bcx.ins().stack_addr(self.pointer_type, stack_slot, 0);
            let misalign_offset = self.bcx.ins().urem_imm(base_ptr, i64::from(align));
            let realign_offset = self.bcx.ins().irsub_imm(misalign_offset, i64::from(align));
            Pointer::new(self.bcx.ins().iadd(base_ptr, realign_offset))
            */
        }
        */

        let _ = name;
        self.bcx.create_sized_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: ty.bytes(),
            align_shift: 1,
        })
    }

    fn stack_load(&mut self, ty: Self::Type, slot: Self::StackSlot, name: &str) -> Self::Value {
        let _ = name;
        self.bcx.ins().stack_load(ty, slot, 0)
    }

    fn stack_store(&mut self, value: Self::Value, slot: Self::StackSlot) {
        self.bcx.ins().stack_store(value, slot, 0);
    }

    fn stack_addr(&mut self, ty: Self::Type, slot: Self::StackSlot) -> Self::Value {
        self.bcx.ins().stack_addr(ty, slot, 0)
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, name: &str) -> Self::Value {
        let _ = name;
        self.bcx.ins().load(ty, MemFlags::trusted(), ptr, 0)
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
        cond: revmc_backend::IntCC,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Self::Value {
        self.bcx.ins().icmp(convert_intcc(cond), lhs, rhs)
    }

    fn icmp_imm(&mut self, cond: revmc_backend::IntCC, lhs: Self::Value, rhs: i64) -> Self::Value {
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

    fn switch(
        &mut self,
        index: Self::Value,
        default: Self::BasicBlock,
        targets: &[(u64, Self::BasicBlock)],
        default_is_cold: bool,
    ) {
        let _ = default_is_cold;
        let mut switch = cranelift::frontend::Switch::new();
        for (value, block) in targets {
            switch.set_entry(*value as u128, *block);
        }
        switch.emit(&mut self.bcx, index, default)
    }

    fn br_indirect(&mut self, _address: Self::Value, _destinations: &[Self::BasicBlock]) {
        unimplemented!()
    }

    fn phi(&mut self, ty: Self::Type, incoming: &[(Self::Value, Self::BasicBlock)]) -> Self::Value {
        let current = self.current_block().unwrap();
        let param = self.bcx.append_block_param(current, ty);
        for &(value, block) in incoming {
            self.bcx.switch_to_block(block);
            let last_inst = self.bcx.func.layout.last_inst(block).unwrap();
            let src = self.bcx.ins().jump(current, &[value]);
            self.bcx.func.transplant_inst(last_inst, src);
        }
        self.bcx.switch_to_block(current);
        param
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
        let done_value = self.bcx.append_block_param(done_block, ty);

        self.brif(cond, then_block, else_block);

        self.seal_block(then_block);
        self.switch_to_block(then_block);
        let then_value = then_value(self);
        self.bcx.ins().jump(done_block, &[then_value]);

        self.seal_block(else_block);
        self.switch_to_block(else_block);
        let else_value = else_value(self);
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

    fn uadd_overflow(&mut self, lhs: Self::Value, rhs: Self::Value) -> (Self::Value, Self::Value) {
        self.bcx.ins().uadd_overflow(lhs, rhs)
    }

    fn usub_overflow(&mut self, lhs: Self::Value, rhs: Self::Value) -> (Self::Value, Self::Value) {
        self.bcx.ins().usub_overflow(lhs, rhs)
    }

    fn uadd_sat(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().uadd_sat(lhs, rhs)
    }

    fn umax(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().umax(lhs, rhs)
    }

    fn umin(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.bcx.ins().umin(lhs, rhs)
    }

    fn bswap(&mut self, value: Self::Value) -> Self::Value {
        self.bcx.ins().bswap(value)
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

    fn bitor_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        self.bcx.ins().bor_imm(lhs, rhs)
    }

    fn bitand_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        self.bcx.ins().band_imm(lhs, rhs)
    }

    fn bitxor_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value {
        self.bcx.ins().bxor_imm(lhs, rhs)
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

    fn ireduce(&mut self, to: Self::Type, value: Self::Value) -> Self::Value {
        self.bcx.ins().ireduce(to, value)
    }

    fn gep(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        indexes: &[Self::Value],
        name: &str,
    ) -> Self::Value {
        let _ = name;
        let offset = self.bcx.ins().imul_imm(*indexes.first().unwrap(), ty.bytes() as i64);
        self.bcx.ins().iadd(ptr, offset)
    }

    fn tail_call(
        &mut self,
        function: Self::Function,
        args: &[Self::Value],
        tail_call: TailCallKind,
    ) -> Option<Self::Value> {
        if tail_call != TailCallKind::None {
            todo!();
        }
        let ins = self.bcx.ins().call(function, args);
        self.bcx.inst_results(ins).first().copied()
    }

    fn is_compile_time_known(&mut self, _value: Self::Value) -> Option<Self::Value> {
        None
    }

    fn memcpy(&mut self, dst: Self::Value, src: Self::Value, len: Self::Value) {
        let config = self.module.get().target_config();
        self.bcx.call_memcpy(config, dst, src, len)
    }

    fn unreachable(&mut self) {
        self.bcx.ins().trap(TrapCode::UnreachableCodeReached);
    }

    fn get_or_build_function(
        &mut self,
        name: &str,
        params: &[Self::Type],
        ret: Option<Self::Type>,
        linkage: revmc_backend::Linkage,
        build: impl FnOnce(&mut Self),
    ) -> Self::Function {
        if let Some(f) = self.get_function(name) {
            return f;
        }

        let mut sig = self.module.get().make_signature();
        if let Some(ret) = ret {
            sig.returns.push(AbiParam::new(ret));
        }
        for param in params {
            sig.params.push(AbiParam::new(*param));
        }

        let id =
            self.module.get_mut().declare_function(name, convert_linkage(linkage), &sig).unwrap();

        let mut func = Function::new();
        func.signature = sig;
        let mut builder_ctx = FunctionBuilderContext::new();
        let new_bcx = FunctionBuilder::new(&mut func, &mut builder_ctx);
        // TODO: SAFETY: Not really safe, lifetime extension.
        let new_bcx =
            unsafe { std::mem::transmute::<FunctionBuilder<'_>, FunctionBuilder<'a>>(new_bcx) };
        let old_bcx = std::mem::replace(&mut self.bcx, new_bcx);

        let f = self.module.get_mut().declare_func_in_func(id, self.bcx.func);

        let entry = self.bcx.create_block();
        self.bcx.append_block_params_for_function_params(entry);
        build(self);

        self.bcx = old_bcx;

        f
    }

    fn get_function(&mut self, name: &str) -> Option<Self::Function> {
        self.module
            .get()
            .get_name(name)
            .and_then(|id| match id {
                FuncOrDataId::Func(f) => Some(f),
                FuncOrDataId::Data(_) => None,
            })
            .map(|id| self.module.get_mut().declare_func_in_func(id, self.bcx.func))
    }

    fn get_printf_function(&mut self) -> Self::Function {
        if let Some(f) = self.get_function("printf") {
            return f;
        }

        unimplemented!()
    }

    fn add_function(
        &mut self,
        name: &str,
        params: &[Self::Type],
        ret: Option<Self::Type>,
        address: Option<usize>,
        linkage: revmc_backend::Linkage,
    ) -> Self::Function {
        let mut sig = self.module.get().make_signature();
        if let Some(ret) = ret {
            sig.returns.push(AbiParam::new(ret));
        }
        for param in params {
            sig.params.push(AbiParam::new(*param));
        }
        if let Some(address) = address {
            self.symbols.insert(name.to_string(), address as *const u8);
        }
        let id =
            self.module.get_mut().declare_function(name, convert_linkage(linkage), &sig).unwrap();
        self.module.get_mut().declare_func_in_func(id, self.bcx.func)
    }

    fn add_function_attribute(
        &mut self,
        function: Option<Self::Function>,
        attribute: revmc_backend::Attribute,
        loc: revmc_backend::FunctionAttributeLocation,
    ) {
        let _ = function;
        let _ = attribute;
        let _ = loc;
        // TODO
    }
}

#[derive(Clone, Debug, Default)]
struct Symbols(Arc<RwLock<HashMap<String, usize>>>);

impl Symbols {
    fn new() -> Self {
        Self::default()
    }

    fn get(&self, name: &str) -> Option<*const u8> {
        self.0.read().unwrap().get(name).copied().map(|addr| addr as *const u8)
    }

    fn insert(&self, name: String, ptr: *const u8) -> Option<*const u8> {
        self.0.write().unwrap().insert(name, ptr as usize).map(|addr| addr as *const u8)
    }
}

enum ModuleWrapper {
    Jit(JITModule),
    Aot(ObjectModule),
}

impl ModuleWrapper {
    fn new(aot: bool, opt_level: OptimizationLevel, symbols: &Symbols) -> Result<Self> {
        if aot {
            Self::new_aot(opt_level)
        } else {
            Self::new_jit(opt_level, symbols.clone())
        }
    }

    fn new_jit(opt_level: OptimizationLevel, symbols: Symbols) -> Result<Self> {
        let mut builder = JITBuilder::with_flags(
            &[("opt_level", opt_level_flag(opt_level))],
            cranelift_module::default_libcall_names(),
        )?;
        builder.symbol_lookup_fn(Box::new(move |s| symbols.get(s)));
        Ok(Self::Jit(JITModule::new(builder)))
    }

    fn new_aot(opt_level: OptimizationLevel) -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", opt_level_flag(opt_level))?;
        let isa_builder = cranelift_native::builder().map_err(|s| eyre!(s))?;
        let isa = isa_builder.finish(settings::Flags::new(flag_builder))?;

        let builder =
            ObjectBuilder::new(isa, "jit".to_string(), cranelift_module::default_libcall_names())?;
        Ok(Self::Aot(ObjectModule::new(builder)))
    }

    fn is_aot(&self) -> bool {
        matches!(self, Self::Aot(_))
    }

    #[inline]
    fn get(&self) -> &dyn Module {
        match self {
            Self::Jit(module) => module,
            Self::Aot(module) => module,
        }
    }

    #[inline]
    fn get_mut(&mut self) -> &mut dyn Module {
        match self {
            Self::Jit(module) => module,
            Self::Aot(module) => module,
        }
    }

    fn finalize_definitions(&mut self) -> Result<(), ModuleError> {
        match self {
            Self::Jit(module) => module.finalize_definitions(),
            Self::Aot(_) => Ok(()),
        }
    }

    fn get_finalized_function(&self, id: FuncId) -> Result<*const u8> {
        match self {
            Self::Jit(module) => Ok(module.get_finalized_function(id)),
            Self::Aot(_) => Err(eyre!("cannot get finalized JIT function in AOT mode")),
        }
    }
}

fn convert_intcc(cond: revmc_backend::IntCC) -> IntCC {
    match cond {
        revmc_backend::IntCC::Equal => IntCC::Equal,
        revmc_backend::IntCC::NotEqual => IntCC::NotEqual,
        revmc_backend::IntCC::SignedLessThan => IntCC::SignedLessThan,
        revmc_backend::IntCC::SignedGreaterThanOrEqual => IntCC::SignedGreaterThanOrEqual,
        revmc_backend::IntCC::SignedGreaterThan => IntCC::SignedGreaterThan,
        revmc_backend::IntCC::SignedLessThanOrEqual => IntCC::SignedLessThanOrEqual,
        revmc_backend::IntCC::UnsignedLessThan => IntCC::UnsignedLessThan,
        revmc_backend::IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
        revmc_backend::IntCC::UnsignedGreaterThan => IntCC::UnsignedGreaterThan,
        revmc_backend::IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedLessThanOrEqual,
    }
}

fn convert_linkage(linkage: revmc_backend::Linkage) -> Linkage {
    match linkage {
        revmc_backend::Linkage::Import => Linkage::Import,
        revmc_backend::Linkage::Public => Linkage::Export,
        revmc_backend::Linkage::Private => Linkage::Local,
    }
}

fn opt_level_flag(opt_level: OptimizationLevel) -> &'static str {
    match opt_level {
        OptimizationLevel::None => "none",
        OptimizationLevel::Less | OptimizationLevel::Default | OptimizationLevel::Aggressive => {
            "speed"
        }
    }
}

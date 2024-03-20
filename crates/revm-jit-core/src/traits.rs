use crate::{JitEvmFn, Result};
use revm_primitives::U256;
use std::{fmt, path::Path};

/// Integer comparison condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntCC {
    /// `==`.
    Equal,
    /// `!=`.
    NotEqual,
    /// Signed `<`.
    SignedLessThan,
    /// Signed `>=`.
    SignedGreaterThanOrEqual,
    /// Signed `>`.
    SignedGreaterThan,
    /// Signed `<=`.
    SignedLessThanOrEqual,
    /// Unsigned `<`.
    UnsignedLessThan,
    /// Unsigned `>=`.
    UnsignedGreaterThanOrEqual,
    /// Unsigned `>`.
    UnsignedGreaterThan,
    /// Unsigned `<=`.
    UnsignedLessThanOrEqual,
}

pub trait CodegenObject: Copy + PartialEq + fmt::Debug {}
impl<T: Copy + PartialEq + fmt::Debug> CodegenObject for T {}

pub trait Builder {
    type Type: CodegenObject;
    type Value: CodegenObject;
    type StackSlot: CodegenObject;
    type BasicBlock: CodegenObject;

    fn type_ptr(&self) -> Self::Type;
    fn type_ptr_sized_int(&self) -> Self::Type;
    fn type_int(&self, bits: u32) -> Self::Type;
    fn type_array(&self, ty: Self::Type, size: u32) -> Self::Type;

    fn create_block(&mut self) -> Self::BasicBlock;
    fn switch_to_block(&mut self, block: Self::BasicBlock);
    fn seal_block(&mut self, block: Self::BasicBlock);
    fn set_cold_block(&mut self, block: Self::BasicBlock);
    fn current_block(&mut self) -> Option<Self::BasicBlock>;

    fn fn_param(&mut self, index: usize) -> Self::Value;

    fn iconst(&mut self, ty: Self::Type, value: i64) -> Self::Value;
    fn iconst_256(&mut self, value: U256) -> Self::Value;

    fn new_stack_slot(&mut self, ty: Self::Type) -> Self::StackSlot;
    fn stack_load(&mut self, ty: Self::Type, slot: Self::StackSlot, offset: i32) -> Self::Value;
    fn stack_store(&mut self, value: Self::Value, slot: Self::StackSlot, offset: i32);

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, offset: i32) -> Self::Value;
    fn store(&mut self, value: Self::Value, ptr: Self::Value, offset: i32);

    fn nop(&mut self);
    fn ret(&mut self, values: &[Self::Value]);

    fn icmp(&mut self, cond: IntCC, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn icmp_imm(&mut self, cond: IntCC, lhs: Self::Value, rhs: i64) -> Self::Value;
    fn br(&mut self, dest: Self::BasicBlock);
    fn brif(
        &mut self,
        cond: Self::Value,
        then_block: Self::BasicBlock,
        else_block: Self::BasicBlock,
    );

    fn iadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn isub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn imul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;

    fn iadd_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value;
    fn imul_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value;

    fn gep_add(&mut self, ty: Self::Type, ptr: Self::Value, offset: Self::Value) -> Self::Value;
}

pub trait Backend {
    type Builder<'a>: Builder
    where
        Self: 'a;

    fn ir_extension(&self) -> &'static str;

    fn set_is_dumping(&mut self, yes: bool);
    fn no_optimize(&mut self);
    fn dump_ir(&mut self, path: &Path) -> Result<()>;
    fn dump_disasm(&mut self, path: &Path) -> Result<()>;

    fn build_function(&mut self, name: &str) -> Result<Self::Builder<'_>>;
    fn optimize_function(&mut self, name: &str) -> Result<()>;
    fn get_function(&mut self, name: &str) -> Result<JitEvmFn>;
}

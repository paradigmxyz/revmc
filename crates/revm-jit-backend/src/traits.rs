use crate::Result;
use ruint::aliases::U256;
use std::{fmt, path::Path};

/// Optimization level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OptimizationLevel {
    /// No optimizations.
    None,
    /// Less optimizations.
    Less,
    /// Default optimizations.
    Default,
    /// Aggressive optimizations.
    Aggressive,
}

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

/// Function or parameter attribute.
///
/// Mostly copied from [LLVM](https://llvm.org/docs/LangRef.html).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Attribute {
    // Function attributes.
    WillReturn,
    NoFree,
    NoRecurse,
    NoSync,
    NoUnwind,
    AllFramePointers,
    NativeTargetCpu,
    Cold,
    Hot,
    HintInline,
    AlwaysInline,
    NoInline,

    // Parameter attributes.
    NoAlias,
    NoCapture,
    NoUndef,
    Align(u64),
    NonNull,
    Dereferenceable(u64),
    ReadNone,
    ReadOnly,
    WriteOnly,
    Writeable,
    // TODO: Range?
}

/// Determines where on a function an attribute is assigned to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionAttributeLocation {
    /// Assign to the function's return type.
    Return,
    /// Assign to one of the function's params (0-indexed).
    Param(u32),
    /// Assign to the function itself.
    Function,
}

pub trait BackendTypes {
    type Type: Copy + Eq + fmt::Debug;
    type Value: Copy + Eq + fmt::Debug;
    type StackSlot: Copy + Eq + fmt::Debug;
    type BasicBlock: Copy + Eq + fmt::Debug;
    type Function: Copy + Eq + fmt::Debug;
}

#[allow(clippy::missing_safety_doc)]
pub trait Backend: BackendTypes + TypeMethods {
    type Builder<'a>: Builder<
        Type = Self::Type,
        Value = Self::Value,
        StackSlot = Self::StackSlot,
        BasicBlock = Self::BasicBlock,
        Function = Self::Function,
    >
    where
        Self: 'a;

    fn ir_extension(&self) -> &'static str;

    fn set_is_dumping(&mut self, yes: bool);
    fn set_debug_assertions(&mut self, yes: bool);
    fn set_opt_level(&mut self, level: OptimizationLevel);
    fn dump_ir(&mut self, path: &Path) -> Result<()>;
    fn dump_disasm(&mut self, path: &Path) -> Result<()>;

    fn build_function(
        &mut self,
        name: &str,
        ret: Option<Self::Type>,
        params: &[Self::Type],
        param_names: &[&str],
    ) -> Result<Self::Builder<'_>>;
    fn verify_function(&mut self, name: &str) -> Result<()>;
    fn optimize_function(&mut self, name: &str) -> Result<()>;
    fn get_function(&mut self, name: &str) -> Result<usize>;
    unsafe fn free_function(&mut self, name: &str) -> Result<()>;
    unsafe fn free_all_functions(&mut self) -> Result<()>;
}

pub trait TypeMethods: BackendTypes {
    fn type_ptr(&self) -> Self::Type;
    fn type_ptr_sized_int(&self) -> Self::Type;
    fn type_int(&self, bits: u32) -> Self::Type;
    fn type_array(&self, ty: Self::Type, size: u32) -> Self::Type;
}

pub trait Builder: BackendTypes + TypeMethods {
    fn create_block(&mut self, name: &str) -> Self::BasicBlock;
    fn create_block_after(&mut self, after: Self::BasicBlock, name: &str) -> Self::BasicBlock;
    fn switch_to_block(&mut self, block: Self::BasicBlock);
    fn seal_block(&mut self, block: Self::BasicBlock);
    fn seal_all_blocks(&mut self);
    fn set_cold_block(&mut self, block: Self::BasicBlock);
    fn current_block(&mut self) -> Option<Self::BasicBlock>;

    fn add_comment_to_current_inst(&mut self, comment: &str);

    fn fn_param(&mut self, index: usize) -> Self::Value;

    fn bool_const(&mut self, value: bool) -> Self::Value;
    fn iconst(&mut self, ty: Self::Type, value: i64) -> Self::Value;
    fn iconst_256(&mut self, value: U256) -> Self::Value;
    fn str_const(&mut self, value: &str) -> Self::Value;

    fn new_stack_slot(&mut self, ty: Self::Type, name: &str) -> Self::StackSlot;
    fn stack_load(&mut self, ty: Self::Type, slot: Self::StackSlot, name: &str) -> Self::Value;
    fn stack_store(&mut self, value: Self::Value, slot: Self::StackSlot);
    fn stack_addr(&mut self, stack_slot: Self::StackSlot) -> Self::Value;

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, name: &str) -> Self::Value;
    fn store(&mut self, value: Self::Value, ptr: Self::Value);

    fn nop(&mut self);
    fn ret(&mut self, values: &[Self::Value]);

    fn icmp(&mut self, cond: IntCC, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn icmp_imm(&mut self, cond: IntCC, lhs: Self::Value, rhs: i64) -> Self::Value;
    fn is_null(&mut self, ptr: Self::Value) -> Self::Value;
    fn is_not_null(&mut self, ptr: Self::Value) -> Self::Value;
    fn br(&mut self, dest: Self::BasicBlock);
    fn brif(
        &mut self,
        cond: Self::Value,
        then_block: Self::BasicBlock,
        else_block: Self::BasicBlock,
    );
    fn select(
        &mut self,
        cond: Self::Value,
        then_value: Self::Value,
        else_value: Self::Value,
    ) -> Self::Value;
    fn lazy_select(
        &mut self,
        cond: Self::Value,
        ty: Self::Type,
        then_value: impl FnOnce(&mut Self, Self::BasicBlock) -> Self::Value,
        else_value: impl FnOnce(&mut Self, Self::BasicBlock) -> Self::Value,
    ) -> Self::Value;

    fn iadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn isub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn imul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;

    fn iadd_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value;
    fn isub_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value;
    fn imul_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value;

    fn bitor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn bitand(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn bitxor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn bitnot(&mut self, value: Self::Value) -> Self::Value;

    fn bitor_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value;
    fn bitand_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value;
    fn bitxor_imm(&mut self, lhs: Self::Value, rhs: i64) -> Self::Value;

    fn ishl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn ushr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn sshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;

    fn zext(&mut self, ty: Self::Type, value: Self::Value) -> Self::Value;
    fn sext(&mut self, ty: Self::Type, value: Self::Value) -> Self::Value;

    fn gep(&mut self, ty: Self::Type, ptr: Self::Value, offset: Self::Value) -> Self::Value;
    fn extract_value(&mut self, value: Self::Value, index: u32) -> Self::Value;

    fn call(&mut self, function: Self::Function, args: &[Self::Value]) -> Option<Self::Value>;

    fn unreachable(&mut self);

    /// Adds a callback function to the IR that's located at `address`.
    fn add_callback_function(
        &mut self,
        name: &str,
        ret: Option<Self::Type>,
        params: &[Self::Type],
        address: usize,
    ) -> Self::Function;

    /// Adds an attribute to a function, one of its parameters, or its return value.
    ///
    /// If `function` is `None`, the attribute is added to the current function.
    fn add_function_attribute(
        &mut self,
        function: Option<Self::Function>,
        attribute: Attribute,
        loc: FunctionAttributeLocation,
    );
}
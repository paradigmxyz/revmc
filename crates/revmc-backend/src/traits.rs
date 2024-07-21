use crate::{Pointer, Result};
use ruint::aliases::U256;
use std::{fmt, path::Path};

/// Target machine.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Target {
    /// The host machine.
    Native,
    /// LLVM-style target triple.
    ///
    /// Ref: <https://llvm.org/docs/LangRef.html#target-triple>
    Triple {
        /// The target triple.
        triple: String,
        /// The target CPU.
        cpu: Option<String>,
        /// The target features string.
        features: Option<String>,
    },
}

impl Default for Target {
    fn default() -> Self {
        Self::Native
    }
}

impl std::str::FromStr for Target {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::triple(s))
    }
}

impl Target {
    /// Creates a target from a triple string.
    ///
    /// `cpu` and `features` are ignored if `triple` is `native`.
    pub fn new(
        triple: impl AsRef<str> + Into<String>,
        cpu: Option<String>,
        features: Option<String>,
    ) -> Self {
        if triple.as_ref() == "native" {
            return Self::Native;
        }
        Self::Triple { triple: triple.into(), cpu, features }
    }

    /// Creates a target from a triple string.
    pub fn triple(triple: impl AsRef<str> + Into<String>) -> Self {
        Self::new(triple, None, None)
    }
}

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

impl std::str::FromStr for OptimizationLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "0" | "none" => Self::None,
            "1" | "less" => Self::Less,
            "2" | "default" => Self::Default,
            "3" | "aggressive" => Self::Aggressive,
            _ => return Err(format!("unknown optimization level: {s}")),
        })
    }
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
    NoReturn,
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
    Speculatable,

    // Parameter attributes.
    NoAlias,
    NoCapture,
    NoUndef,
    Align(u64),
    NonNull,
    Dereferenceable(u64),
    /// Size of the return type in bytes.
    SRet(u64),
    ReadNone,
    ReadOnly,
    WriteOnly,
    Writable,
    // TODO: Range?
}

/// Linkage type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Linkage {
    /// Defined outside of the module.
    Import,
    /// Defined in the module and visible outside.
    Public,
    /// Defined in the module, but not visible outside.
    Private,
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

/// Tail call kind.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum TailCallKind {
    #[default]
    None,
    Tail,
    MustTail,
    NoTail,
}

pub trait BackendTypes: Sized {
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
    type FuncId: Copy + Eq + std::hash::Hash + fmt::Debug;

    fn ir_extension(&self) -> &'static str;

    fn set_module_name(&mut self, name: &str);

    fn set_is_dumping(&mut self, yes: bool);
    fn set_debug_assertions(&mut self, yes: bool);
    fn opt_level(&self) -> OptimizationLevel;
    fn set_opt_level(&mut self, level: OptimizationLevel);
    fn dump_ir(&mut self, path: &Path) -> Result<()>;
    fn dump_disasm(&mut self, path: &Path) -> Result<()>;

    fn is_aot(&self) -> bool;

    fn function_name_is_unique(&self, name: &str) -> bool;

    fn build_function(
        &mut self,
        name: &str,
        ret: Option<Self::Type>,
        params: &[Self::Type],
        param_names: &[&str],
        linkage: Linkage,
    ) -> Result<(Self::Builder<'_>, Self::FuncId)>;
    fn verify_module(&mut self) -> Result<()>;
    fn optimize_module(&mut self) -> Result<()>;
    fn write_object<W: std::io::Write>(&mut self, w: W) -> Result<()>;
    fn jit_function(&mut self, id: Self::FuncId) -> Result<usize>;
    unsafe fn free_function(&mut self, id: Self::FuncId) -> Result<()>;
    unsafe fn free_all_functions(&mut self) -> Result<()>;
}

pub trait TypeMethods: BackendTypes {
    fn type_ptr(&self) -> Self::Type;
    fn type_ptr_sized_int(&self) -> Self::Type;
    fn type_int(&self, bits: u32) -> Self::Type;
    fn type_array(&self, ty: Self::Type, size: u32) -> Self::Type;
    fn type_bit_width(&self, ty: Self::Type) -> u32;
}

pub trait Builder: BackendTypes + TypeMethods {
    fn create_block(&mut self, name: &str) -> Self::BasicBlock;
    fn create_block_after(&mut self, after: Self::BasicBlock, name: &str) -> Self::BasicBlock;
    fn switch_to_block(&mut self, block: Self::BasicBlock);
    fn seal_block(&mut self, block: Self::BasicBlock);
    fn seal_all_blocks(&mut self);
    fn set_current_block_cold(&mut self);
    fn current_block(&mut self) -> Option<Self::BasicBlock>;
    fn block_addr(&mut self, block: Self::BasicBlock) -> Option<Self::Value>;

    fn add_comment_to_current_inst(&mut self, comment: &str);

    fn fn_param(&mut self, index: usize) -> Self::Value;
    fn num_fn_params(&self) -> usize;

    fn bool_const(&mut self, value: bool) -> Self::Value;
    /// Sign-extends negative values to `ty`.
    fn iconst(&mut self, ty: Self::Type, value: i64) -> Self::Value;
    fn uconst(&mut self, ty: Self::Type, value: u64) -> Self::Value;
    fn iconst_256(&mut self, value: U256) -> Self::Value;
    fn cstr_const(&mut self, value: &std::ffi::CStr) -> Self::Value {
        self.str_const(value.to_str().unwrap())
    }
    fn str_const(&mut self, value: &str) -> Self::Value;
    fn nullptr(&mut self) -> Self::Value;

    fn new_stack_slot(&mut self, ty: Self::Type, name: &str) -> Pointer<Self> {
        Pointer::new_stack_slot(self, ty, name)
    }
    fn new_stack_slot_raw(&mut self, ty: Self::Type, name: &str) -> Self::StackSlot;
    fn stack_load(&mut self, ty: Self::Type, slot: Self::StackSlot, name: &str) -> Self::Value;
    fn stack_store(&mut self, value: Self::Value, slot: Self::StackSlot);
    fn stack_addr(&mut self, ty: Self::Type, slot: Self::StackSlot) -> Self::Value;

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
    fn brif_cold(
        &mut self,
        cond: Self::Value,
        then_block: Self::BasicBlock,
        else_block: Self::BasicBlock,
        then_is_cold: bool,
    ) {
        let _ = then_is_cold;
        self.brif(cond, then_block, else_block)
    }
    fn switch(
        &mut self,
        index: Self::Value,
        default: Self::BasicBlock,
        targets: &[(u64, Self::BasicBlock)],
        default_is_cold: bool,
    );
    fn br_indirect(&mut self, address: Self::Value, destinations: &[Self::BasicBlock]);
    fn phi(&mut self, ty: Self::Type, incoming: &[(Self::Value, Self::BasicBlock)]) -> Self::Value;
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
        then_value: impl FnOnce(&mut Self) -> Self::Value,
        else_value: impl FnOnce(&mut Self) -> Self::Value,
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

    // `(result, overflow)`
    fn uadd_overflow(&mut self, lhs: Self::Value, rhs: Self::Value) -> (Self::Value, Self::Value);
    fn usub_overflow(&mut self, lhs: Self::Value, rhs: Self::Value) -> (Self::Value, Self::Value);

    fn uadd_sat(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;

    fn umax(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn umin(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn bswap(&mut self, value: Self::Value) -> Self::Value;

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
    #[doc(alias = "trunc")]
    fn ireduce(&mut self, to: Self::Type, value: Self::Value) -> Self::Value;

    fn gep(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        indexes: &[Self::Value],
        name: &str,
    ) -> Self::Value;

    #[must_use]
    fn call(&mut self, function: Self::Function, args: &[Self::Value]) -> Option<Self::Value> {
        self.tail_call(function, args, TailCallKind::None)
    }
    #[must_use]
    fn tail_call(
        &mut self,
        function: Self::Function,
        args: &[Self::Value],
        tail_call: TailCallKind,
    ) -> Option<Self::Value>;

    /// Returns `Some(is_value_compile_time)`, or `None` if unsupported.
    fn is_compile_time_known(&mut self, value: Self::Value) -> Option<Self::Value>;

    fn memcpy(&mut self, dst: Self::Value, src: Self::Value, len: Self::Value);
    fn memcpy_inline(&mut self, dst: Self::Value, src: Self::Value, len: i64) {
        let len = self.iconst(self.type_int(64), len);
        self.memcpy(dst, src, len);
    }

    fn unreachable(&mut self);

    fn get_or_build_function(
        &mut self,
        name: &str,
        params: &[Self::Type],
        ret: Option<Self::Type>,
        linkage: Linkage,
        build: impl FnOnce(&mut Self),
    ) -> Self::Function;

    fn get_function(&mut self, name: &str) -> Option<Self::Function>;

    fn get_printf_function(&mut self) -> Self::Function;

    /// Adds a function to the module that's located at `address`.
    ///
    /// If `address` is `None`, the function must be built.
    fn add_function(
        &mut self,
        name: &str,
        params: &[Self::Type],
        ret: Option<Self::Type>,
        address: Option<usize>,
        linkage: Linkage,
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

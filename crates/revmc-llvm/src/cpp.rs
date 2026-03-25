//! Custom bindings to LLVM.

use inkwell::{
    attributes::Attribute,
    context::Context,
    llvm_sys::prelude::{LLVMAttributeRef, LLVMContextRef},
};

#[link(name = "revmc_llvm_cpp", kind = "static")]
unsafe extern "C" {
    fn revmc_llvm_create_initializes_attr(
        ctx: LLVMContextRef,
        lower: i64,
        upper: i64,
    ) -> LLVMAttributeRef;
}

pub(crate) fn create_initializes_attr(cx: &Context, lower: i64, upper: i64) -> Attribute {
    unsafe {
        let raw = revmc_llvm_create_initializes_attr(cx.raw(), lower, upper);
        Attribute::new(raw)
    }
}

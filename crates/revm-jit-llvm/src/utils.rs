use inkwell::support::LLVMString;
use std::ffi::c_char;

pub(crate) unsafe fn llvm_string(ptr: *const c_char) -> LLVMString {
    // `LLVMString::new` is private
    std::mem::transmute(ptr)
}

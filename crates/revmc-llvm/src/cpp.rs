//! Custom bindings to LLVM.

use inkwell::{
    attributes::Attribute,
    context::Context,
    llvm_sys::{
        error::LLVMErrorRef,
        orc2::{
            LLVMOrcExecutionSessionRef, LLVMOrcExecutorAddress, LLVMOrcJITDylibRef,
            lljit::{LLVMOrcLLJITBuilderRef, LLVMOrcLLJITRef},
        },
        prelude::{LLVMAttributeRef, LLVMContextRef},
    },
};
use std::ffi::{c_char, c_void};

#[link(name = "revmc_llvm_cpp", kind = "static")]
unsafe extern "C" {
    fn revmc_llvm_create_initializes_attr(
        ctx: LLVMContextRef,
        lower: i64,
        upper: i64,
    ) -> LLVMAttributeRef;

    pub(crate) fn revmc_llvm_jit_dylib_add_to_link_order(
        jd: LLVMOrcJITDylibRef,
        other: LLVMOrcJITDylibRef,
    );

    pub(crate) fn revmc_llvm_lljit_builder_set_dual_compiler(
        builder: LLVMOrcLLJITBuilderRef,
    ) -> *mut c_void;

    pub(crate) fn revmc_llvm_asm_capture_request(ctx: *mut c_void);

    pub(crate) fn revmc_llvm_asm_capture_data(ctx: *const c_void, len: *mut usize) -> *const u8;

    pub(crate) fn revmc_llvm_execution_session_remove_jit_dylib(
        es: LLVMOrcExecutionSessionRef,
        jd: LLVMOrcJITDylibRef,
    ) -> LLVMErrorRef;

    pub(crate) fn revmc_llvm_lljit_lookup_in(
        jit: LLVMOrcLLJITRef,
        jd: LLVMOrcJITDylibRef,
        result: *mut LLVMOrcExecutorAddress,
        name: *const c_char,
    ) -> LLVMErrorRef;

    pub(crate) fn revmc_llvm_lljit_enable_perf_support(jit: LLVMOrcLLJITRef) -> LLVMErrorRef;
}

pub(crate) fn create_initializes_attr(cx: &Context, lower: i64, upper: i64) -> Attribute {
    unsafe {
        let raw = revmc_llvm_create_initializes_attr(cx.raw(), lower, upper);
        Attribute::new(raw)
    }
}

/// Opaque handle to the C++ `RevmcAsmCaptureCtx` shared with the DualOutputCompiler.
///
/// The context lives as long as the process-global LLJIT. It is NOT owned by
/// this handle — it is intentionally leaked on the C++ side.
#[derive(Debug)]
pub(crate) struct AsmCaptureCtx {
    pub(crate) ptr: *mut c_void,
}

// The context is shared with the global LLJIT which is Send+Sync.
unsafe impl Send for AsmCaptureCtx {}
unsafe impl Sync for AsmCaptureCtx {}

impl AsmCaptureCtx {
    /// Requests capture — the next JIT compilation will emit verbose assembly
    /// into the internal buffer.
    pub(crate) fn request(&self) {
        unsafe { revmc_llvm_asm_capture_request(self.ptr) };
    }

    /// Returns the captured assembly text, or empty if nothing was captured.
    pub(crate) fn get(&self) -> &str {
        let mut len = 0usize;
        let data = unsafe { revmc_llvm_asm_capture_data(self.ptr, &mut len) };
        if len == 0 {
            return "";
        }
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(data, len)) }
    }
}

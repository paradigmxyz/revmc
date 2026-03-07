#![allow(clippy::needless_doctest_main)]
#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[macro_use]
extern crate tracing;

// For features.
use alloy_primitives as _;

mod bytecode;
pub use bytecode::*;

mod compiler;
pub use compiler::{EvmCompiler, EvmCompilerInput};

mod linker;
pub use linker::Linker;

/// Internal tests and testing utilities. Not public API.
#[cfg(any(test, feature = "__fuzzing"))]
pub mod tests;

#[allow(ambiguous_glob_reexports)]
#[doc(inline)]
pub use revmc_backend::*;
#[allow(ambiguous_glob_reexports)]
#[doc(inline)]
pub use revmc_context::*;

#[cfg(feature = "llvm")]
#[doc(no_inline)]
pub use llvm::EvmLlvmBackend;
#[cfg(feature = "llvm")]
#[doc(inline)]
pub use revmc_llvm as llvm;

#[cfg(feature = "cranelift")]
#[doc(no_inline)]
pub use cranelift::EvmCraneliftBackend;
#[cfg(feature = "cranelift")]
#[doc(inline)]
pub use revmc_cranelift as cranelift;

#[doc(no_inline)]
pub use revm_bytecode;
#[doc(no_inline)]
pub use revm_context_interface as context_interface;
#[doc(no_inline)]
pub use revm_interpreter::{self as interpreter};
#[doc(no_inline)]
pub use revm_primitives as primitives;
#[doc(no_inline)]
pub use revm_primitives::hardfork::SpecId;

const I256_MIN: U256 = U256::from_limbs([
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x8000000000000000,
]);

#[doc(hidden)]
#[cfg(feature = "llvm")]
pub fn with_llvm_backend<R>(
    opt_level: OptimizationLevel,
    aot: bool,
    f: impl FnOnce(EvmLlvmBackend<'_>) -> R,
) -> R {
    llvm::with_llvm_context(|cx| f(EvmLlvmBackend::new(cx, aot, opt_level).unwrap()))
}

#[doc(hidden)]
#[cfg(feature = "llvm")]
pub fn with_compiler<R>(
    opt_level: OptimizationLevel,
    aot: bool,
    f: impl FnOnce(&mut EvmCompiler<EvmLlvmBackend<'_>>) -> R,
) -> R {
    with_llvm_backend(opt_level, aot, |backend| f(&mut EvmCompiler::new(backend)))
}

#[doc(hidden)]
#[cfg(feature = "llvm")]
pub fn with_jit_compiler<R>(
    opt_level: OptimizationLevel,
    f: fn(&mut EvmCompiler<EvmLlvmBackend<'_>>) -> R,
) -> R {
    with_compiler(opt_level, false, f)
}

/// Enable for `cargo asm -p revmc --lib`.
#[cfg(any())]
pub fn generate_all_assembly() -> EvmCompiler<EvmLlvmBackend<'static>> {
    let cx = Box::leak(Box::new(llvm::inkwell::context::Context::create()));
    let mut compiler =
        EvmCompiler::new(EvmLlvmBackend::new(cx, OptimizationLevel::Aggressive).unwrap());
    let _ = compiler.jit(None, &[], primitives::SpecId::ARROW_GLACIER).unwrap();
    unsafe { compiler.clear().unwrap() };
    compiler
}

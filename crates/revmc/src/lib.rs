#![allow(clippy::needless_doctest_main)]
#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[macro_use]
extern crate tracing;

mod bytecode;
pub use bytecode::*;

mod compiler;
pub use compiler::{CompileTimings, EvmCompiler, EvmCompilerInput};

mod linker;
pub use linker::{Linker, shared_library_path};

#[cfg(feature = "alloy-evm")]
pub mod alloy_evm;

pub mod revm_evm;

pub mod runtime;

/// ABI version of compiled artifacts. Bump when the calling convention changes.
pub const ABI_VERSION: u32 = 0;

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

#[doc(no_inline)]
pub use revm_bytecode;
#[doc(no_inline)]
pub use revm_context_interface as context_interface;
#[doc(no_inline)]
pub use revm_handler as handler;
#[doc(no_inline)]
pub use revm_interpreter::{self as interpreter};
#[doc(no_inline)]
pub use revm_primitives as primitives;
#[doc(no_inline)]
pub use revm_primitives::hardfork::SpecId;

type FxHashMap<K, V> = alloy_primitives::map::HashMap<K, V, alloy_primitives::map::FxBuildHasher>;

/// Enable for `cargo asm -p revmc --lib`.
#[cfg(any())]
pub fn generate_all_assembly() -> EvmCompiler<EvmLlvmBackend> {
    let mut compiler = EvmCompiler::new(EvmLlvmBackend::new(false).unwrap());
    let _ = compiler.jit(None, &[], primitives::SpecId::ARROW_GLACIER).unwrap();
    unsafe { compiler.clear().unwrap() };
    compiler
}

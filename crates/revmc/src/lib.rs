#![allow(clippy::needless_doctest_main)]
#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[doc(inline)]
pub use revmc_codegen::*;
#[doc(inline)]
pub use revmc_runtime::{revm_evm, runtime};

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

#[doc(hidden)]
pub use revmc_builtins as builtins;

#[doc(no_inline)]
pub use revm_bytecode;
#[doc(no_inline)]
pub use revm_context_interface as context_interface;
#[doc(no_inline)]
pub use revm_handler as handler;
#[doc(no_inline)]
pub use revm_inspector as inspector;
#[doc(no_inline)]
pub use revm_interpreter::{self as interpreter};
#[doc(no_inline)]
pub use revm_primitives as primitives;
#[doc(no_inline)]
pub use revm_primitives::hardfork::SpecId;

#[cfg(feature = "alloy-evm")]
#[doc(inline)]
pub use revmc_runtime::alloy_evm;

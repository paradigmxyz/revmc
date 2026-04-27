#![doc = include_str!("../README.md")]
#![allow(clippy::needless_doctest_main)]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[macro_use]
extern crate tracing;

pub use ::eyre;
pub use revmc_codegen::*;

use revmc_backend::OptimizationLevel;
use revmc_context::EvmCompilerFn;
#[cfg(feature = "llvm")]
use revmc_llvm as llvm;
#[cfg(feature = "llvm")]
use revmc_llvm::EvmLlvmBackend;

pub mod runtime;

pub mod revm_evm;

#[cfg(feature = "alloy-evm")]
pub mod alloy_evm;

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

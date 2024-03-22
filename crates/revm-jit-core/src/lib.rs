#![doc = include_str!("../README.md")]
#![allow(missing_docs)]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

#[macro_use]
mod macros;

mod context;
pub use context::*;

mod traits;
pub use traits::*;

#[doc(no_inline)]
pub use revm_interpreter as interpreter;
#[doc(no_inline)]
pub use revm_interpreter::InstructionResult;
#[doc(no_inline)]
pub use revm_primitives as primitives;

/// JIT compilation result.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// JIT compilation error.
pub type Error = color_eyre::eyre::Error;

// Not public API.
#[doc(hidden)]
pub mod private {
    pub use tracing;
}

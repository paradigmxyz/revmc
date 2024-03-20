//! TODO

#![allow(missing_docs)]

mod context;
pub use context::*;

mod traits;
pub use traits::*;

/// JIT compilation result.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// JIT compilation error.
pub type Error = color_eyre::eyre::Error;

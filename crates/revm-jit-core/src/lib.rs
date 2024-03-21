//! TODO

#![allow(missing_docs)]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

mod context;
pub use context::*;

mod traits;
pub use traits::*;

/// JIT compilation result.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// JIT compilation error.
pub type Error = color_eyre::eyre::Error;

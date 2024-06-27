#![doc = include_str!("../README.md")]
#![allow(missing_docs)]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

mod traits;
pub use traits::*;

#[doc(no_inline)]
pub use eyre;
#[doc(no_inline)]
pub use ruint::{self, aliases::U256, uint};

mod pointer;
pub use pointer::{Pointer, PointerBase};

/// Compilation result.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Compilation error.
pub type Error = eyre::Error;

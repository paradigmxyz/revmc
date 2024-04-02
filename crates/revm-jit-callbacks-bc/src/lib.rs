#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

/// The bitcode of the callbacks module.
#[cfg(llvm_bc)]
pub const CALLBACKS_BITCODE: Option<&[u8]> =
    Some(include_bytes!(concat!(env!("OUT_DIR"), "/callbacks.bc")));

/// The bitcode of the callbacks module.
#[cfg(not(llvm_bc))]
pub const CALLBACKS_BITCODE: Option<&[u8]> = None;

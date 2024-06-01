#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

use melior::Context;

/// The MLIR-based EVM bytecode compiler backend.
#[derive(Debug)]
#[must_use]
pub struct EvmMlirBackend<'ctx> {
    #[allow(dead_code)]
    ctx: &'ctx Context,
}

impl<'ctx> EvmMlirBackend<'ctx> {
    /// Creates a new MLIR backend.
    pub fn new(ctx: &'ctx Context) -> Self {
        Self { ctx }
    }
}

//! Minimal, `no_std` runner.

#![no_std]

extern crate alloc;

// This dependency is needed to define the necessary symbols used by the compiled bytecodes,
// but we don't use it directly, so silence the unused crate dependency warning.
use revmc_builtins as _;

use revm::{
    MainnetEvm,
    context::{BlockEnv, CfgEnv, Context, Journal, TxEnv},
    database_interface::Database,
    handler::MainBuilder,
    primitives::{B256, hardfork::SpecId, hex},
};
use revmc_context::EvmCompilerFn;

include!("./common.rs");

// The bytecode we statically linked.
revmc_context::extern_revmc! {
    fn fibonacci;
}

/// External context for tracking compiled functions.
#[derive(Clone, Default)]
pub struct ExternalContext;

impl ExternalContext {
    /// Creates a new external context.
    pub fn new() -> Self {
        Self
    }

    /// Get a compiled function for the given bytecode hash.
    pub fn get_function(&self, bytecode_hash: B256) -> Option<EvmCompilerFn> {
        // Can use any mapping between bytecode hash and function.
        if bytecode_hash == B256::from(FIBONACCI_HASH) {
            return Some(EvmCompilerFn::new(fibonacci));
        }
        None
    }
}

/// Type alias for mainnet context with custom database.
pub type MainnetContext<DB> = Context<BlockEnv, TxEnv, CfgEnv, DB, Journal<DB>, ()>;

/// Build a mainnet EVM.
///
/// Note: In revm v34, the frame execution is handled differently.
/// For now, this returns a standard mainnet EVM. To integrate compiled
/// bytecode, you would need to customize the instruction handler or
/// use the revmc-context's call_with_interpreter method directly.
pub fn build_evm<DB: Database>(db: DB) -> MainnetEvm<MainnetContext<DB>> {
    Context::<BlockEnv, TxEnv, CfgEnv, DB, Journal<DB>, ()>::new(db, SpecId::CANCUN).build_mainnet()
}

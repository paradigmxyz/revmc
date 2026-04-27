//! Minimal runner that hooks a statically compiled bytecode into a revm mainnet EVM.

use revm_context::{BlockEnv, CfgEnv, Context, Journal, TxEnv};
use revm_database_interface::Database;
use revm_handler::{MainBuilder, MainnetEvm};
use revm_primitives::{B256, hardfork::SpecId, hex, map::B256Map};
use revmc::{JitEvm, RawEvmCompilerFn};

include!("./common.rs");

// The bytecode we statically linked.
revmc::extern_revmc! {
    fn fibonacci;
}

/// Type alias for mainnet context with custom database.
pub type MainnetContext<DB> = Context<BlockEnv, TxEnv, CfgEnv, DB, Journal<DB>, ()>;

/// Build a mainnet EVM wrapped in [`JitEvm`] so that calls to known code hashes
/// are dispatched to compiled functions instead of the interpreter.
pub fn build_evm<DB: Database>(db: DB) -> JitEvm<MainnetEvm<MainnetContext<DB>>> {
    let inner = Context::<BlockEnv, TxEnv, CfgEnv, DB, Journal<DB>, ()>::new(db, SpecId::CANCUN)
        .build_mainnet();

    let mut functions = B256Map::default();
    functions.insert(B256::from(FIBONACCI_HASH), fibonacci as RawEvmCompilerFn);

    JitEvm::new(inner, functions)
}

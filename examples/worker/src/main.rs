//! Simple compiler worker example.

use revm::{
    db::{CacheDB, EmptyDB},
    primitives::{
        address, hex, AccessList, AccessListItem, AccountInfo, Bytecode, TransactTo, B256, U256,
    },
};
use revmc_worker::{register_handler, EXTCompileWorker};
use std::{sync::Arc, thread};

pub const FIBONACCI_CODE: &[u8] =
    &hex!("5f355f60015b8215601a578181019150909160019003916005565b9150505f5260205ff3");

/// First call executes the transaction and compiles into embedded db
/// embedded db: ~/.revmc/db, ~/.revmc/output
/// It is crucial to reset the embedded db and do 'cargo clean' for reproducing the same steps
/// Otherwise, both calls will utilize cached ExternalFn or unexpected behavior will happen
///
/// Second call loads the ExternalFn from embedded db to cache
/// and executes transaction with it
fn main() {
    let ext_worker = Arc::new(EXTCompileWorker::new(1, 3, 128));

    let db = CacheDB::new(EmptyDB::new());
    let mut evm = revm::Evm::builder()
        .with_db(db)
        .with_external_context(ext_worker.clone())
        .append_handler_register(register_handler)
        .build();

    let fibonacci_address = address!("0000000000000000000000000000000000001234");

    let fib_bytecode = Bytecode::new_raw(FIBONACCI_CODE.into());
    let fib_hash = fib_bytecode.hash_slow();

    evm.db_mut().insert_account_info(
        fibonacci_address,
        AccountInfo {
            code_hash: fib_hash,
            code: Some(Bytecode::new_raw(FIBONACCI_CODE.into())),
            ..Default::default()
        },
    );

    let access_list = AccessList(vec![AccessListItem {
        address: fibonacci_address,
        storage_keys: vec![B256::ZERO],
    }]);

    // First call - compiles ExternalFn
    evm.context.evm.env.tx.transact_to = TransactTo::Call(fibonacci_address);
    evm.context.evm.env.tx.data = U256::from(9).to_be_bytes_vec().into();
    evm.context.evm.inner.env.tx.access_list = access_list.to_vec();
    let mut result = evm.transact().unwrap();
    println!("fib(10) = {}", U256::from_be_slice(result.result.output().unwrap()));
    thread::sleep(std::time::Duration::from_secs(2));

    ext_worker.preload_cache(vec![B256::from(fib_hash)]).unwrap();

    // Second call - uses cached ExternalFn
    evm.context.evm.env.tx.transact_to = TransactTo::Call(fibonacci_address);
    evm.context.evm.env.tx.data = U256::from(9).to_be_bytes_vec().into();
    evm.context.evm.inner.env.tx.access_list = access_list.to_vec();

    result = evm.transact().unwrap();
    println!("fib(10) = {}", U256::from_be_slice(result.result.output().unwrap()));
}

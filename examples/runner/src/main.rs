use revm::{
    db::{CacheDB, EmptyDB},
    primitives::{address, hex, AccountInfo, Bytecode, TransactTo, U256},
};
use revmc_examples_runner::build_evm;

include!("./common.rs");

fn main() {
    let num =
        std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or_else(|| U256::from(100));
    // The bytecode runs fib(input + 1), so we need to subtract 1.
    let actual_num = num.saturating_sub(U256::from(1));

    let db = CacheDB::new(EmptyDB::new());
    let mut evm = build_evm(db);
    let fibonacci_address = address!("0000000000000000000000000000000000001234");
    evm.db_mut().insert_account_info(
        fibonacci_address,
        AccountInfo {
            code_hash: FIBONACCI_HASH.into(),
            code: Some(Bytecode::new_raw(FIBONACCI_CODE.into())),
            ..Default::default()
        },
    );
    evm.context.evm.env.tx.transact_to = TransactTo::Call(fibonacci_address);
    evm.context.evm.env.tx.data = actual_num.to_be_bytes_vec().into();
    let result = evm.transact().unwrap();
    // eprintln!("{:#?}", result.result);

    println!("fib({num}) = {}", U256::from_be_slice(result.result.output().unwrap()));
}

use crate::Bench;
use revm_bytecode::Bytecode;
use revm_context::{BlockEnv, CfgEnv, Context, Journal, TxEnv};
use revm_context_interface::result::ResultAndState;
use revm_database::{CacheDB, EmptyDB};
use revm_handler::{ExecuteEvm, MainBuilder, MainnetEvm};
use revm_primitives::{Address, B256, B256Map, Bytes, StorageKeyMap, StorageValue, TxKind, U256};
use revm_state::AccountInfo;
use revmc::{EvmCompiler, EvmLlvmBackend, JitEvm, RawEvmCompilerFn, primitives::hardfork::SpecId};
use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};

// ── Types ────────────────────────────────────────────────────────────────────

/// Simple owned-DB JIT EVM. Wraps a `MainnetEvm` with an owned `CacheDB` so
/// there are no lifetime issues. Suitable for benchmarks and simple runners.
pub type SimpleJitEvm = JitEvm<MainnetEvm<revm_handler::MainnetContext<CacheDB<EmptyDB>>>>;

// ── Fixture Serde ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct FixtureFile {
    #[serde(flatten)]
    cases: BTreeMap<String, FixtureCase>,
}

#[derive(Deserialize)]
struct FixtureCase {
    env: FixtureEnv,
    pre: BTreeMap<String, RawAccount>,
    transaction: Vec<RawTransaction>,
}

#[derive(Deserialize)]
struct FixtureEnv {
    #[serde(rename = "currentBaseFee")]
    current_base_fee: Option<String>,
    #[serde(rename = "currentCoinbase")]
    current_coinbase: Option<String>,
    #[serde(rename = "currentGasLimit")]
    current_gas_limit: Option<String>,
    #[serde(rename = "currentNumber")]
    current_number: Option<String>,
    #[serde(rename = "currentTimestamp")]
    current_timestamp: Option<String>,
    #[serde(rename = "currentRandom")]
    current_random: Option<String>,
}

#[derive(Deserialize)]
struct RawAccount {
    balance: String,
    code: String,
    nonce: String,
    #[serde(default)]
    storage: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct RawTransaction {
    data: String,
    #[serde(rename = "gasLimit")]
    gas_limit: String,
    #[serde(default, rename = "gasPrice")]
    gas_price: Option<String>,
    nonce: Option<String>,
    #[allow(dead_code)]
    sender: Option<String>,
    #[serde(rename = "secretKey")]
    #[allow(dead_code)]
    secret_key: Option<String>,
    to: Option<String>,
    #[serde(default)]
    value: Option<String>,
}

// ── Parsed fixture state ─────────────────────────────────────────────────────

/// Pre-parsed fixture state; cheap to clone for building a fresh DB per run.
#[derive(Clone)]
struct ParsedAccount {
    address: Address,
    balance: U256,
    nonce: u64,
    bytecode: Bytecode,
    code_hash: B256,
    storage: StorageKeyMap<StorageValue>,
}

/// A prepared benchmark, ready to run via interpreter or JIT.
///
/// Handles both fixture-based (multi-contract transaction) and simple bytecode
/// benchmarks — the latter are converted to a single-contract fixture
/// internally so the execution path is the same.
#[allow(missing_debug_implementations)]
pub struct PreparedBench {
    name: &'static str,
    runnable: bool,
    accounts: Vec<ParsedAccount>,
    block: BlockEnv,
    cfg: CfgEnv,
    tx: TxEnv,
    functions: B256Map<RawEvmCompilerFn>,
}

/// Caller address used for synthetic bytecode benchmarks.
const BENCH_CALLER: Address = Address::new([
    0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
    0x11, 0x11, 0x11, 0x11,
]);

/// Contract address used for synthetic bytecode benchmarks.
const BENCH_CONTRACT: Address = Address::new([
    0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
    0xcc, 0xcc, 0xcc, 0xcc,
]);

impl PreparedBench {
    /// Load and JIT-compile a benchmark using a fresh compiler.
    pub fn load(bench: &Bench, default_spec_id: SpecId) -> (Self, EvmCompiler<EvmLlvmBackend>) {
        let mut compiler = EvmCompiler::new_llvm(false).expect("LLVM backend");
        let prepared = Self::load_with(bench, default_spec_id, &mut compiler);
        (prepared, compiler)
    }

    /// Load and JIT-compile a benchmark, reusing an existing compiler.
    ///
    /// The caller must keep `compiler` alive as long as the returned `PreparedBench` is used,
    /// since the JIT'd function pointers live in the compiler's code memory.
    /// Call [`EvmCompiler::clear_ir`] between invocations to free IR while retaining code.
    pub fn load_with(
        bench: &Bench,
        default_spec_id: SpecId,
        compiler: &mut EvmCompiler<EvmLlvmBackend>,
    ) -> Self {
        let (accounts, block, cfg, tx) = if bench.is_fixture() {
            Self::parse_fixture(bench)
        } else {
            Self::from_bytecode(bench, default_spec_id)
        };

        // JIT compile all contract bytecodes.
        let spec_id = cfg.spec;
        let mut seen = HashSet::new();
        let mut pending = Vec::new();
        for acct in &accounts {
            if acct.bytecode.is_empty() || !seen.insert(acct.code_hash) {
                continue;
            }
            let name = format!("contract_{}", revmc::primitives::hex::encode(acct.code_hash));
            let func_id = compiler
                .translate(&name, acct.bytecode.original_byte_slice(), spec_id)
                .expect("translation failed");
            pending.push((acct.code_hash, func_id));
        }
        let mut functions = B256Map::default();
        for (hash, func_id) in pending {
            let fn_ptr = unsafe { compiler.jit_function(func_id).expect("JIT failed") };
            functions.insert(hash, fn_ptr.into_inner());
        }
        compiler.clear_ir().expect("clear_ir failed");

        let runnable = bench.stack_input.is_empty();
        Self { name: bench.name, runnable, accounts, block, cfg, tx, functions }
    }

    /// Whether this benchmark can be run as a transaction.
    ///
    /// Benchmarks with `stack_input` push values onto the stack before execution,
    /// which is not possible via `transact()`. They can still be compiled and
    /// benchmarked at the bytecode level.
    pub fn is_runnable(&self) -> bool {
        self.runnable
    }

    /// Convert a bytecode [`Bench`] into fixture state.
    fn from_bytecode(
        bench: &Bench,
        spec_id: SpecId,
    ) -> (Vec<ParsedAccount>, BlockEnv, CfgEnv, TxEnv) {
        let bytecode = Bytecode::new_raw(Bytes::copy_from_slice(&bench.bytecode));
        let code_hash = bytecode.hash_slow();

        let storage: StorageKeyMap<StorageValue> =
            bench.storage.iter().map(|&(k, v)| (k, v)).collect();

        let contract = ParsedAccount {
            address: BENCH_CONTRACT,
            balance: U256::ZERO,
            nonce: 1,
            bytecode,
            code_hash,
            storage,
        };
        let caller = ParsedAccount {
            address: BENCH_CALLER,
            balance: U256::MAX / U256::from(2),
            nonce: 0,
            bytecode: Bytecode::default(),
            code_hash: B256::ZERO,
            storage: Default::default(),
        };

        let gas_limit = u64::MAX / 2;
        let mut block = BlockEnv {
            number: bench.block_number.unwrap_or(U256::from(1)),
            timestamp: bench.timestamp.unwrap_or(U256::from(1)),
            gas_limit,
            basefee: 0,
            ..Default::default()
        };
        block.set_blob_excess_gas_and_price(
            0,
            revm_primitives::eip4844::BLOB_BASE_FEE_UPDATE_FRACTION_CANCUN,
        );

        let mut cfg = CfgEnv::new_with_spec(spec_id);
        cfg.tx_gas_limit_cap = Some(gas_limit);
        cfg.disable_nonce_check = true;
        let tx = TxEnv {
            tx_type: 0,
            caller: BENCH_CALLER,
            gas_limit,
            gas_price: 0,
            kind: TxKind::Call(BENCH_CONTRACT),
            value: U256::ZERO,
            data: Bytes::from(bench.calldata.clone()),
            nonce: 0,
            chain_id: Some(cfg.chain_id),
            access_list: Default::default(),
            gas_priority_fee: None,
            blob_hashes: Vec::new(),
            max_fee_per_blob_gas: 0,
            authorization_list: Vec::new(),
        };

        (vec![caller, contract], block, cfg, tx)
    }

    /// Parse a fixture JSON into accounts, block, cfg, tx.
    fn parse_fixture(bench: &Bench) -> (Vec<ParsedAccount>, BlockEnv, CfgEnv, TxEnv) {
        let fixture_json = bench.fixture_json.expect("fixture_json required for fixture bench");
        let spec_id = bench.spec_id.expect("spec_id required for fixture bench");
        let file: FixtureFile =
            serde_json::from_str(fixture_json).expect("failed to parse fixture JSON");
        let case = file.cases.into_values().next().expect("no cases in fixture");
        let first_tx = case.transaction.into_iter().next().expect("no transactions");

        // Parse accounts.
        let mut accounts = Vec::new();
        for (addr_hex, raw) in &case.pre {
            let address = parse_address(addr_hex);
            let balance = parse_u256(&raw.balance);
            let nonce = parse_u64(&raw.nonce);
            let bytecode_bytes = parse_hex_bytes(&raw.code);
            let bytecode = Bytecode::new_raw(Bytes::from(bytecode_bytes));
            let code_hash = bytecode.hash_slow();
            let storage: StorageKeyMap<StorageValue> =
                raw.storage.iter().map(|(k, v)| (parse_u256(k), parse_u256(v))).collect();
            accounts.push(ParsedAccount { address, balance, nonce, bytecode, code_hash, storage });
        }

        // Build block env.
        let env = &case.env;
        let mut block = BlockEnv {
            number: parse_u256(env.current_number.as_deref().unwrap_or("0x1")),
            timestamp: parse_u256(env.current_timestamp.as_deref().unwrap_or("0x1")),
            gas_limit: parse_u64(env.current_gas_limit.as_deref().unwrap_or("0x1000000")),
            basefee: parse_u64(env.current_base_fee.as_deref().unwrap_or("0x1")),
            beneficiary: parse_address(
                env.current_coinbase
                    .as_deref()
                    .unwrap_or("0x0000000000000000000000000000000000000000"),
            ),
            ..Default::default()
        };
        if let Some(random) = &env.current_random {
            block.prevrandao = Some(B256::from_slice(&parse_fixed_bytes(random, 32)));
        }
        block.set_blob_excess_gas_and_price(
            0,
            revm_primitives::eip4844::BLOB_BASE_FEE_UPDATE_FRACTION_CANCUN,
        );

        // Build tx.
        let caller = first_tx.sender.as_deref().map(parse_address).unwrap_or(BENCH_CALLER);
        let cfg = CfgEnv::new_with_spec(spec_id);
        let tx = TxEnv {
            tx_type: 0,
            caller,
            gas_limit: parse_u64(&first_tx.gas_limit),
            gas_price: parse_u128(first_tx.gas_price.as_deref().unwrap_or("0x0")),
            kind: match first_tx.to.as_deref() {
                Some(v) if !v.trim().is_empty() && v.trim() != "0x" => {
                    TxKind::Call(parse_address(v))
                }
                _ => TxKind::Create,
            },
            value: parse_u256(first_tx.value.as_deref().unwrap_or("0x0")),
            data: Bytes::from(parse_hex_bytes(&first_tx.data)),
            nonce: parse_u64(first_tx.nonce.as_deref().unwrap_or("0x0")),
            chain_id: Some(cfg.chain_id),
            access_list: Default::default(),
            gas_priority_fee: None,
            blob_hashes: Vec::new(),
            max_fee_per_blob_gas: 0,
            authorization_list: Vec::new(),
        };

        (accounts, block, cfg, tx)
    }

    /// Benchmark name.
    pub fn name(&self) -> &str {
        self.name
    }

    /// Transaction environment.
    pub fn tx(&self) -> &TxEnv {
        &self.tx
    }

    /// Build a fresh `CacheDB` from the parsed prestate.
    fn fresh_db(&self) -> CacheDB<EmptyDB> {
        let mut db = CacheDB::new(EmptyDB::new());
        for acct in &self.accounts {
            db.insert_account_info(
                acct.address,
                AccountInfo {
                    balance: acct.balance,
                    nonce: acct.nonce,
                    code_hash: acct.code_hash,
                    code: Some(acct.bytecode.clone()),
                    account_id: None,
                },
            );
            if !acct.storage.is_empty() {
                db.replace_account_storage(acct.address, acct.storage.clone()).unwrap();
            }
        }
        db
    }

    /// Build a fresh EVM with the given JIT functions.
    fn new_evm(&self, functions: B256Map<RawEvmCompilerFn>) -> SimpleJitEvm {
        let ctx = Context::<BlockEnv, TxEnv, CfgEnv, _, Journal<_>, ()>::new(
            self.fresh_db(),
            self.cfg.spec,
        );
        let mut inner = ctx.build_mainnet();
        inner.ctx.block = self.block.clone();
        inner.ctx.cfg = self.cfg.clone();
        JitEvm::new(inner, functions)
    }

    /// Build a fresh interpreter EVM (no JIT functions).
    pub fn new_interpreter_evm(&self) -> SimpleJitEvm {
        self.new_evm(B256Map::default())
    }

    /// Run via the plain interpreter.
    pub fn run_interpreter(&self) -> ResultAndState {
        let mut evm = self.new_interpreter_evm();
        evm.transact(self.tx.clone()).expect("interpreter execution failed")
    }

    /// Build a fresh JIT EVM with compiled functions.
    pub fn new_jit_evm(&self) -> SimpleJitEvm {
        self.new_evm(self.functions.clone())
    }

    /// Run via the JIT-compiled handler.
    pub fn run_jit(&self) -> ResultAndState {
        let mut evm = self.new_jit_evm();
        evm.transact(self.tx.clone()).expect("JIT execution failed")
    }

    /// Sanity-check that interpreter and JIT produce matching results.
    ///
    /// Panics if `!self.is_runnable()`.
    pub fn sanity_check(&self) {
        assert!(self.runnable, "cannot sanity-check a non-runnable benchmark");
        let interp = self.run_interpreter();
        let jit = self.run_jit();
        assert_eq!(
            interp.result.is_success(),
            jit.result.is_success(),
            "interpreter and JIT results differ:\n  interpreter: {:?}\n  JIT: {:?}",
            interp.result,
            jit.result,
        );
    }
}

// ── Hex parsing helpers ──────────────────────────────────────────────────────

fn parse_address(value: &str) -> Address {
    Address::from_slice(&parse_fixed_bytes(value, 20))
}

fn parse_u256(value: &str) -> U256 {
    let trimmed = strip_0x(value.trim());
    if trimmed.is_empty() { U256::ZERO } else { U256::from_str_radix(trimmed, 16).unwrap() }
}

fn parse_u128(value: &str) -> u128 {
    let trimmed = strip_0x(value.trim());
    if trimmed.is_empty() { 0 } else { u128::from_str_radix(trimmed, 16).unwrap() }
}

fn parse_u64(value: &str) -> u64 {
    let trimmed = strip_0x(value.trim());
    if trimmed.is_empty() { 0 } else { u64::from_str_radix(trimmed, 16).unwrap() }
}

fn parse_hex_bytes(value: &str) -> Vec<u8> {
    let trimmed = strip_0x(value.trim());
    if trimmed.is_empty() {
        return Vec::new();
    }
    let even =
        if trimmed.len().is_multiple_of(2) { trimmed.to_owned() } else { format!("0{trimmed}") };
    revmc::primitives::hex::decode(even).unwrap()
}

fn parse_fixed_bytes(value: &str, expected_len: usize) -> Vec<u8> {
    let mut bytes = parse_hex_bytes(value);
    if bytes.len() < expected_len {
        let mut padded = vec![0u8; expected_len - bytes.len()];
        padded.extend_from_slice(&bytes);
        bytes = padded;
    }
    bytes
}

fn strip_0x(value: &str) -> &str {
    value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")).unwrap_or(value)
}

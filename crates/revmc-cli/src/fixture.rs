use revm::{
    ExecuteEvm, MainnetEvm,
    bytecode::Bytecode,
    context::{BlockEnv, CfgEnv, TxEnv},
    context_interface::{
        ContextSetters,
        result::{EVMError, HaltReason, InvalidTransaction, ResultAndState},
    },
    database::{CacheDB, EmptyDB},
    handler::{EvmTr, FrameResult, Handler, ItemOrResult, MainBuilder},
    primitives::{Address, B256, Bytes, StorageKeyMap, StorageValue, TxKind, U256},
    state::AccountInfo,
};
use revmc::{EvmCompiler, EvmCompilerFn, EvmLlvmBackend, OptimizationLevel, RawEvmCompilerFn};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use crate::BenchDef;

// ── Types ────────────────────────────────────────────────────────────────────

type BenchEvm<'a> = MainnetEvm<revm::handler::MainnetContext<&'a mut CacheDB<EmptyDB>>>;
type BenchError = EVMError<core::convert::Infallible, InvalidTransaction>;

// ── JIT Handler ──────────────────────────────────────────────────────────────

struct JitHandler {
    functions: HashMap<B256, RawEvmCompilerFn>,
}

impl Handler for JitHandler {
    type Evm = BenchEvm<'static>;
    type Error = BenchError;
    type HaltReason = HaltReason;

    fn run_exec_loop(
        &mut self,
        evm: &mut Self::Evm,
        first_frame_input: revm::interpreter::interpreter_action::FrameInit,
    ) -> Result<FrameResult, Self::Error> {
        let res = evm.frame_init(first_frame_input)?;
        if let ItemOrResult::Result(frame_result) = res {
            return Ok(frame_result);
        }
        loop {
            let call_or_result = {
                let frame = evm.frame_stack.get();
                let bytecode_hash = frame.interpreter.bytecode.get_or_calculate_hash();
                if let Some(&raw_fn) = self.functions.get(&bytecode_hash) {
                    let ctx = &mut evm.ctx;
                    let f = EvmCompilerFn::new(raw_fn);
                    let action = unsafe { f.call_with_interpreter(&mut frame.interpreter, ctx) };
                    frame.process_next_action::<_, BenchError>(ctx, action).inspect(|i| {
                        if i.is_result() {
                            frame.set_finished(true);
                        }
                    })?
                } else {
                    evm.frame_run()?
                }
            };
            let result = match call_or_result {
                ItemOrResult::Item(init) => match evm.frame_init(init)? {
                    ItemOrResult::Item(_) => continue,
                    ItemOrResult::Result(result) => result,
                },
                ItemOrResult::Result(result) => result,
            };
            if let Some(result) = evm.frame_return_result(result)? {
                return Ok(result);
            }
        }
    }
}

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

/// A prepared transaction-fixture benchmark, ready to run.
#[allow(missing_debug_implementations)]
pub struct PreparedFixtureBench {
    accounts: Vec<ParsedAccount>,
    block: BlockEnv,
    cfg: CfgEnv,
    tx: TxEnv,
    functions: HashMap<B256, RawEvmCompilerFn>,
    // Prevent drop of compiler while JIT functions are alive.
    _compiler: Box<EvmCompiler<EvmLlvmBackend>>,
}

impl PreparedFixtureBench {
    /// Load and JIT-compile a fixture benchmark.
    pub fn load(def: &BenchDef) -> Self {
        let fixture_json = def.fixture_json.expect("fixture_json required for fixture bench");
        let spec_id = def.spec_id.expect("spec_id required for fixture bench");
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
            revm::primitives::eip4844::BLOB_BASE_FEE_UPDATE_FRACTION_CANCUN,
        );

        // Build tx (use sender from fixture, or derive from secret key).
        let caller = first_tx
            .sender
            .as_deref()
            .map(parse_address)
            .unwrap_or_else(|| parse_address("0x89d5e72a8a4a0330a65bbcef3032be2f728264a8"));
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

        // JIT compile all contract bytecodes.
        let backend =
            EvmLlvmBackend::new(false, OptimizationLevel::Aggressive).expect("LLVM backend");
        let mut compiler = Box::new(EvmCompiler::new(backend));
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
        let mut functions = HashMap::new();
        for (hash, func_id) in pending {
            let fn_ptr = unsafe { compiler.jit_function(func_id).expect("JIT failed") };
            functions.insert(hash, fn_ptr.into_inner());
        }

        Self { accounts, block, cfg, tx, functions, _compiler: compiler }
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

    /// Run via the plain interpreter.
    pub fn run_interpreter(&self) -> ResultAndState {
        let mut db = self.fresh_db();
        let ctx = revm::context::Context::<BlockEnv, TxEnv, CfgEnv, _, revm::context::Journal<_>, ()>::new(&mut db, self.cfg.spec);
        let mut evm = ctx.build_mainnet();
        evm.ctx.block = self.block.clone();
        evm.ctx.cfg = self.cfg.clone();
        evm.transact(self.tx.clone()).expect("interpreter execution failed")
    }

    /// Run via the JIT-compiled handler.
    pub fn run_jit(&self) -> ResultAndState {
        let db = Arc::new(self.fresh_db());
        // SAFETY: The `JitHandler` requires `BenchEvm<'static>` due to the `Handler` trait
        // associated types. The `Arc` keeps the DB alive for the duration of the call, and
        // the mutable reference is not aliased since we hold the only `Arc`.
        let db_ref = unsafe { &mut *(Arc::as_ptr(&db) as *mut CacheDB<EmptyDB>) };
        let ctx = revm::context::Context::<BlockEnv, TxEnv, CfgEnv, _, revm::context::Journal<_>, ()>::new(db_ref, self.cfg.spec);
        let mut evm = ctx.build_mainnet();
        evm.ctx.block = self.block.clone();
        evm.ctx.cfg = self.cfg.clone();
        evm.ctx.set_tx(self.tx.clone());
        let mut handler = JitHandler { functions: self.functions.clone() };
        let result = handler.run(&mut evm).expect("JIT execution failed");
        let state = evm.ctx.journaled_state.finalize();
        ResultAndState::new(result, state)
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

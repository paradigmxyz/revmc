#![allow(missing_docs)]
//! Curve StableSwap benchmark: plain interpreter vs JIT.
//!
//! This is the key A/B test for the DIV/MOD ruint builtin optimization.
//! Curve's `exchange()` calls `get_y()` which runs a Newton iteration with
//! heavy uint256 division. Before the fix, JIT was 43% SLOWER than the
//! interpreter on this workload due to LLVM's poor i256 udiv codegen.
//!
//! Run: `cargo bench -p revmc-cli --bench bench_curve`

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use revm::{
    bytecode::Bytecode,
    context::{BlockEnv, CfgEnv, TxEnv},
    context_interface::{
        result::{EVMError, HaltReason, InvalidTransaction, ResultAndState},
        ContextSetters,
    },
    database::{CacheDB, EmptyDB},
    handler::{EvmTr, FrameResult, Handler, ItemOrResult, MainBuilder},
    primitives::{
        hardfork::SpecId, Address, Bytes, StorageKeyMap, StorageValue, TxKind,
        B256, U256,
    },
    state::AccountInfo,
    ExecuteEvm, MainnetEvm,
};
use revmc::{EvmCompiler, EvmLlvmBackend, OptimizationLevel};
use revmc_context::{EvmCompilerFn, RawEvmCompilerFn};
use serde::Deserialize;

// ── Types ────────────────────────────────────────────────────────────────────

type BenchEvm<'a> = MainnetEvm<revm::handler::MainnetContext<&'a mut CacheDB<EmptyDB>>>;
type BenchError = EVMError<core::convert::Infallible, InvalidTransaction>;

// ── JIT Handler ──────────────────────────────────────────────────────────────

struct JitHandler {
    functions: Arc<HashMap<B256, RawEvmCompilerFn>>,
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

// ── Fixture ──────────────────────────────────────────────────────────────────

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
    #[serde(rename = "secretKey")]
    #[allow(dead_code)]
    secret_key: String,
    to: Option<String>,
    #[serde(default)]
    value: Option<String>,
}

// ── Setup ────────────────────────────────────────────────────────────────────

struct CurveBench {
    db: Arc<CacheDB<EmptyDB>>,
    block: BlockEnv,
    cfg: CfgEnv,
    tx: TxEnv,
    functions: Arc<HashMap<B256, RawEvmCompilerFn>>,
    // Prevent drop of compiler/context while JIT functions are alive.
    _compiler: &'static mut EvmCompiler<EvmLlvmBackend<'static>>,
    _context: &'static revmc::llvm::inkwell::context::Context,
}

impl CurveBench {
    fn load() -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../data/curve-stableswap-2pool.json");
        let json = fs::read_to_string(&path).expect("failed to read fixture");
        let file: FixtureFile = serde_json::from_str(&json).expect("failed to parse fixture");
        let case = file.cases.into_values().next().expect("no cases in fixture");
        let first_tx = case.transaction.into_iter().next().expect("no transactions");

        // Build DB
        let mut db = CacheDB::new(EmptyDB::new());
        let mut code_hashes = Vec::new();
        for (addr_hex, raw) in &case.pre {
            let address = parse_address(addr_hex);
            let balance = parse_u256(&raw.balance);
            let nonce = parse_u64(&raw.nonce);
            let bytecode_bytes = parse_hex_bytes(&raw.code);
            let bytecode = Bytecode::new_raw(Bytes::from(bytecode_bytes));
            let code_hash = bytecode.hash_slow();
            let storage: StorageKeyMap<StorageValue> =
                raw.storage.iter().map(|(k, v)| (parse_u256(k), parse_u256(v))).collect();
            db.insert_account_info(
                address,
                AccountInfo {
                    balance,
                    nonce,
                    code_hash,
                    code: Some(bytecode.clone()),
                    account_id: None,
                },
            );
            if !storage.is_empty() {
                db.replace_account_storage(address, storage).unwrap();
            }
            if !bytecode.is_empty() {
                code_hashes.push((code_hash, bytecode));
            }
        }

        // Build block env
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

        // Build tx (hardcoded sender — derived from the fixture's secret key)
        let caller = parse_address("0x89d5e72a8a4a0330a65bbcef3032be2f728264a8");
        let cfg = CfgEnv::new_with_spec(SpecId::CANCUN);
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

        // JIT compile
        let context = Box::leak(Box::new(revmc::llvm::inkwell::context::Context::create()));
        let backend = EvmLlvmBackend::new(context, false, OptimizationLevel::Aggressive)
            .expect("LLVM backend");
        let compiler: &'static mut EvmCompiler<EvmLlvmBackend<'static>> =
            Box::leak(Box::new(EvmCompiler::new(backend)));
        let mut seen = HashSet::new();
        let mut pending = Vec::new();
        for (hash, bytecode) in &code_hashes {
            if !seen.insert(*hash) {
                continue;
            }
            let name = format!("contract_{}", hex::encode(hash.as_slice()));
            let func_id = compiler
                .translate(&name, bytecode.original_byte_slice(), SpecId::CANCUN)
                .expect("translation failed");
            pending.push((*hash, func_id));
        }
        let mut functions = HashMap::new();
        for (hash, func_id) in pending {
            let fn_ptr = unsafe { compiler.jit_function(func_id).expect("JIT failed") };
            functions.insert(hash, fn_ptr.into_inner());
        }

        Self {
            db: Arc::new(db),
            block,
            cfg,
            tx,
            functions: Arc::new(functions),
            _compiler: compiler,
            _context: context,
        }
    }

    fn run_plain(&self) -> ResultAndState {
        let db_ref = unsafe { &mut *(Arc::as_ptr(&self.db) as *mut CacheDB<EmptyDB>) };
        let ctx = revm::context::Context::<BlockEnv, TxEnv, CfgEnv, _, revm::context::Journal<_>, ()>::new(db_ref, SpecId::CANCUN);
        let mut evm = ctx.build_mainnet();
        evm.ctx.block = self.block.clone();
        evm.ctx.cfg = self.cfg.clone();
        evm.transact(self.tx.clone()).expect("plain execution failed")
    }

    fn run_jit(&self) -> ResultAndState {
        let db_ref = unsafe { &mut *(Arc::as_ptr(&self.db) as *mut CacheDB<EmptyDB>) };
        let ctx = revm::context::Context::<BlockEnv, TxEnv, CfgEnv, _, revm::context::Journal<_>, ()>::new(db_ref, SpecId::CANCUN);
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

// ── Benchmark ────────────────────────────────────────────────────────────────

/// Benchmark Curve StableSwap: plain interpreter vs JIT-compiled execution.
pub fn bench_curve_stableswap(c: &mut Criterion) {
    let bench = CurveBench::load();

    // Sanity check
    assert!(bench.run_plain().result.is_success(), "plain execution reverted");
    assert!(
        bench.run_jit().result.is_success(),
        "JIT execution reverted — check revmc/revm version compatibility"
    );

    let mut group = c.benchmark_group("curve_stableswap");
    group.bench_function("plain_execution", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();
                let result = bench.run_plain();
                total += start.elapsed();
                black_box(result);
            }
            total
        });
    });
    group.bench_function("jit_optimized", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();
                let result = bench.run_jit();
                total += start.elapsed();
                black_box(result);
            }
            total
        });
    });
    group.finish();
}

criterion_group!(benches, bench_curve_stableswap);
criterion_main!(benches);

// ── Hex parsing helpers ──────────────────────────────────────────────────────

fn parse_address(value: &str) -> Address {
    Address::from_slice(&parse_fixed_bytes(value, 20))
}

fn parse_u256(value: &str) -> U256 {
    let trimmed = strip_0x(value.trim());
    if trimmed.is_empty() {
        U256::ZERO
    } else {
        U256::from_str_radix(trimmed, 16).unwrap()
    }
}

fn parse_u128(value: &str) -> u128 {
    let trimmed = strip_0x(value.trim());
    if trimmed.is_empty() {
        0
    } else {
        u128::from_str_radix(trimmed, 16).unwrap()
    }
}

fn parse_u64(value: &str) -> u64 {
    let trimmed = strip_0x(value.trim());
    if trimmed.is_empty() {
        0
    } else {
        u64::from_str_radix(trimmed, 16).unwrap()
    }
}

fn parse_hex_bytes(value: &str) -> Vec<u8> {
    let trimmed = strip_0x(value.trim());
    if trimmed.is_empty() {
        return Vec::new();
    }
    let even =
        if trimmed.len().is_multiple_of(2) { trimmed.to_owned() } else { format!("0{trimmed}") };
    hex::decode(even).unwrap()
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

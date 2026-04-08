// revmc-specific code: compilation, handler integration, and test orchestration.

use crate::runner::{
    check_evm_execution, execute_test_suite, skip_test, TestError, TestErrorKind, TestRunnerState,
};
use dashmap::DashMap;
use revm_context::{block::BlockEnv, cfg::CfgEnv, tx::TxEnv, Context};
use revm_context_interface::result::{EVMError, HaltReason, InvalidTransaction};
use revm_database::{self as database, bal::EvmDatabaseError};
use revm_database_interface::{DatabaseCommit, EmptyDB};
use revm_handler::{
    EvmTr, FrameResult, Handler, ItemOrResult, MainBuilder, MainContext, MainnetEvm,
};
use revm_primitives::{hardfork::SpecId, keccak256, B256, U256};
use revm_statetest_types::{SpecName, TestSuite, TestUnit};
use revmc::{shared_library_path, EvmCompiler, EvmCompilerFn, EvmLlvmBackend, Linker};
use std::{
    cell::RefCell,
    collections::HashMap,
    mem::ManuallyDrop,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier, Mutex, OnceLock,
    },
    time::{Duration, Instant},
};
use thread_local::ThreadLocal;

// ── Compile mode ────────────────────────────────────────────────────────────

/// How to compile and execute bytecodes in the test suite.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompileMode {
    /// Standard interpreter execution (no compilation).
    #[default]
    Interpreter,
    /// JIT-compile all bytecodes before execution.
    Jit,
    /// AOT-compile all bytecodes to a shared library, then load and execute.
    Aot,
    /// Use the runtime backend: AOT-compile, load through `JitBackend`, and look up
    /// compiled functions via `JitBackend::lookup()`.
    Runtime,
}

// ── Compiled contracts ──────────────────────────────────────────────────────

/// Compiled contracts cache mapping bytecode hash to compiled function.
#[derive(Default)]
pub struct CompiledContracts {
    functions: HashMap<B256, EvmCompilerFn>,
}

impl CompiledContracts {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, code_hash: &B256) -> Option<EvmCompilerFn> {
        self.functions.get(code_hash).copied()
    }

    pub fn insert(&mut self, code_hash: B256, func: EvmCompilerFn) {
        self.functions.insert(code_hash, func);
    }
}

// ── Handler ─────────────────────────────────────────────────────────────────

type StateTestEvm = MainnetEvm<revm_handler::MainnetContext<database::State<EmptyDB>>>;
type StateTestError = EVMError<EvmDatabaseError<std::convert::Infallible>, InvalidTransaction>;

/// Custom handler that dispatches to compiled functions. All bytecodes —
/// including runtime-created ones (CREATE/CREATE2) — are JIT-compiled before
/// execution. Never falls back to the interpreter.
pub struct CompiledHandler<'a> {
    pub compiled: &'a CompiledContracts,
    pub cache: &'a CompileCache,
    pub spec_id: SpecId,
}

impl Handler for CompiledHandler<'_> {
    type Evm = StateTestEvm;
    type Error = StateTestError;
    type HaltReason = HaltReason;

    fn run_exec_loop(
        &mut self,
        evm: &mut Self::Evm,
        first_frame_input: revm_interpreter::interpreter_action::FrameInit,
    ) -> Result<FrameResult, Self::Error> {
        let res = evm.frame_init(first_frame_input)?;
        if let ItemOrResult::Result(frame_result) = res {
            return Ok(frame_result);
        }
        loop {
            let call_or_result = {
                let frame = evm.frame_stack.get();
                let bytecode_hash = frame.interpreter.bytecode.get_or_calculate_hash();
                let f = if let Some(f) = self.compiled.get(&bytecode_hash) {
                    f
                } else {
                    // Runtime-created contract (CREATE/CREATE2); compile it.
                    let code = frame.interpreter.bytecode.original_byte_slice();
                    self.cache
                        .compile_single(bytecode_hash, code, self.spec_id)
                        .expect("compilation failed for runtime bytecode")
                };
                {
                    let ctx = &mut evm.ctx;
                    let action = unsafe { f.call_with_interpreter(&mut frame.interpreter, ctx) };
                    frame.process_next_action::<_, StateTestError>(ctx, action).inspect(|i| {
                        if i.is_result() {
                            frame.set_finished(true);
                        }
                    })?
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

// ── Compilation cache ────────────────────────────────────────────────────────

type ClaimedEntry<'a> = (B256, &'a [u8], String, Arc<OnceLock<EvmCompilerFn>>);

/// Thread-safe compilation cache shared across workers.
pub struct CompileCache {
    mode: CompileMode,
    functions: DashMap<(B256, SpecId), Arc<OnceLock<EvmCompilerFn>>>,
    /// Keep AOT shared libraries alive. Unused for JIT mode.
    libs: Mutex<Vec<(tempfile::TempDir, libloading::Library)>>,
    /// `ManuallyDrop` because `ThreadLocal::drop` drops values on the calling thread, but each
    /// `EvmLlvmBackend` holds a reference to a thread-local LLVM context that only lives on the
    /// thread that created it. Dropping on a different thread would use-after-free.
    compiler: ManuallyDrop<ThreadLocal<RefCell<EvmCompiler<EvmLlvmBackend>>>>,
    n_hits: AtomicUsize,
    n_misses: AtomicUsize,
}

impl CompileCache {
    pub fn new(mode: CompileMode) -> Self {
        Self {
            mode,
            functions: Default::default(),
            libs: Default::default(),
            compiler: Default::default(),
            n_hits: Default::default(),
            n_misses: Default::default(),
        }
    }

    /// Partition a test unit's contracts into cached (already compiled) and
    /// claimed (this thread must compile them). Uses `DashMap::entry()` to
    /// atomically distinguish vacant (we compile) from occupied (we wait).
    fn claim_missing<'a>(
        &self,
        unit: &'a TestUnit,
        spec_id: SpecId,
    ) -> (CompiledContracts, Vec<ClaimedEntry<'a>>) {
        use dashmap::mapref::entry::Entry;

        let mut compiled = CompiledContracts::new();
        let mut claimed = Vec::new();

        for info in unit.pre.values() {
            if info.code.is_empty() {
                continue;
            }
            let code_hash = keccak256(&info.code);
            if compiled.get(&code_hash).is_some() {
                continue;
            }
            if claimed.iter().any(|(h, _, _, _): &(B256, _, _, _)| h == &code_hash) {
                continue;
            }

            match self.functions.entry((code_hash, spec_id)) {
                Entry::Occupied(e) => {
                    let lock = e.get().clone();
                    drop(e);
                    // Already compiled or being compiled by another thread.
                    if let Some(f) = lock.get() {
                        self.n_hits.fetch_add(1, Ordering::Relaxed);
                        compiled.insert(code_hash, *f);
                    }
                    // Otherwise: another thread is compiling it, wait_for_all handles it.
                }
                Entry::Vacant(e) => {
                    // We're first — claim it.
                    let lock = Arc::new(OnceLock::new());
                    e.insert(lock.clone());
                    self.n_misses.fetch_add(1, Ordering::Relaxed);
                    claimed.push((
                        code_hash,
                        &info.code[..],
                        format!("contract_{code_hash:x}_{spec_id}"),
                        lock,
                    ));
                }
            }
        }

        (compiled, claimed)
    }

    /// Wait for all contracts in a test unit to be compiled, returning the
    /// fully populated `CompiledContracts`.
    fn wait_for_all(&self, unit: &TestUnit, spec_id: SpecId) -> CompiledContracts {
        let mut compiled = CompiledContracts::new();
        for info in unit.pre.values() {
            if info.code.is_empty() {
                continue;
            }
            let code_hash = keccak256(&info.code);
            if compiled.get(&code_hash).is_some() {
                continue;
            }
            if let Some(entry) = self.functions.get(&(code_hash, spec_id)) {
                let f = entry.value().wait();
                compiled.insert(code_hash, *f);
            }
        }
        compiled
    }

    /// Compile all contracts in a test unit, returning the compiled functions.
    /// Uses the cache's mode (JIT or AOT) to determine compilation strategy.
    pub fn compile(
        &self,
        unit: &TestUnit,
        spec_id: SpecId,
    ) -> Result<CompiledContracts, TestErrorKind> {
        let (mut compiled, claimed) = self.claim_missing(unit, spec_id);
        if claimed.is_empty() {
            // Still need to wait for contracts another thread claimed.
            let rest = self.wait_for_all(unit, spec_id);
            for (hash, f) in rest.functions {
                compiled.functions.entry(hash).or_insert(f);
            }
            return Ok(compiled);
        }

        match self.mode {
            CompileMode::Jit => self.compile_jit_batch(&claimed, &mut compiled, spec_id)?,
            CompileMode::Aot => self.compile_aot_batch(&claimed, &mut compiled, spec_id)?,
            CompileMode::Runtime | CompileMode::Interpreter => unreachable!(),
        }

        // Wait for contracts claimed by other threads.
        let rest = self.wait_for_all(unit, spec_id);
        for (hash, f) in rest.functions {
            compiled.functions.entry(hash).or_insert(f);
        }

        Ok(compiled)
    }

    /// Compile a single bytecode (e.g. from CREATE/CREATE2 at runtime).
    /// Claims the cache slot first, then compiles via the batch path.
    fn compile_single(
        &self,
        code_hash: B256,
        code: &[u8],
        spec_id: SpecId,
    ) -> Result<EvmCompilerFn, TestErrorKind> {
        use dashmap::mapref::entry::Entry;

        match self.functions.entry((code_hash, spec_id)) {
            Entry::Occupied(e) => {
                let lock = e.get().clone();
                drop(e);
                let f = lock.wait();
                self.n_hits.fetch_add(1, Ordering::Relaxed);
                Ok(*f)
            }
            Entry::Vacant(e) => {
                let lock = Arc::new(OnceLock::new());
                e.insert(lock.clone());
                self.n_misses.fetch_add(1, Ordering::Relaxed);

                let name = format!("runtime_{code_hash:x}_{spec_id}");
                let claimed = vec![(code_hash, code, name, lock)];
                let mut compiled = CompiledContracts::new();

                match self.mode {
                    CompileMode::Jit => self.compile_jit_batch(&claimed, &mut compiled, spec_id)?,
                    CompileMode::Aot => self.compile_aot_batch(&claimed, &mut compiled, spec_id)?,
                    CompileMode::Runtime | CompileMode::Interpreter => unreachable!(),
                }

                Ok(compiled.get(&code_hash).unwrap())
            }
        }
    }

    fn compile_jit_batch(
        &self,
        claimed: &[ClaimedEntry<'_>],
        compiled: &mut CompiledContracts,
        spec_id: SpecId,
    ) -> Result<(), TestErrorKind> {
        let mut compiler = self.compiler.get_or(|| make_compiler(false)).borrow_mut();

        let mut func_ids = Vec::new();
        for (code_hash, code, name, _) in claimed {
            let func_id = compiler
                .translate(name, *code, spec_id)
                .map_err(|e| TestErrorKind::CompilationError(format!("translate {name}: {e}")))?;
            func_ids.push((*code_hash, func_id));
        }

        for (i, (code_hash, func_id)) in func_ids.into_iter().enumerate() {
            let func = unsafe { compiler.jit_function(func_id) }.map_err(|e| {
                TestErrorKind::CompilationError(format!("jit {:x}: {e}", code_hash))
            })?;
            claimed[i].3.set(func).ok();
            compiled.insert(code_hash, func);
        }

        let _ = compiler.clear_ir();

        Ok(())
    }

    fn compile_aot_batch(
        &self,
        claimed: &[ClaimedEntry<'_>],
        compiled: &mut CompiledContracts,
        spec_id: SpecId,
    ) -> Result<(), TestErrorKind> {
        let mut compiler = self.compiler.get_or(|| make_compiler(true)).borrow_mut();

        let mut names: Vec<(B256, String)> = Vec::new();
        for (code_hash, code, name, _) in claimed {
            compiler
                .translate(name, *code, spec_id)
                .map_err(|e| TestErrorKind::CompilationError(format!("translate {name}: {e}")))?;
            names.push((*code_hash, name.clone()));
        }

        let tmp_dir = tempfile::tempdir()
            .map_err(|e| TestErrorKind::CompilationError(format!("tempdir: {e}")))?;
        let obj_path = tmp_dir.path().join("a.o");
        let shared_lib_path = shared_library_path(tmp_dir.path(), "a");

        compiler
            .write_object_to_file(&obj_path)
            .map_err(|e| TestErrorKind::CompilationError(format!("write object: {e}")))?;

        let linker = Linker::new();
        linker
            .link(&shared_lib_path, [obj_path.to_str().unwrap()])
            .map_err(|e| TestErrorKind::CompilationError(format!("link: {e}")))?;

        let lib = unsafe { libloading::Library::new(&shared_lib_path) }
            .map_err(|e| TestErrorKind::CompilationError(format!("load: {e}")))?;

        for (i, (code_hash, name)) in names.iter().enumerate() {
            let f: libloading::Symbol<'_, EvmCompilerFn> = unsafe { lib.get(name.as_bytes()) }
                .map_err(|e| TestErrorKind::CompilationError(format!("symbol {name}: {e}")))?;
            claimed[i].3.set(*f).ok();
            compiled.insert(*code_hash, *f);
        }

        self.libs.lock().unwrap().push((tmp_dir, lib));

        let _ = compiler.clear_ir();

        Ok(())
    }

    pub fn print_stats(&self) {
        let hits = self.n_hits.load(Ordering::Relaxed);
        let misses = self.n_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total > 0 {
            let label = match self.mode {
                CompileMode::Jit => "JIT",
                CompileMode::Aot => "AOT",
                CompileMode::Runtime | CompileMode::Interpreter => unreachable!(),
            };
            let rate = hits as f64 / total as f64 * 100.0;
            let n_libs = self.libs.lock().unwrap().len();
            if n_libs > 0 {
                println!(
                    "{label} cache: {total} lookups, {hits} hits, {misses} misses ({rate:.1}% hit rate), {} unique, {n_libs} shared libs",
                    self.functions.len()
                );
            } else {
                println!(
                    "{label} cache: {total} lookups, {hits} hits, {misses} misses ({rate:.1}% hit rate), {} unique",
                    self.functions.len()
                );
            }
        }
    }
}

fn make_compiler(aot: bool) -> RefCell<EvmCompiler<EvmLlvmBackend>> {
    RefCell::new(EvmCompiler::new(EvmLlvmBackend::new(aot).unwrap()))
}

// ── Compiled test execution ─────────────────────────────────────────────────

pub struct CompiledTestContext<'a> {
    pub compiled: &'a CompiledContracts,
    pub cache: &'a CompileCache,
    pub spec_id: SpecId,
    pub test: &'a revm_statetest_types::Test,
    pub unit: &'a TestUnit,
    pub name: &'a str,
    pub cfg: &'a CfgEnv,
    pub block: &'a BlockEnv,
    pub tx: &'a TxEnv,
    pub cache_state: &'a database::CacheState,
    pub elapsed: &'a Arc<Mutex<Duration>>,
}

/// Execute a single test using compiled functions via the custom handler.
pub fn execute_single_test_compiled(ctx: CompiledTestContext<'_>) -> Result<(), TestErrorKind> {
    let prestate = ctx.cache_state.clone();
    let state =
        database::State::builder().with_cached_prestate(prestate).with_bundle_update().build();

    let timer = Instant::now();
    let evm_context = Context::mainnet()
        .with_block(ctx.block.clone())
        .with_tx(ctx.tx.clone())
        .with_cfg(ctx.cfg.clone())
        .with_db(state);
    let mut handler =
        CompiledHandler { compiled: ctx.compiled, cache: ctx.cache, spec_id: ctx.spec_id };
    let mut evm = evm_context.build_mainnet();
    let exec_result = handler.run(&mut evm);
    if exec_result.is_ok() {
        let s = evm.ctx.journaled_state.finalize();
        DatabaseCommit::commit(&mut evm.ctx.journaled_state.database, s);
    }
    *ctx.elapsed.lock().unwrap() += timer.elapsed();

    check_evm_execution(
        ctx.test,
        ctx.unit.out.as_ref(),
        ctx.name,
        &exec_result,
        &mut evm.ctx.journaled_state.database,
        *ctx.cfg.spec(),
        false,
    )
}

// ── Suite-level execution (compiled) ─────────────────────────────────────────

/// Execute a single test suite file, compiling all contracts before execution.
fn execute_test_suite_compiled(
    path: &Path,
    elapsed: &Arc<Mutex<Duration>>,
    cache: &CompileCache,
) -> Result<(), TestError> {
    if skip_test(path) {
        return Ok(());
    }

    let s = std::fs::read_to_string(path).unwrap();
    let path_str = path.to_string_lossy().into_owned();
    let suite: TestSuite = serde_json::from_str(&s).map_err(|e| TestError {
        name: "Unknown".to_string(),
        path: path_str.clone(),
        kind: e.into(),
    })?;

    for (name, unit) in suite.0 {
        let cache_state = unit.state();

        let mut cfg = CfgEnv::default();
        cfg.chain_id = unit.env.current_chain_id.unwrap_or(U256::ONE).try_into().unwrap_or(1);

        for (spec_name, tests) in &unit.post {
            if *spec_name == SpecName::Constantinople {
                continue;
            }

            let spec_id = spec_name.to_spec_id();
            cfg.set_spec_and_mainnet_gas_params(spec_id);

            if cfg.spec().is_enabled_in(SpecId::OSAKA) {
                cfg.set_max_blobs_per_tx(6);
            } else if cfg.spec().is_enabled_in(SpecId::PRAGUE) {
                cfg.set_max_blobs_per_tx(9);
            } else {
                cfg.set_max_blobs_per_tx(6);
            }

            let block = unit.block_env(&mut cfg);

            let compiled = cache.compile(&unit, spec_id).map_err(|e| TestError {
                name: name.clone(),
                path: path_str.clone(),
                kind: e,
            })?;

            for test in tests.iter() {
                let tx = match test.tx_env(&unit) {
                    Ok(tx) => tx,
                    Err(_) if test.expect_exception.is_some() => continue,
                    Err(_) => {
                        return Err(TestError {
                            name,
                            path: path_str,
                            kind: TestErrorKind::UnknownPrivateKey(unit.transaction.secret_key),
                        });
                    }
                };

                let result = execute_single_test_compiled(CompiledTestContext {
                    compiled: &compiled,
                    cache,
                    spec_id,
                    test,
                    unit: &unit,
                    name: &name,
                    cfg: &cfg,
                    block: &block,
                    tx: &tx,
                    cache_state: &cache_state,
                    elapsed,
                });

                if let Err(e) = result {
                    return Err(TestError { name, path: path_str, kind: e });
                }
            }
        }
    }
    Ok(())
}

// ── Runtime backend mode ─────────────────────────────────────────────────────

use revmc::{
    revm_evm::JitEvm,
    runtime::{JitBackend, LookupRequest, RuntimeConfig, RuntimeTuning},
};

/// Execute a single test using the runtime backend via [`JitEvm`].
fn execute_single_test_runtime(
    backend: &JitBackend,
    ctx: RuntimeTestContext<'_>,
) -> Result<(), TestErrorKind> {
    let prestate = ctx.cache_state.clone();
    let state =
        database::State::builder().with_cached_prestate(prestate).with_bundle_update().build();

    let timer = Instant::now();
    let evm_context = Context::mainnet()
        .with_block(ctx.block.clone())
        .with_tx(ctx.tx.clone())
        .with_cfg(ctx.cfg.clone())
        .with_db(state);
    let inner = evm_context.build_mainnet();
    let mut evm = JitEvm::new(inner, backend.clone());
    let mut handler = revm_handler::MainnetHandler::default();
    let exec_result = handler.run(&mut evm);
    if exec_result.is_ok() {
        let s = evm.ctx.journaled_state.finalize();
        DatabaseCommit::commit(&mut evm.ctx.journaled_state.database, s);
    }
    *ctx.elapsed.lock().unwrap() += timer.elapsed();

    check_evm_execution(
        ctx.test,
        ctx.unit.out.as_ref(),
        ctx.name,
        &exec_result,
        &mut evm.ctx.journaled_state.database,
        *ctx.cfg.spec(),
        false,
    )
}

struct RuntimeTestContext<'a> {
    test: &'a revm_statetest_types::Test,
    unit: &'a TestUnit,
    name: &'a str,
    cfg: &'a CfgEnv,
    block: &'a BlockEnv,
    tx: &'a TxEnv,
    cache_state: &'a database::CacheState,
    elapsed: &'a Arc<Mutex<Duration>>,
}

/// Execute a test suite file using the runtime backend.
///
/// For each test unit, we enqueue JIT compilation via the backend and wait for all
/// contracts to be compiled before executing. This exercises the full JIT pipeline.
fn execute_test_suite_runtime(
    path: &Path,
    elapsed: &Arc<Mutex<Duration>>,
    backend: &JitBackend,
) -> Result<(), TestError> {
    if skip_test(path) {
        return Ok(());
    }

    let s = std::fs::read_to_string(path).unwrap();
    let path_str = path.to_string_lossy().into_owned();
    let suite: TestSuite = serde_json::from_str(&s).map_err(|e| TestError {
        name: "Unknown".to_string(),
        path: path_str.clone(),
        kind: e.into(),
    })?;

    for (name, unit) in suite.0 {
        let cache_state = unit.state();
        let mut cfg = CfgEnv::default();
        cfg.chain_id = unit.env.current_chain_id.unwrap_or(U256::ONE).try_into().unwrap_or(1);

        for (spec_name, tests) in &unit.post {
            if *spec_name == SpecName::Constantinople {
                continue;
            }

            let spec_id = spec_name.to_spec_id();
            cfg.set_spec_and_mainnet_gas_params(spec_id);

            if cfg.spec().is_enabled_in(SpecId::OSAKA) {
                cfg.set_max_blobs_per_tx(6);
            } else if cfg.spec().is_enabled_in(SpecId::PRAGUE) {
                cfg.set_max_blobs_per_tx(9);
            } else {
                cfg.set_max_blobs_per_tx(6);
            }

            let block = unit.block_env(&mut cfg);

            // Enqueue JIT compilation for all contracts in this unit.
            for info in unit.pre.values() {
                if info.code.is_empty() {
                    continue;
                }
                let code_hash = keccak256(&info.code);
                let req = LookupRequest { code_hash, code: info.code.clone(), spec_id };
                backend.compile_jit(req);
            }

            for test in tests.iter() {
                let tx = match test.tx_env(&unit) {
                    Ok(tx) => tx,
                    Err(_) if test.expect_exception.is_some() => continue,
                    Err(_) => {
                        return Err(TestError {
                            name,
                            path: path_str,
                            kind: TestErrorKind::UnknownPrivateKey(unit.transaction.secret_key),
                        });
                    }
                };

                let result = execute_single_test_runtime(
                    backend,
                    RuntimeTestContext {
                        test,
                        unit: &unit,
                        name: &name,
                        cfg: &cfg,
                        block: &block,
                        tx: &tx,
                        cache_state: &cache_state,
                        elapsed,
                    },
                );

                if let Err(e) = result {
                    return Err(TestError { name, path: path_str, kind: e });
                }
            }
        }
    }
    Ok(())
}

// ── Top-level runner ────────────────────────────────────────────────────────

fn run_test_worker(
    state: TestRunnerState,
    keep_going: bool,
    mode: CompileMode,
    cache: Option<&CompileCache>,
    backend: Option<&JitBackend>,
) -> Result<(), TestError> {
    loop {
        if !keep_going && state.n_errors.load(Ordering::SeqCst) > 0 {
            return Ok(());
        }

        let Some(test_path) = state.next_test() else {
            return Ok(());
        };

        let result = match mode {
            CompileMode::Interpreter => {
                execute_test_suite(&test_path, &state.elapsed, false, false)
            }
            CompileMode::Jit | CompileMode::Aot => {
                execute_test_suite_compiled(&test_path, &state.elapsed, cache.unwrap())
            }
            CompileMode::Runtime => {
                execute_test_suite_runtime(&test_path, &state.elapsed, backend.unwrap())
            }
        };

        state.console_bar.inc(1);

        if let Err(err) = result {
            state.n_errors.fetch_add(1, Ordering::SeqCst);
            if !keep_going {
                return Err(err);
            }
        }
    }
}

/// Run all test files.
pub fn run(
    test_files: Vec<PathBuf>,
    single_thread: bool,
    keep_going: bool,
    mode: CompileMode,
) -> Result<(), TestError> {
    let _ = tracing_subscriber::fmt::try_init();

    let n_files = test_files.len();
    let state = TestRunnerState::new(test_files);

    let cache = match mode {
        CompileMode::Interpreter | CompileMode::Runtime => None,
        CompileMode::Jit | CompileMode::Aot => Some(Arc::new(CompileCache::new(mode))),
    };

    // For runtime mode, start the backend with an empty store.
    // All lookups will miss → enqueue JIT → compile on worker threads.
    // This exercises the full JIT pipeline end-to-end.
    let backend = if mode == CompileMode::Runtime {
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let config = RuntimeConfig {
            enabled: true,
            tuning: RuntimeTuning { jit_worker_count: cpus, ..Default::default() },
            ..Default::default()
        };
        Some(JitBackend::new(config).map_err(|e| TestError {
            name: "backend".to_string(),
            path: String::new(),
            kind: TestErrorKind::CompilationError(format!("backend start: {e}")),
        })?)
    } else {
        None
    };

    let num_threads = if single_thread {
        1
    } else {
        match std::thread::available_parallelism() {
            Ok(n) => n.get().min(n_files),
            Err(_) => 1,
        }
    };

    let barrier = Arc::new(Barrier::new(num_threads));

    let mut handles = Vec::with_capacity(num_threads);
    for i in 0..num_threads {
        let state = state.clone();
        let cache = cache.clone();
        let backend = backend.clone();
        let barrier = barrier.clone();

        let thread = std::thread::Builder::new()
            .name(format!("runner-{i}"))
            .spawn(move || {
                let result =
                    run_test_worker(state, keep_going, mode, cache.as_deref(), backend.as_ref());
                // Wait for all threads before exiting. Each thread holds a thread-local
                // LLVM context that is destroyed on thread exit; concurrent context
                // disposal crashes LLVM.
                barrier.wait();
                result
            })
            .unwrap();

        handles.push(thread);
    }

    let mut thread_errors = Vec::new();
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => thread_errors.push(e),
            Err(_) => thread_errors.push(TestError {
                name: format!("thread {i} panicked"),
                path: String::new(),
                kind: TestErrorKind::Panic,
            }),
        }
    }

    state.console_bar.finish();

    println!(
        "Finished execution. Total CPU time: {:.6}s",
        state.elapsed.lock().unwrap().as_secs_f64()
    );

    if let Some(cache) = &cache {
        cache.print_stats();
    }

    // Print runtime backend stats.
    if let Some(backend) = &backend {
        let stats = backend.stats();
        println!(
            "Runtime backend: {} hits, {} misses, {} events sent, {} events dropped, {} resident",
            stats.lookup_hits,
            stats.lookup_misses,
            stats.events_sent,
            stats.events_dropped,
            stats.resident_entries,
        );
    }

    drop(backend);

    let n_errors = state.n_errors.load(Ordering::SeqCst);
    let n_thread_errors = thread_errors.len();

    if n_errors == 0 && n_thread_errors == 0 {
        println!("All tests passed!");
        Ok(())
    } else {
        println!("Encountered {n_errors} errors out of {n_files} total tests");

        if n_thread_errors == 0 {
            std::process::exit(1);
        }

        if n_thread_errors > 1 {
            println!("{n_thread_errors} threads returned an error, out of {num_threads} total:");
            for error in &thread_errors {
                println!("{error}");
            }
        }
        Err(thread_errors.swap_remove(0))
    }
}

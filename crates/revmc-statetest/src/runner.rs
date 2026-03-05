use crate::merkle_trie::{compute_test_roots, TestValidationResult};
use revm::{
    context::{block::BlockEnv, cfg::CfgEnv, tx::TxEnv},
    context_interface::result::{EVMError, ExecutionResult, HaltReason, InvalidTransaction},
    database::{self, bal::EvmDatabaseError},
    database_interface::EmptyDB,
    primitives::{hardfork::SpecId, keccak256, Bytes, B256, U256},
    statetest_types::{SpecName, Test, TestSuite, TestUnit},
    Context, ExecuteCommitEvm, MainBuilder, MainContext,
};
use indicatif::{ProgressBar, ProgressDrawTarget};
use revmc::{
    llvm::inkwell::context::Context as LlvmContext, Backend, EvmCompiler, EvmCompilerFn,
    EvmLlvmBackend, Linker, OptimizationLevel,
};
use serde_json::json;
use std::{
    collections::HashMap,
    convert::Infallible,
    fmt::Debug,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use thiserror::Error;

/// How to compile and execute bytecodes in the test suite.
#[derive(Clone, Copy, Debug, Default)]
pub enum CompileMode {
    /// Standard interpreter execution (no compilation).
    #[default]
    Interpreter,
    /// JIT-compile all bytecodes before execution.
    Jit,
    /// AOT-compile all bytecodes to a shared library, then load and execute.
    Aot,
}

/// Error that occurs during test execution
#[derive(Debug, Error)]
#[error("Path: {path}\nName: {name}\nError: {kind}")]
pub struct TestError {
    pub name: String,
    pub path: String,
    pub kind: TestErrorKind,
}

/// Specific kind of error that occurred during test execution
#[derive(Debug, Error)]
pub enum TestErrorKind {
    #[error("logs root mismatch: got {got}, expected {expected}")]
    LogsRootMismatch { got: B256, expected: B256 },
    #[error("state root mismatch: got {got}, expected {expected}")]
    StateRootMismatch { got: B256, expected: B256 },
    #[error("unknown private key: {0:?}")]
    UnknownPrivateKey(B256),
    #[error("unexpected exception: got {got_exception:?}, expected {expected_exception:?}")]
    UnexpectedException { expected_exception: Option<String>, got_exception: Option<String> },
    #[error("unexpected output: got {got_output:?}, expected {expected_output:?}")]
    UnexpectedOutput { expected_output: Option<Bytes>, got_output: Option<Bytes> },
    #[error(transparent)]
    SerdeDeserialize(#[from] serde_json::Error),
    #[error("thread panicked")]
    Panic,
    #[error("path does not exist")]
    InvalidPath,
    #[error("no JSON test files found in path")]
    NoJsonFiles,
    #[error("compilation failed: {0}")]
    CompilationError(String),
}

/// Check if a test should be skipped based on its filename.
/// Some tests are known to be problematic or take too long.
fn skip_test(path: &Path) -> bool {
    let path_str = path.to_str().unwrap_or_default();

    // Skip tests that have storage for newly created account.
    if path_str.contains("paris/eip7610_create_collision") {
        return true;
    }

    let name = path.file_name().unwrap().to_str().unwrap_or_default();

    matches!(
        name,
        // Test check if gas price overflows, we handle this correctly but does not match tests
        // specific exception.
        | "CreateTransactionHighNonce.json"

        // Test with some storage check.
        | "RevertInCreateInInit_Paris.json"
        | "RevertInCreateInInit.json"
        | "dynamicAccountOverwriteEmpty.json"
        | "dynamicAccountOverwriteEmpty_Paris.json"
        | "RevertInCreateInInitCreate2Paris.json"
        | "create2collisionStorage.json"
        | "RevertInCreateInInitCreate2.json"
        | "create2collisionStorageParis.json"
        | "InitCollision.json"
        | "InitCollisionParis.json"
        | "test_init_collision_create_opcode.json"

        // Malformed value.
        | "ValueOverflow.json"
        | "ValueOverflowParis.json"

        // These tests are passing, but they take a lot of time to execute so we are going to skip them.
        | "Call50000_sha256.json"
        | "static_Call50000_sha256.json"
        | "loopMul.json"
        | "CALLBlake2f_MaxRounds.json"
    )
}

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

/// JIT-compile all non-empty contracts in the test unit's pre-state.
fn jit_compile_contracts<'ctx>(
    unit: &TestUnit,
    spec_id: SpecId,
    compiler: &mut EvmCompiler<EvmLlvmBackend<'ctx>>,
) -> Result<CompiledContracts, TestErrorKind> {
    let mut compiled = CompiledContracts::new();
    let mut func_ids: Vec<(B256, <EvmLlvmBackend<'ctx> as Backend>::FuncId)> = Vec::new();

    for (address, info) in &unit.pre {
        if info.code.is_empty() {
            continue;
        }

        let code_hash = keccak256(&info.code);
        if func_ids.iter().any(|(hash, _)| hash == &code_hash) {
            continue;
        }

        let name = format!("contract_{:x}", address);
        let func_id = compiler.translate(&name, &info.code[..], spec_id).map_err(|e| {
            TestErrorKind::CompilationError(format!("translate {:x}: {e}", address))
        })?;

        func_ids.push((code_hash, func_id));
    }

    for (code_hash, func_id) in func_ids {
        let func = unsafe { compiler.jit_function(func_id) }
            .map_err(|e| TestErrorKind::CompilationError(format!("jit {:x}: {e}", code_hash)))?;

        compiled.insert(code_hash, func);
    }

    Ok(compiled)
}

/// AOT-compile all non-empty contracts in the test unit's pre-state to a shared library,
/// then load the compiled functions.
fn aot_compile_contracts(
    unit: &TestUnit,
    spec_id: SpecId,
) -> Result<(CompiledContracts, tempfile::TempDir, libloading::Library), TestErrorKind> {
    let cx = LlvmContext::create();
    let backend = EvmLlvmBackend::new(&cx, true, OptimizationLevel::Default)
        .map_err(|e| TestErrorKind::CompilationError(format!("backend: {e}")))?;
    let mut compiler = EvmCompiler::new(backend);

    let mut names: Vec<(B256, String)> = Vec::new();

    for (address, info) in &unit.pre {
        if info.code.is_empty() {
            continue;
        }

        let code_hash = keccak256(&info.code);
        if names.iter().any(|(hash, _)| hash == &code_hash) {
            continue;
        }

        let name = format!("contract_{:x}", address);
        compiler.translate(&name, &info.code[..], spec_id).map_err(|e| {
            TestErrorKind::CompilationError(format!("translate {:x}: {e}", address))
        })?;

        names.push((code_hash, name));
    }

    let tmp_dir = tempfile::tempdir()
        .map_err(|e| TestErrorKind::CompilationError(format!("tempdir: {e}")))?;
    let obj_path = tmp_dir.path().join("a.o");
    let so_path = tmp_dir.path().join("a.so");

    compiler
        .write_object_to_file(&obj_path)
        .map_err(|e| TestErrorKind::CompilationError(format!("write object: {e}")))?;

    let linker = Linker::new();
    linker
        .link(&so_path, [obj_path.to_str().unwrap()])
        .map_err(|e| TestErrorKind::CompilationError(format!("link: {e}")))?;

    let lib = unsafe { libloading::Library::new(&so_path) }
        .map_err(|e| TestErrorKind::CompilationError(format!("load: {e}")))?;

    let mut compiled = CompiledContracts::new();
    for (code_hash, name) in &names {
        let f: libloading::Symbol<'_, EvmCompilerFn> = unsafe { lib.get(name.as_bytes()) }
            .map_err(|e| TestErrorKind::CompilationError(format!("symbol {name}: {e}")))?;
        compiled.insert(*code_hash, *f);
    }

    Ok((compiled, tmp_dir, lib))
}

struct TestExecutionContext<'a> {
    name: &'a str,
    unit: &'a TestUnit,
    test: &'a Test,
    cfg: &'a CfgEnv,
    block: &'a BlockEnv,
    tx: &'a TxEnv,
    cache_state: &'a database::CacheState,
    elapsed: &'a Arc<Mutex<Duration>>,
    #[allow(dead_code)]
    trace: bool,
    print_json_outcome: bool,
}

fn build_json_output(
    test: &Test,
    test_name: &str,
    exec_result: &Result<
        ExecutionResult<HaltReason>,
        EVMError<EvmDatabaseError<Infallible>, InvalidTransaction>,
    >,
    validation: &TestValidationResult,
    spec: SpecId,
    error: Option<String>,
) -> serde_json::Value {
    json!({
        "stateRoot": validation.state_root,
        "logsRoot": validation.logs_root,
        "output": exec_result.as_ref().ok().and_then(|r| r.output().cloned()).unwrap_or_default(),
        "gasUsed": exec_result.as_ref().ok().map(|r| r.gas_used()).unwrap_or_default(),
        "pass": error.is_none(),
        "errorMsg": error.unwrap_or_default(),
        "evmResult": format_evm_result(exec_result),
        "postLogsHash": validation.logs_root,
        "fork": spec,
        "test": test_name,
        "d": test.indexes.data,
        "g": test.indexes.gas,
        "v": test.indexes.value,
    })
}

fn format_evm_result(
    exec_result: &Result<
        ExecutionResult<HaltReason>,
        EVMError<EvmDatabaseError<Infallible>, InvalidTransaction>,
    >,
) -> String {
    match exec_result {
        Ok(r) => match r {
            ExecutionResult::Success { reason, .. } => format!("Success: {reason:?}"),
            ExecutionResult::Revert { .. } => "Revert".to_string(),
            ExecutionResult::Halt { reason, .. } => format!("Halt: {reason:?}"),
        },
        Err(e) => e.to_string(),
    }
}

fn validate_exception(
    test: &Test,
    exec_result: &Result<
        ExecutionResult<HaltReason>,
        EVMError<EvmDatabaseError<Infallible>, InvalidTransaction>,
    >,
) -> Result<bool, TestErrorKind> {
    match (&test.expect_exception, exec_result) {
        (None, Ok(_)) => Ok(false),
        (Some(_), Err(_)) => Ok(true),
        _ => Err(TestErrorKind::UnexpectedException {
            expected_exception: test.expect_exception.clone(),
            got_exception: exec_result.as_ref().err().map(|e| e.to_string()),
        }),
    }
}

fn validate_output(
    expected_output: Option<&Bytes>,
    actual_result: &ExecutionResult<HaltReason>,
) -> Result<(), TestErrorKind> {
    if let Some((expected, actual)) = expected_output.zip(actual_result.output()) {
        if expected != actual {
            return Err(TestErrorKind::UnexpectedOutput {
                expected_output: Some(expected.clone()),
                got_output: actual_result.output().cloned(),
            });
        }
    }
    Ok(())
}

fn check_evm_execution(
    test: &Test,
    expected_output: Option<&Bytes>,
    test_name: &str,
    exec_result: &Result<
        ExecutionResult<HaltReason>,
        EVMError<EvmDatabaseError<Infallible>, InvalidTransaction>,
    >,
    db: &mut database::State<EmptyDB>,
    spec: SpecId,
    print_json_outcome: bool,
) -> Result<(), TestErrorKind> {
    let validation = compute_test_roots(exec_result, db);

    let print_json = |error: Option<&TestErrorKind>| {
        if print_json_outcome {
            let json = build_json_output(
                test,
                test_name,
                exec_result,
                &validation,
                spec,
                error.map(|e| e.to_string()),
            );
            eprintln!("{json}");
        }
    };

    // Check if exception handling is correct
    let exception_expected = validate_exception(test, exec_result).inspect_err(|e| {
        print_json(Some(e));
    })?;

    // If exception was expected and occurred, we're done
    if exception_expected {
        print_json(None);
        return Ok(());
    }

    // Validate output if execution succeeded
    if let Ok(result) = exec_result {
        validate_output(expected_output, result).inspect_err(|e| {
            print_json(Some(e));
        })?;
    }

    // Validate logs root
    if validation.logs_root != test.logs {
        let error =
            TestErrorKind::LogsRootMismatch { got: validation.logs_root, expected: test.logs };
        print_json(Some(&error));
        return Err(error);
    }

    // Validate state root
    if validation.state_root != test.hash {
        let error =
            TestErrorKind::StateRootMismatch { got: validation.state_root, expected: test.hash };
        print_json(Some(&error));
        return Err(error);
    }

    print_json(None);
    Ok(())
}

/// Execute a single test suite file containing multiple tests.
pub fn execute_test_suite(
    path: &Path,
    elapsed: &Arc<Mutex<Duration>>,
    trace: bool,
    print_json_outcome: bool,
) -> Result<(), TestError> {
    if skip_test(path) {
        return Ok(());
    }

    let s = std::fs::read_to_string(path).unwrap();
    let path = path.to_string_lossy().into_owned();
    let suite: TestSuite = serde_json::from_str(&s).map_err(|e| TestError {
        name: "Unknown".to_string(),
        path: path.clone(),
        kind: e.into(),
    })?;

    for (name, unit) in suite.0 {
        // Prepare initial state
        let cache_state = unit.state();

        // Setup base configuration
        let mut cfg = CfgEnv::default();
        cfg.chain_id = unit.env.current_chain_id.unwrap_or(U256::ONE).try_into().unwrap_or(1);

        // Post and execution
        for (spec_name, tests) in &unit.post {
            // Skip Constantinople spec
            if *spec_name == SpecName::Constantinople {
                continue;
            }

            cfg.set_spec_and_mainnet_gas_params(spec_name.to_spec_id());

            // Configure max blobs per spec
            if cfg.spec().is_enabled_in(SpecId::OSAKA) {
                cfg.set_max_blobs_per_tx(6);
            } else if cfg.spec().is_enabled_in(SpecId::PRAGUE) {
                cfg.set_max_blobs_per_tx(9);
            } else {
                cfg.set_max_blobs_per_tx(6);
            }

            // Setup block environment for this spec
            let block = unit.block_env(&mut cfg);

            for test in tests.iter() {
                // Setup transaction environment
                let tx = match test.tx_env(&unit) {
                    Ok(tx) => tx,
                    Err(_) if test.expect_exception.is_some() => continue,
                    Err(_) => {
                        return Err(TestError {
                            name,
                            path,
                            kind: TestErrorKind::UnknownPrivateKey(unit.transaction.secret_key),
                        });
                    }
                };

                // Execute the test
                let result = execute_single_test(TestExecutionContext {
                    name: &name,
                    unit: &unit,
                    test,
                    cfg: &cfg,
                    block: &block,
                    tx: &tx,
                    cache_state: &cache_state,
                    elapsed,
                    trace,
                    print_json_outcome,
                });

                if let Err(e) = result {
                    // Handle error with debug trace if needed
                    static FAILED: AtomicBool = AtomicBool::new(false);
                    if print_json_outcome || FAILED.swap(true, Ordering::SeqCst) {
                        return Err(TestError { name, path, kind: e });
                    }

                    return Err(TestError { path, name, kind: e });
                }
            }
        }
    }
    Ok(())
}

fn execute_single_test(ctx: TestExecutionContext) -> Result<(), TestErrorKind> {
    // Prepare state
    let mut cache = ctx.cache_state.clone();
    let spec = ctx.cfg.spec();
    cache.set_state_clear_flag(spec.is_enabled_in(SpecId::SPURIOUS_DRAGON));
    let mut state =
        database::State::builder().with_cached_prestate(cache).with_bundle_update().build();

    let evm_context = Context::mainnet()
        .with_block(ctx.block)
        .with_tx(ctx.tx)
        .with_cfg(ctx.cfg.clone())
        .with_db(&mut state);

    // Execute
    let timer = Instant::now();
    let mut evm = evm_context.build_mainnet();
    let exec_result = evm.transact_commit(ctx.tx);
    let db = evm.ctx.journaled_state.database;
    *ctx.elapsed.lock().unwrap() += timer.elapsed();

    // Check results
    check_evm_execution(
        ctx.test,
        ctx.unit.out.as_ref(),
        ctx.name,
        &exec_result,
        db,
        *ctx.cfg.spec(),
        ctx.print_json_outcome,
    )
}

/// Execute a single test suite file, JIT-compiling all contracts before execution.
///
/// For each test case, all non-empty contracts in the pre-state are JIT-compiled.
/// The compiled functions are currently not used in the EVM execution path
/// (TODO: integrate with frame execution), but this validates that all bytecodes
/// compile successfully.
pub fn execute_test_suite_jit(
    path: &Path,
    elapsed: &Arc<Mutex<Duration>>,
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

            // JIT-compile all contracts for this spec.
            let cx = LlvmContext::create();
            let backend =
                EvmLlvmBackend::new(&cx, false, OptimizationLevel::Default).map_err(|e| {
                    TestError {
                        name: name.clone(),
                        path: path_str.clone(),
                        kind: TestErrorKind::CompilationError(format!("backend: {e}")),
                    }
                })?;
            let mut compiler = EvmCompiler::new(backend);
            let _compiled = jit_compile_contracts(&unit, spec_id, &mut compiler)
                .map_err(|e| TestError { name: name.clone(), path: path_str.clone(), kind: e })?;

            // Run each test case through the standard EVM.
            // TODO: hook compiled functions into the EVM frame execution so contracts
            // run through JIT instead of the interpreter.
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

                let mut cache = cache_state.clone();
                cache.set_state_clear_flag(cfg.spec().is_enabled_in(SpecId::SPURIOUS_DRAGON));
                let mut state = database::State::builder()
                    .with_cached_prestate(cache)
                    .with_bundle_update()
                    .build();

                let evm_context = Context::mainnet()
                    .with_block(&block)
                    .with_tx(&tx)
                    .with_cfg(cfg.clone())
                    .with_db(&mut state);

                let timer = Instant::now();
                let mut evm = evm_context.build_mainnet();
                let exec_result = evm.transact_commit(&tx);
                let db = evm.ctx.journaled_state.database;
                *elapsed.lock().unwrap() += timer.elapsed();

                let result = check_evm_execution(
                    test,
                    unit.out.as_ref(),
                    &name,
                    &exec_result,
                    db,
                    *cfg.spec(),
                    false,
                );

                if let Err(e) = result {
                    return Err(TestError { name, path: path_str, kind: e });
                }
            }
        }
    }
    Ok(())
}

/// Execute a single test suite file, AOT-compiling all contracts before execution.
///
/// For each test case, all non-empty contracts are compiled to a shared library,
/// loaded, and the functions are verified to exist.
pub fn execute_test_suite_aot(
    path: &Path,
    elapsed: &Arc<Mutex<Duration>>,
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

            // AOT-compile all contracts for this spec.
            let (_compiled, _tmp_dir, _lib) = aot_compile_contracts(&unit, spec_id)
                .map_err(|e| TestError { name: name.clone(), path: path_str.clone(), kind: e })?;

            // Run each test case through the standard EVM.
            // TODO: hook compiled functions into the EVM frame execution so contracts
            // run through the AOT-loaded functions instead of the interpreter.
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

                let mut cache = cache_state.clone();
                cache.set_state_clear_flag(cfg.spec().is_enabled_in(SpecId::SPURIOUS_DRAGON));
                let mut state = database::State::builder()
                    .with_cached_prestate(cache)
                    .with_bundle_update()
                    .build();

                let evm_context = Context::mainnet()
                    .with_block(&block)
                    .with_tx(&tx)
                    .with_cfg(cfg.clone())
                    .with_db(&mut state);

                let timer = Instant::now();
                let mut evm = evm_context.build_mainnet();
                let exec_result = evm.transact_commit(&tx);
                let db = evm.ctx.journaled_state.database;
                *elapsed.lock().unwrap() += timer.elapsed();

                let result = check_evm_execution(
                    test,
                    unit.out.as_ref(),
                    &name,
                    &exec_result,
                    db,
                    *cfg.spec(),
                    false,
                );

                if let Err(e) = result {
                    return Err(TestError { name, path: path_str, kind: e });
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
struct TestRunnerState {
    n_errors: Arc<AtomicUsize>,
    console_bar: Arc<ProgressBar>,
    queue: Arc<Mutex<(usize, Vec<PathBuf>)>>,
    elapsed: Arc<Mutex<Duration>>,
}

impl TestRunnerState {
    fn new(test_files: Vec<PathBuf>) -> Self {
        let n_files = test_files.len();
        Self {
            n_errors: Arc::new(AtomicUsize::new(0)),
            console_bar: Arc::new(ProgressBar::with_draw_target(
                Some(n_files as u64),
                ProgressDrawTarget::stdout(),
            )),
            queue: Arc::new(Mutex::new((0usize, test_files))),
            elapsed: Arc::new(Mutex::new(Duration::ZERO)),
        }
    }

    fn next_test(&self) -> Option<PathBuf> {
        let (current_idx, queue) = &mut *self.queue.lock().unwrap();
        let idx = *current_idx;
        let test_path = queue.get(idx).cloned()?;
        *current_idx = idx + 1;
        Some(test_path)
    }
}

fn run_test_worker(
    state: TestRunnerState,
    keep_going: bool,
    mode: CompileMode,
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
            CompileMode::Jit => execute_test_suite_jit(&test_path, &state.elapsed),
            CompileMode::Aot => execute_test_suite_aot(&test_path, &state.elapsed),
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
    let n_files = test_files.len();
    let state = TestRunnerState::new(test_files);

    // JIT/AOT modes use single thread since the LLVM context is not Send.
    let single_thread = single_thread || !matches!(mode, CompileMode::Interpreter);
    let num_threads = if single_thread {
        1
    } else {
        match std::thread::available_parallelism() {
            Ok(n) => n.get().min(n_files),
            Err(_) => 1,
        }
    };

    let mut handles = Vec::with_capacity(num_threads);
    for i in 0..num_threads {
        let state = state.clone();

        let thread = std::thread::Builder::new()
            .name(format!("runner-{i}"))
            .spawn(move || run_test_worker(state, keep_going, mode))
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

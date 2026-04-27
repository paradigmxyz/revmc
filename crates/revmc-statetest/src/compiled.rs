// revmc-specific code: compilation, handler integration, and test orchestration.

use crate::runner::{
    TestError, TestErrorKind, TestRunnerState, check_evm_execution, execute_test_suite, skip_test,
};
use revm_context::{Context, block::BlockEnv, cfg::CfgEnv, tx::TxEnv};
use revm_database::{self as database};
use revm_database_interface::DatabaseCommit;
use revm_handler::{Handler, MainBuilder, MainContext};
use revm_primitives::{U256, hardfork::SpecId};
use revm_statetest_types::{SpecName, TestSuite, TestUnit};
use std::{
    fs,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    process,
    sync::{Arc, Barrier, Mutex, atomic::Ordering},
    thread::{self, Builder},
    time::{Duration, Instant},
};

// ── Compile mode ────────────────────────────────────────────────────────────

/// How to compile and execute bytecodes in the test suite.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompileMode {
    /// Standard interpreter execution (no compilation).
    #[default]
    Interpreter,
    /// Use the runtime backend and look up JIT-compiled functions via `JitBackend::lookup()`.
    Jit,
    /// AOT-compile all bytecodes to a shared library, then load and execute.
    Aot,
}

// ── Runtime backend mode ─────────────────────────────────────────────────────

use revmc::{
    revm_evm::JitEvm,
    runtime::{ArtifactStore, JitBackend, RuntimeArtifactStore, RuntimeConfig, RuntimeTuning},
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
/// For each test unit, enqueue JIT compilation via the backend before executing.
fn execute_test_suite_runtime(
    path: &Path,
    elapsed: &Arc<Mutex<Duration>>,
    backend: &JitBackend,
) -> Result<(), TestError> {
    if skip_test(path) {
        return Ok(());
    }

    let s = fs::read_to_string(path).unwrap();
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

    let backend = if matches!(mode, CompileMode::Aot | CompileMode::Jit) {
        let cpus = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let store = if mode == CompileMode::Aot {
            let store = RuntimeArtifactStore::new().map_err(|e| TestError {
                name: "backend".to_string(),
                path: String::new(),
                kind: TestErrorKind::CompilationError(format!("tempdir: {e}")),
            })?;
            Some(Arc::new(store) as Arc<dyn ArtifactStore>)
        } else {
            None
        };
        let config = RuntimeConfig {
            enabled: true,
            blocking: true,
            aot: mode == CompileMode::Aot,
            store,
            tuning: RuntimeTuning {
                jit_hot_threshold: 0,
                jit_worker_count: cpus,
                ..Default::default()
            },
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
        match thread::available_parallelism() {
            Ok(n) => n.get().min(n_files),
            Err(_) => 1,
        }
    };

    let barrier = Arc::new(Barrier::new(num_threads));

    let mut handles = Vec::with_capacity(num_threads);
    for i in 0..num_threads {
        let state = state.clone();
        let backend = backend.clone();
        let barrier = barrier.clone();

        let thread = Builder::new()
            .name(format!("runner-{i}"))
            .spawn(move || {
                // Catch panics so we always reach `barrier.wait()` below; otherwise a
                // panicking worker would never advance the barrier and the remaining
                // workers would deadlock waiting for it. Also flips the shared `stop`
                // flag so siblings exit promptly instead of finishing the whole queue.
                let stop = state.stop.clone();
                let result = panic::catch_unwind(AssertUnwindSafe(|| {
                    run_test_worker(state, keep_going, mode, backend.as_ref())
                }));
                if result.is_err() || (!keep_going && result.as_ref().is_ok_and(|r| r.is_err())) {
                    stop.store(true, Ordering::SeqCst);
                }
                // Wait for all threads before exiting. Each thread holds a thread-local
                // LLVM context that is destroyed on thread exit; concurrent context
                // disposal crashes LLVM.
                barrier.wait();
                match result {
                    Ok(r) => r,
                    Err(payload) => panic::resume_unwind(payload),
                }
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

    if let Some(backend) = &backend {
        let stats = backend.stats();
        println!(
            "Runtime backend: {} hits, {} misses, {} resident",
            stats.lookup_hits, stats.lookup_misses, stats.resident_entries,
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
            process::exit(1);
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

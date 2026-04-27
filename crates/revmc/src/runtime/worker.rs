//! Background JIT and AOT compilation workers.
//!
//! Each worker thread owns a long-lived `EvmCompiler` instance tied to its
//! thread-local LLVM context.
//!
//! With ORCv2, JIT code lifetime is managed per-module via `ResourceTracker`s.
//! Each successful JIT compilation extracts the committed module's tracker and
//! returns it as a [`JitCodeBacking`], which frees the machine code on drop.
//! Workers no longer need to stay alive for code lifetime — they exit as soon
//! as the job channel closes.

use crate::{
    CompileTimings, EvmCompilerFn, OptimizationLevel,
    runtime::{
        config::{CompilationKind, RuntimeConfig},
        storage::RuntimeCacheKey,
    },
};
use alloy_primitives::Bytes;
use crossbeam_channel as chan;
use crossbeam_deque::{Injector, Steal, Worker};
use std::{
    fs::File,
    io::Read,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[cfg(feature = "llvm")]
use crate::{
    EvmCompiler, EvmLlvmBackend, Linker,
    llvm::{JitDylibGuard, orc::ResourceTracker},
};

/// Notifier for synchronous compilation requests.
///
/// Wraps an optional sender that is notified when the compilation
/// completes (success or failure). Passed from the backend through
/// the worker and back, then fired after the result is processed.
pub(crate) struct SyncNotifier(Option<chan::Sender<()>>);

impl SyncNotifier {
    pub(crate) fn none() -> Self {
        Self(None)
    }

    pub(crate) fn new(tx: chan::Sender<()>) -> Self {
        Self(Some(tx))
    }

    pub(crate) fn notify(self) {
        if let Some(tx) = self.0 {
            let _ = tx.send(());
        }
    }
}

/// A compilation job sent from the backend to a worker.
#[derive(derive_more::Debug)]
pub(crate) struct CompileJob {
    /// Whether this job compiles JIT or AOT output.
    pub(crate) kind: CompilationKind,
    /// The key to compile for.
    pub(crate) key: RuntimeCacheKey,
    /// The raw bytecode to compile.
    pub(crate) bytecode: Bytes,
    /// The symbol name to use for the compiled function.
    pub(crate) symbol_name: String,
    /// Optimization level for compilation.
    pub(crate) opt_level: OptimizationLevel,
    /// Optional notifier for synchronous callers.
    #[debug(skip)]
    pub(crate) sync_notifier: SyncNotifier,
    /// Generation at the time the job was dispatched.
    pub(crate) generation: u64,
}

/// Result of a compilation attempt, sent back from a worker to the backend.
pub(crate) struct WorkerResult {
    /// The key that was compiled.
    pub(crate) key: RuntimeCacheKey,
    /// The compilation outcome.
    pub(crate) outcome: Result<WorkerSuccess, String>,
    /// Whether this was a JIT or AOT compilation job.
    pub(crate) kind: CompilationKind,
    /// Optional notifier for synchronous callers, passed through from the job.
    pub(crate) sync_notifier: SyncNotifier,
    /// Generation at the time the job was dispatched.
    pub(crate) generation: u64,
    /// Wall-clock time spent compiling.
    pub(crate) compile_duration: Duration,
    /// Per-phase timing breakdown from the compiler.
    pub(crate) timings: CompileTimings,
}

/// Successful compilation output.
pub(crate) enum WorkerSuccess {
    /// JIT compilation produced an in-memory function pointer.
    Jit(JitSuccess),
    /// AOT compilation produced shared-library bytes.
    Aot(AotSuccess),
}

/// Successful JIT compilation output.
pub(crate) struct JitSuccess {
    /// The compiled function pointer.
    pub(crate) func: EvmCompilerFn,
    /// Owns the JIT machine code via an ORCv2 `ResourceTracker`.
    /// Dropping this frees the compiled code.
    pub(crate) backing: Arc<JitCodeBacking>,
}

/// Successful AOT compilation output.
pub(crate) struct AotSuccess {
    /// The symbol name in the shared library.
    pub(crate) symbol_name: String,
    /// The raw shared-library bytes (.so / .dylib).
    pub(crate) dylib_bytes: Vec<u8>,
    /// Length of the original bytecode.
    pub(crate) bytecode_len: usize,
}

/// Owns JIT-compiled machine code via an ORCv2 `ResourceTracker` and a
/// [`JitDylibGuard`](crate::llvm::JitDylibGuard).
///
/// The tracker provides per-entry code removal: dropping it calls
/// `tracker.remove()` which frees this entry's machine code.
/// The guard keeps the owning `JITDylib` alive — it won't be cleared
/// or recycled until all guards are dropped.
///
/// This enables true per-entry eviction: removing a `CompiledProgram`
/// from the resident map reclaims its machine code once all callers
/// release their `Arc<CompiledProgram>` handles.
pub(crate) struct JitCodeBacking {
    /// The tracker that owns this entry's machine code.
    /// Must be dropped (after `remove()`) BEFORE `_jd_guard` to ensure
    /// the JITDylib is still valid when we remove resources from it.
    #[cfg(feature = "llvm")]
    tracker: Option<ResourceTracker>,
    /// Keeps the owning JITDylib alive. Dropped after `tracker`.
    #[cfg(feature = "llvm")]
    _jd_guard: Arc<JitDylibGuard>,
}

impl JitCodeBacking {
    #[cfg(feature = "llvm")]
    pub(crate) fn new(tracker: ResourceTracker, jd_guard: Arc<JitDylibGuard>) -> Self {
        Self { tracker: Some(tracker), _jd_guard: jd_guard }
    }
}

#[cfg(feature = "llvm")]
impl Drop for JitCodeBacking {
    fn drop(&mut self) {
        if let Some(tracker) = self.tracker.take()
            && let Err(e) = tracker.remove()
        {
            warn!("failed to remove JIT code: {e}");
        }
    }
}

/// Handle to the worker pool. Manages worker threads and their job queues.
pub(crate) struct WorkerPool {
    /// Global queue jobs are injected into by the backend thread.
    injector: Arc<Injector<CompileJob>>,
    /// Wakeup channel for sleeping workers.
    notify_tx: Option<chan::Sender<()>>,
    /// Number of queued jobs across the injector and worker-local queues.
    queued: Arc<AtomicUsize>,
    /// Maximum queued jobs accepted by the pool.
    queue_capacity: usize,
    /// Signals workers to stop after draining queued work.
    shutdown: Arc<AtomicBool>,
    /// Worker thread handles.
    threads: Vec<Option<JoinHandle<()>>>,
}

impl WorkerPool {
    /// Creates and starts the worker pool.
    pub(crate) fn new(result_tx: chan::Sender<WorkerResult>, config: RuntimeConfig) -> Self {
        let worker_count = config.tuning.jit_worker_count;
        let queue_capacity = worker_count.saturating_mul(config.tuning.jit_worker_queue_capacity);
        let injector = Arc::new(Injector::new());
        let queued = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let (notify_tx, notify_rx) = chan::unbounded();
        let config = Arc::new(config);

        let mut threads = Vec::with_capacity(worker_count);
        for worker_id in 0..worker_count {
            let local = Worker::new_fifo();
            let result_tx = result_tx.clone();
            let config = config.clone();
            let injector = injector.clone();
            let queued = queued.clone();
            let shutdown = shutdown.clone();
            let notify_rx = notify_rx.clone();
            let thread = thread::Builder::new()
                .name(format!("revmc-{worker_id:02}"))
                .spawn(move || {
                    worker_loop(
                        worker_id, injector, local, notify_rx, queued, shutdown, result_tx, &config,
                    );
                })
                .expect("failed to spawn compile worker");

            threads.push(Some(thread));
        }

        Self { injector, notify_tx: Some(notify_tx), queued, queue_capacity, shutdown, threads }
    }

    /// Tries to send a job to the worker pool.
    /// Returns the job back on failure (queue full or shut down).
    pub(crate) fn try_send(&mut self, job: CompileJob) -> Result<(), CompileJob> {
        if self.threads.is_empty() || self.shutdown.load(Ordering::Acquire) {
            return Err(job);
        }

        let mut current = self.queued.load(Ordering::Acquire);
        loop {
            if current >= self.queue_capacity {
                return Err(job);
            }
            match self.queued.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }

        self.injector.push(job);
        if let Some(tx) = &self.notify_tx {
            let _ = tx.send(());
        }
        Ok(())
    }

    /// Shuts down all workers after draining queued jobs.
    pub(crate) fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.notify_tx.take();

        for thread in &mut self.threads {
            if let Some(t) = thread.take() {
                let _ = t.join();
            }
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn steal_job(
    injector: &Injector<CompileJob>,
    local: &Worker<CompileJob>,
    queued: &AtomicUsize,
) -> Option<CompileJob> {
    loop {
        if let Some(job) = local.pop() {
            queued.fetch_sub(1, Ordering::AcqRel);
            return Some(job);
        }

        match injector.steal_batch_and_pop(local) {
            Steal::Success(job) => {
                queued.fetch_sub(1, Ordering::AcqRel);
                return Some(job);
            }
            Steal::Retry => continue,
            Steal::Empty => {}
        }

        return None;
    }
}

#[cfg(feature = "llvm")]
#[allow(clippy::too_many_arguments)]
fn worker_loop(
    id: usize,
    injector: Arc<Injector<CompileJob>>,
    local: Worker<CompileJob>,
    notify_rx: chan::Receiver<()>,
    queued: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    result_tx: chan::Sender<WorkerResult>,
    config: &RuntimeConfig,
) {
    let _span = debug_span!("revmc_worker", id).entered();
    debug!("compile worker started");

    let create_compiler = || -> Result<EvmCompiler<EvmLlvmBackend>, String> {
        let backend = EvmLlvmBackend::new(false).map_err(|e| e.to_string())?;
        let mut compiler = EvmCompiler::new(backend);
        compiler.set_opt_level(config.tuning.jit_opt_level);
        compiler.debug_assertions(config.debug_assertions);
        compiler.set_dedup(!config.no_dedup);
        compiler.set_dse(!config.no_dse);
        if let Some(gas_params) = &config.gas_params {
            compiler.set_gas_params(gas_params.clone());
        }
        Ok(compiler)
    };

    let mut jit_compiler = match create_compiler() {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "failed to create LLVM backend, worker exiting");
            return;
        }
    };

    let mut compilations_since_recycle: usize = 0;

    loop {
        let Some(job) = steal_job(&injector, &local, &queued) else {
            if shutdown.load(Ordering::Acquire) && queued.load(Ordering::Acquire) == 0 {
                break;
            }
            match notify_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(()) | Err(chan::RecvTimeoutError::Timeout) => continue,
                Err(chan::RecvTimeoutError::Disconnected) => {
                    if queued.load(Ordering::Acquire) == 0 {
                        break;
                    }
                }
            }
            continue;
        };
        trace!(?job, "received job");
        let t0 = Instant::now();
        let kind = job.kind;
        let (key, outcome, sync_notifier, generation, timings) = match kind {
            CompilationKind::Jit => {
                let _span =
                    debug_span!("jit_compile", hash=%job.key.code_hash, spec_id=?job.key.spec_id)
                        .entered();

                if let Some(base) = &config.dump_dir {
                    let dir = base
                        .join(format!("{:?}", job.key.spec_id))
                        .join(format!("{}", job.key.code_hash));
                    jit_compiler.set_dump_to(Some(dir));
                }
                jit_compiler.set_opt_level(job.opt_level);

                let result = unsafe {
                    jit_compiler.jit(&job.symbol_name, &job.bytecode[..], job.key.spec_id)
                };

                let timings = jit_compiler.take_timings();

                let outcome = match result {
                    Ok(func) => {
                        let jd_guard = jit_compiler.backend_mut().jit_dylib_guard();

                        debug!("JIT compilation succeeded");
                        // The last loaded tracker owns this module's machine code.
                        // free_function would also work, but we need the tracker
                        // to outlive the backend so eviction can free code after
                        // the worker exits.
                        let tracker = jit_compiler
                            .backend_mut()
                            .take_last_resource_tracker()
                            .expect("no ResourceTracker after JIT");
                        let backing = Arc::new(JitCodeBacking::new(tracker, jd_guard));
                        Ok(WorkerSuccess::Jit(JitSuccess { func, backing }))
                    }
                    Err(err) => {
                        warn!(%err, "JIT compilation failed");
                        Err(format!("{err}"))
                    }
                };

                // Reset IR so the next job can reuse this compiler.
                if let Err(err) = jit_compiler.clear_ir() {
                    warn!(%err, "clear_ir failed");
                }

                (job.key, outcome, job.sync_notifier, job.generation, timings)
            }
            CompilationKind::Aot => {
                let _span =
                    debug_span!("aot_compile", hash=%job.key.code_hash, spec_id=?job.key.spec_id)
                        .entered();

                let generation = job.generation;
                let outcome = compile_aot_artifact(&job, config);
                (job.key, outcome, job.sync_notifier, generation, CompileTimings::default())
            }
        };
        let compile_duration = t0.elapsed();

        let _ = result_tx.send(WorkerResult {
            key,
            outcome,
            kind,
            sync_notifier,
            generation,
            compile_duration,
            timings,
        });

        // Recycle the compiler to reclaim LLVM context memory.
        compilations_since_recycle += 1;
        if config.tuning.compiler_recycle_threshold > 0
            && compilations_since_recycle >= config.tuning.compiler_recycle_threshold
        {
            debug!(compilations_since_recycle, "recycling compiler");
            drop(jit_compiler);
            jit_compiler = match create_compiler() {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "failed to recreate compiler, worker exiting");
                    return;
                }
            };
            revmc_llvm::global_gc();
            compilations_since_recycle = 0;
        }
    }

    drop(jit_compiler);
    debug!("compile worker shutting down");
}

/// Compiles a single bytecode to a shared library and returns the raw bytes.
#[cfg(feature = "llvm")]
fn compile_aot_artifact(job: &CompileJob, config: &RuntimeConfig) -> Result<WorkerSuccess, String> {
    let backend =
        EvmLlvmBackend::new(true).map_err(|e| format!("AOT backend creation failed: {e}"))?;
    let mut compiler = EvmCompiler::new(backend);
    compiler.set_opt_level(job.opt_level);
    compiler.debug_assertions(config.debug_assertions);
    compiler.set_dedup(!config.no_dedup);
    compiler.set_dse(!config.no_dse);
    if let Some(gas_params) = &config.gas_params {
        compiler.set_gas_params(gas_params.clone());
    }

    compiler
        .translate(&job.symbol_name, &job.bytecode[..], job.key.spec_id)
        .map_err(|e| format!("AOT translate failed: {e}"))?;

    let tmp_dir = tempfile::tempdir().map_err(|e| format!("failed to create temp dir: {e}"))?;

    let obj_path = tmp_dir.path().join("a.o");
    let so_path = tmp_dir.path().join("a.so");

    compiler
        .write_object_to_file(&obj_path)
        .map_err(|e| format!("AOT write object failed: {e}"))?;

    let linker = Linker::new();
    linker
        .link(&so_path, [obj_path.to_str().unwrap()])
        .map_err(|e| format!("AOT link failed: {e}"))?;

    let mut dylib_bytes = Vec::new();
    File::open(&so_path)
        .and_then(|mut f| f.read_to_end(&mut dylib_bytes))
        .map_err(|e| format!("failed to read linked .so: {e}"))?;

    debug!(
        bytecode_len = job.bytecode.len(),
        dylib_len = dylib_bytes.len(),
        "AOT compilation succeeded",
    );

    Ok(WorkerSuccess::Aot(AotSuccess {
        symbol_name: job.symbol_name.clone(),
        dylib_bytes,
        bytecode_len: job.bytecode.len(),
    }))
}

#[cfg(not(feature = "llvm"))]
#[allow(clippy::too_many_arguments)]
fn worker_loop(
    worker_id: usize,
    injector: Arc<Injector<CompileJob>>,
    local: Worker<CompileJob>,
    notify_rx: chan::Receiver<()>,
    queued: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    result_tx: chan::Sender<WorkerResult>,
    _config: &RuntimeConfig,
) {
    debug!(worker_id, "compile worker started (no LLVM, all jobs will fail)");

    loop {
        let Some(job) = steal_job(&injector, &local, &queued) else {
            if shutdown.load(Ordering::Acquire) && queued.load(Ordering::Acquire) == 0 {
                break;
            }
            match notify_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(()) | Err(chan::RecvTimeoutError::Timeout) => continue,
                Err(chan::RecvTimeoutError::Disconnected) => {
                    if queued.load(Ordering::Acquire) == 0 {
                        break;
                    }
                }
            }
            continue;
        };

        let _ = result_tx.send(WorkerResult {
            key: job.key,
            outcome: Err("LLVM backend not available".into()),
            kind: job.kind,
            sync_notifier: job.sync_notifier,
            generation: job.generation,
            compile_duration: Duration::ZERO,
            timings: CompileTimings::default(),
        });
    }
}

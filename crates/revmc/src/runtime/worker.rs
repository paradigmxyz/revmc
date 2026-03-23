//! Background JIT and AOT compilation workers.
//!
//! Each worker thread owns a long-lived `EvmCompiler` instance tied to its
//! thread-local LLVM context. Compiled function pointers remain valid as long
//! as the worker (and its backing compiler) is alive.
//!
//! The worker thread will not exit until all `Arc<WorkerBacking>` references
//! for that worker have been dropped, ensuring function pointers remain valid.

use crate::{EvmCompilerFn, runtime::storage::RuntimeCacheKey};
use alloy_primitives::Bytes;
use crossbeam_channel as chan;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex},
};

/// Notifier for synchronous compilation requests.
///
/// Wraps an optional sender that is notified when the compilation
/// completes (success or failure). Passed from the coordinator through
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

/// A compilation job sent from the coordinator to a worker.
#[derive(Debug)]
pub(crate) enum WorkerJob {
    /// JIT compilation: produce an in-memory function pointer.
    Jit(JitJob),
    /// AOT compilation: produce shared-library bytes for persistence.
    Aot(AotJob),
}

/// A JIT compilation job sent from the coordinator to a worker.
#[derive(derive_more::Debug)]
pub(crate) struct JitJob {
    /// The key to compile for.
    pub(crate) key: RuntimeCacheKey,
    /// The raw bytecode to compile.
    pub(crate) bytecode: Bytes,
    /// The symbol name to use for the compiled function.
    pub(crate) symbol_name: String,
    /// Optional notifier for synchronous callers.
    #[debug(skip)]
    pub(crate) sync_notifier: SyncNotifier,
}

/// An AOT compilation job sent from the coordinator to a worker.
#[derive(Debug)]
pub(crate) struct AotJob {
    /// The key to compile for.
    pub(crate) key: RuntimeCacheKey,
    /// The raw bytecode to compile.
    pub(crate) bytecode: Bytes,
    /// The symbol name to use for the compiled function.
    pub(crate) symbol_name: String,
    /// Optimization level for AOT compilation.
    pub(crate) opt_level: crate::OptimizationLevel,
}

/// Result of a compilation attempt, sent back from a worker to the coordinator.
pub(crate) struct WorkerResult {
    /// The key that was compiled.
    pub(crate) key: RuntimeCacheKey,
    /// The worker that produced this result (used to get its backing Arc).
    pub(crate) worker_id: usize,
    /// The compilation outcome.
    pub(crate) outcome: Result<WorkerSuccess, String>,
    /// Optional notifier for synchronous callers, passed through from the job.
    pub(crate) sync_notifier: SyncNotifier,
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
    /// Approximate size of the compiled code in bytes.
    pub(crate) approx_size_bytes: usize,
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

/// Handle to the worker pool. Manages worker threads and their job queues.
pub(crate) struct WorkerPool {
    /// Per-worker job senders.
    job_txs: Vec<chan::Sender<WorkerJob>>,
    /// Worker thread handles.
    threads: Vec<Option<std::thread::JoinHandle<()>>>,
    /// Shared backing owners — one per worker, keeps JIT code alive.
    backings: Vec<Arc<WorkerBacking>>,
    /// Round-robin index for job distribution.
    next_worker: usize,
}

/// Backing that keeps a worker's JIT-compiled code alive.
///
/// The worker thread blocks on a condvar until all external `Arc<WorkerBacking>`
/// references are dropped, ensuring the `EvmCompiler` (and its machine code)
/// stays alive on the worker thread's stack.
pub(crate) struct WorkerBacking {
    /// Signals the worker thread that it's safe to exit.
    exit_signal: Condvar,
    /// Protected by mutex: `true` when the worker should exit.
    should_exit: Mutex<bool>,
}

impl WorkerBacking {
    fn new() -> Self {
        Self { exit_signal: Condvar::new(), should_exit: Mutex::new(false) }
    }

    /// Signals the worker that it's safe to exit.
    fn signal_exit(&self) {
        *self.should_exit.lock().unwrap() = true;
        self.exit_signal.notify_one();
    }

    /// Blocks until signaled to exit.
    fn wait_for_exit(&self) {
        let mut should_exit = self.should_exit.lock().unwrap();
        while !*should_exit {
            should_exit = self.exit_signal.wait(should_exit).unwrap();
        }
    }
}

impl WorkerPool {
    /// Creates and starts the worker pool.
    pub(crate) fn new(
        worker_count: usize,
        job_queue_capacity: usize,
        result_tx: chan::Sender<WorkerResult>,
        opt_level: crate::OptimizationLevel,
        dump_dir: Option<PathBuf>,
    ) -> Self {
        let mut job_txs = Vec::with_capacity(worker_count);
        let mut threads = Vec::with_capacity(worker_count);
        let mut backings = Vec::with_capacity(worker_count);

        for worker_id in 0..worker_count {
            let (job_tx, job_rx) = chan::bounded::<WorkerJob>(job_queue_capacity);
            let result_tx = result_tx.clone();
            let backing = Arc::new(WorkerBacking::new());
            let backing_for_worker = Arc::clone(&backing);
            let dump_dir = dump_dir.clone();

            let thread = std::thread::Builder::new()
                .name(format!("revmc-{worker_id:02}"))
                .spawn(move || {
                    worker_loop(
                        worker_id,
                        job_rx,
                        result_tx,
                        opt_level,
                        &backing_for_worker,
                        dump_dir.as_deref(),
                    );
                })
                .expect("failed to spawn compile worker");

            job_txs.push(job_tx);
            threads.push(Some(thread));
            backings.push(backing);
        }

        Self { job_txs, threads, backings, next_worker: 0 }
    }

    /// Tries to send a job to a worker (round-robin). Returns false if all queues are full.
    pub(crate) fn try_send(&mut self, mut job: WorkerJob) -> bool {
        let count = self.job_txs.len();
        for _ in 0..count {
            let idx = self.next_worker % count;
            self.next_worker = self.next_worker.wrapping_add(1);
            match self.job_txs[idx].try_send(job) {
                Ok(()) => return true,
                Err(chan::TrySendError::Full(j)) => job = j,
                Err(chan::TrySendError::Disconnected(_)) => return false,
            }
        }
        false
    }

    /// Returns the backing Arc for the given worker, used to keep JIT code alive.
    pub(crate) fn backing(&self, worker_id: usize) -> Arc<WorkerBacking> {
        Arc::clone(&self.backings[worker_id])
    }

    /// Shuts down all workers by dropping job senders and signaling exit.
    pub(crate) fn shutdown(&mut self) {
        // Drop all job senders so workers exit their recv loops.
        self.job_txs.clear();

        // Signal each worker that it's safe to exit (no more external refs
        // will be created). The worker will wait until all existing
        // Arc<WorkerBacking> refs are dropped before actually exiting.
        for backing in &self.backings {
            backing.signal_exit();
        }

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

/// The per-worker event loop. Owns a long-lived JIT compiler.
///
/// After all jobs are processed, the worker blocks until all `Arc<WorkerBacking>`
/// references are dropped, ensuring compiled function pointers remain valid.
#[cfg(feature = "llvm")]
fn worker_loop(
    worker_id: usize,
    job_rx: chan::Receiver<WorkerJob>,
    result_tx: chan::Sender<WorkerResult>,
    opt_level: crate::OptimizationLevel,
    backing: &WorkerBacking,
    dump_dir: Option<&Path>,
) {
    use crate::{EvmCompiler, EvmLlvmBackend};

    debug!(worker_id, "compile worker started");

    let backend = match EvmLlvmBackend::new(false, opt_level) {
        Ok(b) => b,
        Err(e) => {
            error!(worker_id, error = %e, "failed to create LLVM backend, worker exiting");
            return;
        }
    };
    let mut jit_compiler = EvmCompiler::new(backend);

    while let Ok(job) = job_rx.recv() {
        debug!(?job, "received job");
        let (key, outcome, sync_notifier) = match job {
            WorkerJob::Jit(job) => {
                let _span =
                    debug_span!("jit_compile", hash=%job.key.code_hash, spec_id=?job.key.spec_id)
                        .entered();

                if let Some(base) = dump_dir {
                    let dir = base
                        .join(format!("{:?}", job.key.spec_id))
                        .join(format!("{}", job.key.code_hash));
                    jit_compiler.set_dump_to(Some(dir));
                }

                let result = unsafe {
                    jit_compiler.jit(&job.symbol_name, &job.bytecode[..], job.key.spec_id)
                };

                let outcome = match result {
                    Ok(func) => {
                        // Reset IR so the next job can reuse this compiler.
                        // This frees IR but keeps JIT machine code alive.
                        if let Err(e) = jit_compiler.clear_ir() {
                            warn!(worker_id, error = %e, "clear_ir failed");
                        }
                        debug!(worker_id, "JIT compilation succeeded");
                        Ok(WorkerSuccess::Jit(JitSuccess {
                            func,
                            approx_size_bytes: job.bytecode.len(),
                        }))
                    }
                    Err(e) => {
                        warn!(worker_id, error = %e, "JIT compilation failed");
                        Err(format!("{e}"))
                    }
                };
                (job.key, outcome, job.sync_notifier)
            }
            WorkerJob::Aot(job) => {
                let _span =
                    debug_span!("aot_compile", hash=%job.key.code_hash, spec_id=?job.key.spec_id)
                        .entered();

                let outcome = compile_aot_artifact(&job);
                (job.key, outcome, SyncNotifier::none())
            }
        };

        let _ = result_tx.send(WorkerResult { key, worker_id, outcome, sync_notifier });
    }

    debug!(worker_id, "compile worker done processing jobs, waiting for backing refs to drop");

    // Block until the coordinator signals exit (all external program refs dropped).
    // This keeps the compiler (and its JIT machine code) alive on this thread.
    backing.wait_for_exit();

    debug!(worker_id, "compile worker shutting down");
    // `jit_compiler` is dropped here, freeing all JIT machine code.
}

/// Compiles a single bytecode to a shared library and returns the raw bytes.
#[cfg(feature = "llvm")]
fn compile_aot_artifact(job: &AotJob) -> Result<WorkerSuccess, String> {
    use crate::{EvmCompiler, EvmLlvmBackend, Linker};
    use std::io::Read;

    let backend = EvmLlvmBackend::new(true, job.opt_level)
        .map_err(|e| format!("AOT backend creation failed: {e}"))?;
    let mut compiler = EvmCompiler::new(backend);

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
    std::fs::File::open(&so_path)
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
fn worker_loop(
    worker_id: usize,
    job_rx: chan::Receiver<WorkerJob>,
    result_tx: chan::Sender<WorkerResult>,
    _opt_level: crate::OptimizationLevel,
    backing: &WorkerBacking,
    _dump_dir: Option<&Path>,
) {
    debug!(worker_id, "compile worker started (no LLVM, all jobs will fail)");

    while let Ok(job) = job_rx.recv() {
        let (key, sync_notifier) = match job {
            WorkerJob::Jit(j) => (j.key, j.sync_notifier),
            WorkerJob::Aot(j) => (j.key, SyncNotifier::none()),
        };
        let _ = result_tx.send(WorkerResult {
            key,
            worker_id,
            outcome: Err("LLVM backend not available".into()),
            sync_notifier,
        });
    }

    backing.wait_for_exit();
}

//! Coordinator thread: single-threaded event loop for runtime state management.
//!
//! The coordinator is the sole writer of mutable state. It processes lookup-observed
//! events, tracks hotness, admits JIT compilation jobs, and inserts results into
//! the shared resident `DashMap`.

use crate::runtime::{
    api::{CompiledProgram, LoadedLibrary},
    config::RuntimeTuning,
    storage::{ArtifactKey, ArtifactManifest, ArtifactStore, BackendSelection, RuntimeCacheKey},
    worker::{AotJob, JitJob, SyncNotifier, WorkerJob, WorkerPool, WorkerResult, WorkerSuccess},
};
use alloy_primitives::{Bytes, keccak256, map::HashMap};
use crossbeam_channel as chan;
use dashmap::DashMap;
use revmc_backend::Target;
use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

/// The resident map type: code_hash+spec_id → compiled program.
pub(crate) type ResidentMap = DashMap<RuntimeCacheKey, Arc<CompiledProgram>>;

/// Commands sent to the coordinator thread.
pub(crate) enum Command {
    /// A lookup was observed on the hot path.
    LookupObserved(LookupObservedEvent),
    /// Explicit request to JIT-compile a bytecode.
    CompileJit(CompileJitRequest),
    /// Explicit request to prepare AOT artifacts.
    PrepareAot(Vec<PrepareAotRequest>),
    /// Clear the resident compiled map.
    ClearResident,
    /// Clear persisted artifacts from the artifact store.
    ClearPersisted,
    /// Clear both resident and persisted.
    ClearAll,
    /// Shut down the coordinator.
    Shutdown,
}

/// An explicit JIT compilation request.
pub(crate) struct CompileJitRequest {
    /// The key to compile for.
    pub(crate) key: RuntimeCacheKey,
    /// The raw bytecode.
    pub(crate) bytecode: Bytes,
    /// Optional notifier for synchronous callers.
    pub(crate) sync_notifier: SyncNotifier,
}

/// An explicit AOT preparation request.
pub(crate) struct PrepareAotRequest {
    /// The key to compile for.
    pub(crate) key: RuntimeCacheKey,
    /// The raw bytecode.
    pub(crate) bytecode: Bytes,
}

/// A lookup-observed event.
pub(crate) struct LookupObservedEvent {
    /// The key that was looked up.
    pub(crate) key: RuntimeCacheKey,
    /// Whether the lookup was a hit (compiled found).
    pub(crate) was_hit: bool,
    /// The bytecode, present only on misses.
    pub(crate) bytecode: Option<Bytes>,
}

/// Per-key state tracked by the coordinator.
struct EntryState {
    /// Number of observed misses.
    hotness: u32,
    /// Current phase.
    phase: EntryPhase,
    /// The bytecode for this key (captured from a miss event).
    bytecode: Bytes,
}

/// Phase of a coordinator entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryPhase {
    /// Not yet hot enough for JIT.
    Cold,
    /// JIT compilation in progress on a worker.
    Working,
    /// JIT compilation failed.
    Failed,
}

/// All coordinator-owned mutable state.
struct CoordinatorState {
    /// The shared resident map (handles read, coordinator writes).
    resident: Arc<ResidentMap>,
    /// Per-key tracking state (coordinator-only).
    entries: HashMap<RuntimeCacheKey, EntryState>,
    /// Worker pool for JIT compilation.
    workers: WorkerPool,
    /// Receiver for worker results.
    result_rx: chan::Receiver<WorkerResult>,
    /// Artifact store for persisted artifacts.
    store: Option<Arc<dyn ArtifactStore>>,
    /// Tuning knobs.
    tuning: RuntimeTuning,
    /// Number of keys currently in Working phase.
    pending_jobs: usize,
    /// JIT stats.
    jit_promotions: u64,
    jit_successes: u64,
    jit_failures: u64,
}

impl CoordinatorState {
    fn handle_lookup_observed(&mut self, event: LookupObservedEvent) {
        // Hits don't need any action.
        if event.was_hit {
            return;
        }

        // Already in the resident map (may have been inserted since the event was emitted).
        if self.resident.contains_key(&event.key) {
            return;
        }

        let bytecode = match event.bytecode {
            Some(b) => b,
            None => return,
        };

        // Skip empty bytecodes.
        if bytecode.is_empty() {
            return;
        }

        let entry = self.entries.entry(event.key.clone()).or_insert_with(|| EntryState {
            hotness: 0,
            phase: EntryPhase::Cold,
            bytecode: bytecode.clone(),
        });

        // Only increment hotness for cold entries.
        if entry.phase != EntryPhase::Cold {
            return;
        }

        entry.hotness = entry.hotness.saturating_add(1);

        if entry.hotness >= self.tuning.jit_hot_threshold
            && self.pending_jobs < self.tuning.max_pending_jit_jobs
        {
            debug!(
                code_hash = %event.key.code_hash,
                spec_id = ?event.key.spec_id,
                hotness = entry.hotness,
                "promoting hot key to JIT compilation",
            );
            let symbol = format!("jit_{:x}_{:?}", event.key.code_hash, event.key.spec_id);
            let job = WorkerJob::Jit(JitJob {
                key: event.key,
                bytecode: entry.bytecode.clone(),
                symbol_name: symbol,
                sync_notifier: SyncNotifier::none(),
            });

            if self.workers.try_send(job) {
                entry.phase = EntryPhase::Working;
                self.pending_jobs += 1;
                self.jit_promotions += 1;
            }
        }
    }

    fn handle_compile_jit(&mut self, req: CompileJitRequest) {
        // Already compiled — notify and return.
        if self.resident.contains_key(&req.key) {
            debug!(code_hash = %req.key.code_hash, "compile_jit: already resident");
            req.sync_notifier.notify();
            return;
        }

        // Skip empty bytecodes.
        if req.bytecode.is_empty() {
            req.sync_notifier.notify();
            return;
        }

        // Check if already working or failed.
        if let Some(entry) = self.entries.get(&req.key)
            && entry.phase != EntryPhase::Cold
        {
            req.sync_notifier.notify();
            return;
        }

        let code_hash = req.key.code_hash;
        let symbol = format!("jit_{:x}_{:?}", req.key.code_hash, req.key.spec_id);
        let job = WorkerJob::Jit(JitJob {
            key: req.key.clone(),
            bytecode: req.bytecode.clone(),
            symbol_name: symbol,
            sync_notifier: req.sync_notifier,
        });

        let entry = self.entries.entry(req.key).or_insert_with(|| EntryState {
            hotness: 0,
            phase: EntryPhase::Cold,
            bytecode: req.bytecode,
        });

        if self.workers.try_send(job) {
            debug!(
                %code_hash,
                pending_jobs = self.pending_jobs + 1,
                "compile_jit: dispatched to worker",
            );
            entry.phase = EntryPhase::Working;
            self.pending_jobs += 1;
            self.jit_promotions += 1;
        }
    }

    fn handle_prepare_aot(&mut self, reqs: Vec<PrepareAotRequest>) {
        for req in reqs {
            // Already compiled in resident map.
            if self.resident.contains_key(&req.key) {
                continue;
            }

            // Skip empty bytecodes.
            if req.bytecode.is_empty() {
                continue;
            }

            // Check if already working or failed.
            if let Some(entry) = self.entries.get(&req.key)
                && entry.phase != EntryPhase::Cold
            {
                continue;
            }

            let symbol = format!("aot_{:x}_{:?}", req.key.code_hash, req.key.spec_id);
            let job = WorkerJob::Aot(AotJob {
                key: req.key.clone(),
                bytecode: req.bytecode.clone(),
                symbol_name: symbol,
                opt_level: self.tuning.aot_opt_level,
            });

            let entry = self.entries.entry(req.key).or_insert_with(|| EntryState {
                hotness: 0,
                phase: EntryPhase::Cold,
                bytecode: req.bytecode,
            });

            if self.workers.try_send(job) {
                entry.phase = EntryPhase::Working;
                self.pending_jobs += 1;
                self.jit_promotions += 1;
            }
        }
    }

    fn handle_clear_resident(&mut self) {
        self.resident.clear();
        self.entries.clear();
        self.pending_jobs = 0;
        debug!("resident map cleared");
    }

    fn handle_clear_persisted(&mut self) {
        if let Some(store) = &self.store {
            if let Err(e) = store.clear() {
                warn!(error = %e, "failed to clear artifact store");
            } else {
                debug!("artifact store cleared");
            }
        }
    }

    fn handle_clear_all(&mut self) {
        self.handle_clear_resident();
        self.handle_clear_persisted();
    }

    fn handle_worker_result(&mut self, result: WorkerResult) {
        self.pending_jobs = self.pending_jobs.saturating_sub(1);

        match result.outcome {
            Ok(WorkerSuccess::Jit(success)) => {
                let backing = self.workers.backing(result.worker_id);
                let program =
                    Arc::new(CompiledProgram::new_jit(result.key.clone(), success.func, backing));

                self.resident.insert(result.key.clone(), program);
                self.entries.remove(&result.key);
                self.jit_successes += 1;

                debug!(
                    code_hash = %result.key.code_hash,
                    spec_id = ?result.key.spec_id,
                    "JIT program published to resident map",
                );
            }
            Ok(WorkerSuccess::Aot(success)) => {
                self.handle_aot_success(result.key.clone(), success);
            }
            Err(err) => {
                if let Some(entry) = self.entries.get_mut(&result.key) {
                    entry.phase = EntryPhase::Failed;
                }
                self.jit_failures += 1;

                warn!(
                    code_hash = %result.key.code_hash,
                    error = %err,
                    "compilation failed",
                );
            }
        }

        // Notify synchronous callers after the result has been fully processed.
        result.sync_notifier.notify();
    }

    fn handle_aot_success(
        &mut self,
        key: RuntimeCacheKey,
        success: crate::runtime::worker::AotSuccess,
    ) {
        let artifact_key = ArtifactKey {
            runtime: key.clone(),
            backend: BackendSelection::Llvm,
            target: Target::Native,
            opt_level: self.tuning.aot_opt_level,
        };

        let sha256 = keccak256(&success.dylib_bytes).0;

        let manifest = ArtifactManifest {
            artifact_key: artifact_key.clone(),
            symbol_name: success.symbol_name.clone(),
            bytecode_len: success.bytecode_len,
            artifact_len: success.dylib_bytes.len(),
            created_at_unix_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            sha256,
        };

        // Persist to store if available.
        if let Some(store) = &self.store {
            if let Err(e) = store.store(&artifact_key, &manifest, &success.dylib_bytes) {
                warn!(
                    code_hash = %key.code_hash,
                    error = %e,
                    "failed to persist AOT artifact",
                );
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.phase = EntryPhase::Failed;
                }
                self.jit_failures += 1;
                return;
            }

            debug!(
                code_hash = %key.code_hash,
                spec_id = ?key.spec_id,
                dylib_len = success.dylib_bytes.len(),
                "AOT artifact persisted to store",
            );

            // Load from store to get the canonical path, then dlopen.
            match store.load(&artifact_key) {
                Ok(Some(stored)) => {
                    match (|| -> crate::eyre::Result<CompiledProgram> {
                        let library = unsafe { libloading::Library::new(&stored.dylib_path) }
                            .map_err(|e| {
                                crate::eyre::eyre!("dlopen {:?}: {e}", stored.dylib_path)
                            })?;
                        let func: crate::EvmCompilerFn = unsafe {
                            let sym: libloading::Symbol<'_, crate::EvmCompilerFn> =
                                library.get(success.symbol_name.as_bytes()).map_err(|e| {
                                    crate::eyre::eyre!("symbol '{}': {e}", success.symbol_name)
                                })?;
                            *sym
                        };
                        let library = Arc::new(LoadedLibrary::new(library));
                        Ok(CompiledProgram::new_aot(key.clone(), func, library))
                    })() {
                        Ok(program) => {
                            self.resident.insert(key.clone(), Arc::new(program));
                            self.entries.remove(&key);
                            self.jit_successes += 1;

                            debug!(
                                code_hash = %key.code_hash,
                                spec_id = ?key.spec_id,
                                "AOT program loaded into resident map",
                            );
                        }
                        Err(e) => {
                            warn!(
                                code_hash = %key.code_hash,
                                error = %e,
                                "failed to load persisted AOT artifact",
                            );
                            // Persisted successfully but couldn't load — mark as failed.
                            if let Some(entry) = self.entries.get_mut(&key) {
                                entry.phase = EntryPhase::Failed;
                            }
                            self.jit_failures += 1;
                        }
                    }
                }
                Ok(None) => {
                    warn!(
                        code_hash = %key.code_hash,
                        "stored AOT artifact not found on reload",
                    );
                    if let Some(entry) = self.entries.get_mut(&key) {
                        entry.phase = EntryPhase::Failed;
                    }
                    self.jit_failures += 1;
                }
                Err(e) => {
                    warn!(
                        code_hash = %key.code_hash,
                        error = %e,
                        "failed to reload persisted AOT artifact",
                    );
                    if let Some(entry) = self.entries.get_mut(&key) {
                        entry.phase = EntryPhase::Failed;
                    }
                    self.jit_failures += 1;
                }
            }
        } else {
            // No store configured — can't persist, mark as failed.
            warn!(
                code_hash = %key.code_hash,
                "AOT compilation completed but no artifact store configured",
            );
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.phase = EntryPhase::Failed;
            }
            self.jit_failures += 1;
        }
    }
}

/// Runs the coordinator event loop. Called on the coordinator thread.
pub(crate) fn run(
    cmd_rx: chan::Receiver<Command>,
    resident: Arc<ResidentMap>,
    store: Option<Arc<dyn ArtifactStore>>,
    tuning: RuntimeTuning,
    dump_dir: Option<std::path::PathBuf>,
) {
    debug!("coordinator thread started");

    let (result_tx, result_rx) = chan::unbounded::<WorkerResult>();

    let workers = WorkerPool::new(
        tuning.jit_worker_count,
        tuning.jit_worker_queue_capacity,
        result_tx,
        tuning.jit_opt_level,
        dump_dir,
    );

    let mut state = CoordinatorState {
        resident,
        entries: HashMap::default(),
        workers,
        result_rx,
        store,
        tuning,
        pending_jobs: 0,
        jit_promotions: 0,
        jit_successes: 0,
        jit_failures: 0,
    };

    loop {
        chan::select! {
            recv(cmd_rx) -> msg => {
                let cmd = match msg {
                    Ok(cmd) => cmd,
                    Err(_) => {
                        debug!("coordinator channel closed, shutting down");
                        break;
                    }
                };
                match cmd {
                    Command::LookupObserved(event) => state.handle_lookup_observed(event),
                    Command::CompileJit(req) => state.handle_compile_jit(req),
                    Command::PrepareAot(reqs) => state.handle_prepare_aot(reqs),
                    Command::ClearResident => state.handle_clear_resident(),
                    Command::ClearPersisted => state.handle_clear_persisted(),
                    Command::ClearAll => state.handle_clear_all(),
                    Command::Shutdown => {
                        debug!("coordinator shutting down");
                        break;
                    }
                }
            }
            recv(state.result_rx) -> msg => {
                if let Ok(result) = msg {
                    state.handle_worker_result(result);
                }
            }
        }
    }

    // Drain remaining worker results before shutdown.
    while let Ok(result) = state.result_rx.try_recv() {
        state.handle_worker_result(result);
    }

    info!(
        jit_promotions = state.jit_promotions,
        jit_successes = state.jit_successes,
        jit_failures = state.jit_failures,
        resident_entries = state.resident.len(),
        "coordinator stats at shutdown",
    );

    state.workers.shutdown();
}

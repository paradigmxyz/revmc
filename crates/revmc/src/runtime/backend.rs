//! Backend thread: single-threaded event loop for runtime state management.
//!
//! Hotness is tracked on the backend thread. The lookup hot path is push-only:
//! it pushes a [`LookupObservedEvent`] onto a lock-free queue and returns. There is no wakeup
//! signal to the backend, so sends are essentially free (one CAS, occasional segment alloc).
//!
//! The backend drains the queue on every iteration of its event loop, alongside
//! processing explicit user commands and worker results.

use crate::runtime::{
    api::{CompiledProgram, LoadedLibrary, ProgramKind},
    config::{CompilationEvent, RuntimeConfig, RuntimeTuning},
    storage::{ArtifactKey, ArtifactManifest, ArtifactStore, BackendSelection, RuntimeCacheKey},
    worker::{AotJob, JitJob, SyncNotifier, WorkerJob, WorkerPool, WorkerResult, WorkerSuccess},
};
use alloy_primitives::{
    Bytes, keccak256,
    map::{DefaultHashBuilder, HashMap},
};
use crossbeam_channel as chan;
use crossbeam_queue::ArrayQueue;
use dashmap::DashMap;
use quanta::Instant;
use std::{
    ops::ControlFlow,
    sync::{Arc, atomic::Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

/// The resident map type: code_hash+spec_id → compiled program.
pub(crate) type ResidentMap = DashMap<RuntimeCacheKey, Arc<CompiledProgram>, DefaultHashBuilder>;

/// Bounded MPMC lock-free queue of lookup-observed events.
///
/// Producers (lookup hot path) push without blocking; on overflow the event
/// is silently dropped (`stats.events_dropped` is bumped). The backend
/// drains via `pop` on every loop iteration. Hotness signal is best-effort.
pub(crate) type EventQueue = ArrayQueue<LookupObservedEvent>;

/// Per-entry metadata tracked alongside the resident map for eviction decisions.
struct ResidentMeta {
    /// When this entry was last hit by a lookup.
    last_hit_at: Instant,
}

/// Returns the total bytes of JIT-allocated memory via the memory plugin.
fn jit_total_bytes() -> usize {
    #[cfg(feature = "llvm")]
    {
        crate::llvm::jit_memory_usage().map(|u| u.total_bytes()).unwrap_or(0)
    }
    #[cfg(not(feature = "llvm"))]
    {
        0
    }
}

/// Commands sent to the backend thread on the bounded command channel.
///
/// Lookup-observed events are NOT carried here — they go through the
/// [`EventQueue`] to avoid waking the backend on every lookup.
pub(crate) enum Command {
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
    /// Shut down the backend.
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

/// A lookup-observed event pushed by the hot path.
///
/// `Bytes` clone is cheap (single `Arc` bump), so the bytecode is always carried
/// for misses; the backend uses it directly when the hotness threshold trips.
pub(crate) struct LookupObservedEvent {
    /// The key that was looked up.
    pub(crate) key: RuntimeCacheKey,
    /// The bytecode, present only on misses.
    pub(crate) bytecode: Option<Bytes>,
}

/// Per-key state tracked by the backend.
struct EntryState {
    /// Number of observed misses.
    hotness: u32,
    /// Current phase.
    phase: EntryPhase,
    /// The bytecode for this key (captured from a miss event).
    bytecode: Bytes,
    /// Sync notifiers waiting for this entry to finish compiling.
    pending_notifiers: Vec<SyncNotifier>,
}

/// Phase of a backend entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryPhase {
    /// Not yet hot enough for JIT.
    Cold,
    /// JIT compilation in progress on a worker.
    Working,
}

/// Whether a JIT admission request was triggered by hot-path observation
/// (gated on hotness + cold-entry cap) or by an explicit user request
/// (unconditional, may carry a sync notifier).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdmitMode {
    Observed,
    Explicit,
}

/// All backend-thread-owned mutable state.
struct BackendState {
    /// Shared state (resident map, event queue, stats).
    inner: Arc<super::BackendShared>,
    /// Per-key metadata for eviction (backend-only).
    resident_meta: HashMap<RuntimeCacheKey, ResidentMeta>,
    /// Per-key tracking state (backend-only).
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
    /// Monotonically increasing generation counter, bumped on clear/invalidation.
    generation: u64,
    /// Last time an eviction sweep was run.
    last_sweep: Instant,
    /// Optional user callback for compilation events.
    on_compilation: Option<Arc<dyn Fn(CompilationEvent) + Send + Sync>>,
}

impl BackendState {
    fn handle(&mut self, cmd: Command) -> ControlFlow<()> {
        match cmd {
            Command::CompileJit(req) => self.handle_compile_jit(req),
            Command::PrepareAot(reqs) => self.handle_prepare_aot(reqs),
            Command::ClearResident => self.handle_clear_resident(),
            Command::ClearPersisted => self.handle_clear_persisted(),
            Command::ClearAll => self.handle_clear_all(),
            Command::Shutdown => return ControlFlow::Break(()),
        }
        ControlFlow::Continue(())
    }

    fn tick(&mut self) {
        self.drain_events();
        self.run_eviction_sweep();
    }

    /// Drains all currently-queued lookup events.
    fn drain_events(&mut self) {
        // Cap per-iteration drain so a flood of events can't starve other
        // work (commands, worker results, sweeps). Surplus events stay in the
        // queue and are picked up next iteration.
        for _ in 0..self.tuning.max_events_per_drain {
            let Some(event) = self.inner.events.pop() else { break };
            self.handle_lookup_observed(event);
        }
    }

    fn handle_lookup_observed(&mut self, event: LookupObservedEvent) {
        match event.bytecode {
            Some(bytecode) => {
                self.inner.stats.lookup_misses.fetch_add(1, Ordering::Relaxed);
                self.try_admit_jit(event.key, bytecode, SyncNotifier::none(), AdmitMode::Observed);
            }
            None => {
                self.inner.stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
                if let Some(meta) = self.resident_meta.get_mut(&event.key) {
                    meta.last_hit_at = Instant::now();
                }
            }
        }
    }

    fn handle_compile_jit(&mut self, req: CompileJitRequest) {
        self.try_admit_jit(req.key, req.bytecode, req.sync_notifier, AdmitMode::Explicit);
    }

    /// Common admission path for JIT compilation requests.
    ///
    /// Handles the cold→working state machine, hotness gating, in-flight
    /// dedup, and worker dispatch. The two callers differ only in whether
    /// promotion is gated by hotness ([`AdmitMode::Observed`]) or
    /// unconditional ([`AdmitMode::Explicit`]).
    fn try_admit_jit(
        &mut self,
        key: RuntimeCacheKey,
        bytecode: Bytes,
        sync_notifier: SyncNotifier,
        mode: AdmitMode,
    ) {
        // Already resident — nothing to do; notify any sync waiter.
        if self.inner.resident.contains_key(&key) {
            sync_notifier.notify();
            return;
        }

        // Skip empty / oversize bytecodes.
        if !self.tuning.should_compile(&bytecode) {
            sync_notifier.notify();
            return;
        }

        // Cap cold-entry count for the observed path to bound memory growth
        // from miss tracking. Explicit requests bypass this cap.
        if matches!(mode, AdmitMode::Observed) {
            let max_entries = self.tuning.jit_max_pending_jobs * 10;
            if !self.entries.contains_key(&key) && self.entries.len() >= max_entries {
                return;
            }
        }

        let entry = self.entries.entry(key).or_insert_with(|| EntryState {
            hotness: 0,
            phase: EntryPhase::Cold,
            bytecode: bytecode.clone(),
            pending_notifiers: Vec::new(),
        });

        // Already in flight: dedup. Observed has no notifier (None push is
        // a harmless no-op when drained); explicit waiters wake on result.
        if entry.phase == EntryPhase::Working {
            entry.pending_notifiers.push(sync_notifier);
            return;
        }

        // Observed path: count hotness and gate on threshold.
        if matches!(mode, AdmitMode::Observed) {
            entry.hotness = entry.hotness.saturating_add(1);
            if (entry.hotness as usize) < self.tuning.jit_hot_threshold {
                return;
            }
        }

        // Cap in-flight jobs.
        if self.pending_jobs >= self.tuning.jit_max_pending_jobs {
            // Observed: leave entry Cold so we retry on next miss.
            // Explicit: notify so the caller doesn't hang.
            sync_notifier.notify();
            return;
        }

        let symbol = format!("jit_{:x}_{:?}", key.code_hash, key.spec_id);
        let job = WorkerJob::Jit(JitJob {
            key,
            bytecode: entry.bytecode.clone(),
            symbol_name: symbol,
            sync_notifier,
            generation: self.generation,
        });

        match self.workers.try_send(job) {
            Ok(()) => {
                debug!(
                    code_hash = %key.code_hash,
                    spec_id = ?key.spec_id,
                    hotness = entry.hotness,
                    pending_jobs = self.pending_jobs + 1,
                    "dispatched JIT compilation",
                );
                entry.phase = EntryPhase::Working;
                self.pending_jobs += 1;
                self.inner.stats.compilations_dispatched.fetch_add(1, Ordering::Relaxed);
            }
            Err(WorkerJob::Jit(job)) => {
                // Worker queue saturated — notify caller immediately so it doesn't hang.
                warn!(code_hash = %key.code_hash, "worker pool saturated, dropping request");
                job.sync_notifier.notify();
            }
            Err(WorkerJob::Aot(_)) => unreachable!("sent Jit job"),
        }
    }

    fn handle_prepare_aot(&mut self, reqs: Vec<PrepareAotRequest>) {
        for req in reqs {
            // Already compiled in resident map.
            if self.inner.resident.contains_key(&req.key) {
                continue;
            }

            // Skip empty / oversize bytecodes.
            if !self.tuning.should_compile(&req.bytecode) {
                continue;
            }

            // Check if already working.
            if let Some(entry) = self.entries.get(&req.key)
                && entry.phase == EntryPhase::Working
            {
                continue;
            }

            // Probe the store: if already persisted, load directly without recompiling.
            if self.try_load_persisted_aot(&req.key) {
                continue;
            }

            let symbol = format!("aot_{:x}_{:?}", req.key.code_hash, req.key.spec_id);
            let job = WorkerJob::Aot(AotJob {
                key: req.key,
                bytecode: req.bytecode.clone(),
                symbol_name: symbol,
                opt_level: self.tuning.aot_opt_level,
                generation: self.generation,
            });

            let entry = self.entries.entry(req.key).or_insert_with(|| EntryState {
                hotness: 0,
                phase: EntryPhase::Cold,
                bytecode: req.bytecode,
                pending_notifiers: Vec::new(),
            });

            if self.workers.try_send(job).is_ok() {
                entry.phase = EntryPhase::Working;
                self.pending_jobs += 1;
                self.inner.stats.compilations_dispatched.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Tries to load an already-persisted AOT artifact from the store into the resident map.
    /// Returns `true` if the artifact was loaded successfully.
    fn try_load_persisted_aot(&mut self, key: &RuntimeCacheKey) -> bool {
        let store = match &self.store {
            Some(s) => s,
            None => return false,
        };

        let artifact_key = ArtifactKey {
            runtime: *key,
            backend: BackendSelection::Llvm,
            opt_level: self.tuning.aot_opt_level,
        };

        match store.load(&artifact_key) {
            Ok(Some(stored)) => {
                match (|| -> crate::eyre::Result<CompiledProgram> {
                    let library = unsafe { libloading::Library::new(&stored.dylib_path) }
                        .map_err(|e| crate::eyre::eyre!("dlopen {:?}: {e}", stored.dylib_path))?;
                    let func: crate::EvmCompilerFn = unsafe {
                        let sym: libloading::Symbol<'_, crate::EvmCompilerFn> =
                            library.get(stored.manifest.symbol_name.as_bytes()).map_err(|e| {
                                crate::eyre::eyre!("symbol '{}': {e}", stored.manifest.symbol_name)
                            })?;
                        *sym
                    };
                    let library = Arc::new(LoadedLibrary::new(library));
                    Ok(CompiledProgram::new_aot(*key, func, library))
                })() {
                    Ok(program) => {
                        debug!(
                            code_hash = %key.code_hash,
                            spec_id = ?key.spec_id,
                            "loaded existing AOT artifact from store, skipping recompilation",
                        );
                        self.insert_resident(*key, Arc::new(program));
                        true
                    }
                    Err(e) => {
                        warn!(
                            code_hash = %key.code_hash,
                            error = %e,
                            "failed to load persisted AOT artifact, will recompile",
                        );
                        false
                    }
                }
            }
            Ok(None) => false,
            Err(e) => {
                warn!(
                    code_hash = %key.code_hash,
                    error = %e,
                    "failed to probe artifact store",
                );
                false
            }
        }
    }

    fn handle_clear_resident(&mut self) {
        self.inner.resident.clear();
        self.resident_meta.clear();
        // Notify any pending sync callers before clearing entries.
        for (_, entry) in self.entries.drain() {
            for n in entry.pending_notifiers {
                n.notify();
            }
        }
        // Discard pending lookup events: they were observed before the clear
        // and would otherwise get processed against the new generation.
        while self.inner.events.pop().is_some() {}
        // Bump generation so in-flight worker results from before the clear are discarded.
        self.generation += 1;
        debug!(generation = self.generation, "resident map cleared");
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

    fn insert_resident(&mut self, key: RuntimeCacheKey, program: Arc<CompiledProgram>) {
        self.inner.resident.insert(key, program);
        self.resident_meta.insert(key, ResidentMeta { last_hit_at: Instant::now() });
    }

    fn remove_resident(&mut self, key: &RuntimeCacheKey) {
        self.inner.resident.remove(key);
        self.resident_meta.remove(key);
    }

    fn handle_worker_result(&mut self, result: WorkerResult) {
        self.pending_jobs = self.pending_jobs.saturating_sub(1);

        // Drain pending notifiers from the entry before processing.
        let pending_notifiers = self
            .entries
            .get_mut(&result.key)
            .map(|e| std::mem::take(&mut e.pending_notifiers))
            .unwrap_or_default();

        let notify = || {
            result.sync_notifier.notify();
            for n in pending_notifiers {
                n.notify();
            }
        };

        // Discard stale results from a previous generation (e.g. after clear).
        if result.generation != self.generation {
            debug!(
                code_hash = %result.key.code_hash,
                result_gen = result.generation,
                current_gen = self.generation,
                "discarding stale worker result",
            );
            self.entries.remove(&result.key);
            notify();
            return;
        }

        let kind = result.kind;
        let success = result.outcome.is_ok();

        if let Some(cb) = &self.on_compilation {
            cb(CompilationEvent {
                code_hash: result.key.code_hash,
                spec_id: result.key.spec_id,
                duration: result.compile_duration,
                kind,
                success,
                timings: result.timings,
            });
        }

        match result.outcome {
            Ok(WorkerSuccess::Jit(success)) => {
                let program =
                    Arc::new(CompiledProgram::new_jit(result.key, success.func, success.backing));
                self.insert_resident(result.key, program);
                self.entries.remove(&result.key);
                self.inner.stats.compilations_succeeded.fetch_add(1, Ordering::Relaxed);

                debug!(
                    code_hash = %result.key.code_hash,
                    spec_id = ?result.key.spec_id,
                    compile_time = ?result.compile_duration,
                    "JIT program published to resident map",
                );
            }
            Ok(WorkerSuccess::Aot(success)) => {
                self.handle_aot_success(result.key, success);
            }
            Err(err) => {
                self.entries.remove(&result.key);
                self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);

                warn!(
                    code_hash = %result.key.code_hash,
                    error = %err,
                    compile_time = ?result.compile_duration,
                    "compilation failed",
                );
            }
        }

        notify();
    }

    fn handle_aot_success(
        &mut self,
        key: RuntimeCacheKey,
        success: crate::runtime::worker::AotSuccess,
    ) {
        let artifact_key = ArtifactKey {
            runtime: key,
            backend: BackendSelection::Llvm,
            opt_level: self.tuning.aot_opt_level,
        };

        let content_hash = keccak256(&success.dylib_bytes).0;

        let manifest = ArtifactManifest {
            artifact_key: artifact_key.clone(),
            symbol_name: success.symbol_name.clone(),
            bytecode_len: success.bytecode_len,
            artifact_len: success.dylib_bytes.len(),
            created_at_unix_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            content_hash,
        };

        // Persist to store if available.
        if let Some(store) = &self.store {
            if let Err(e) = store.store(&artifact_key, &manifest, &success.dylib_bytes) {
                warn!(
                    code_hash = %key.code_hash,
                    error = %e,
                    "failed to persist AOT artifact",
                );
                self.entries.remove(&key);
                self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);
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
                        Ok(CompiledProgram::new_aot(key, func, library))
                    })() {
                        Ok(program) => {
                            self.insert_resident(key, Arc::new(program));
                            self.entries.remove(&key);
                            self.inner.stats.compilations_succeeded.fetch_add(1, Ordering::Relaxed);

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
                            // Persisted successfully but couldn't load — remove so JIT can retry.
                            self.entries.remove(&key);
                            self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                Ok(None) => {
                    warn!(
                        code_hash = %key.code_hash,
                        "stored AOT artifact not found on reload",
                    );
                    self.entries.remove(&key);
                    self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    warn!(
                        code_hash = %key.code_hash,
                        error = %e,
                        "failed to reload persisted AOT artifact",
                    );
                    self.entries.remove(&key);
                    self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else {
            // No store configured — can't persist, remove so JIT can retry.
            warn!(
                code_hash = %key.code_hash,
                "AOT compilation completed but no artifact store configured",
            );
            self.entries.remove(&key);
            self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Runs an eviction sweep: removes idle entries and enforces the memory budget.
    fn run_eviction_sweep(&mut self) {
        if !self.should_sweep() {
            return;
        }

        let now = Instant::now();
        self.last_sweep = now;

        let idle_duration = self.tuning.idle_evict_duration;
        let budget = self.tuning.resident_code_cache_bytes;

        // Phase 1: evict idle entries.
        if let Some(idle) = idle_duration {
            let idle_keys: Vec<RuntimeCacheKey> = self
                .resident_meta
                .iter()
                .filter(|(_, meta)| now.duration_since(meta.last_hit_at) > idle)
                .map(|(key, _)| *key)
                .collect();

            for key in &idle_keys {
                debug!(
                    code_hash = %key.code_hash,
                    spec_id = ?key.spec_id,
                    "evicting idle entry",
                );
                self.remove_resident(key);
                self.entries.remove(key);
                self.inner.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Phase 2: enforce memory budget by evicting LRU JIT entries.
        if budget > 0 && jit_total_bytes() > budget {
            // Collect JIT entries sorted by last_hit_at ascending (oldest first).
            // AOT entries are excluded because they don't contribute to `jit_total_bytes()`.
            let mut entries: Vec<(RuntimeCacheKey, Instant)> = self
                .resident_meta
                .iter()
                .filter(|(key, _)| {
                    self.inner.resident.get(key).is_some_and(|p| matches!(p.kind, ProgramKind::Jit))
                })
                .map(|(key, meta)| (*key, meta.last_hit_at))
                .collect();
            entries.sort_by_key(|(_, t)| *t);

            for (key, _) in entries {
                if jit_total_bytes() <= budget {
                    break;
                }
                debug!(
                    code_hash = %key.code_hash,
                    spec_id = ?key.spec_id,
                    "evicting entry to stay within memory budget",
                );
                self.remove_resident(&key);
                self.entries.remove(&key);
                self.inner.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Returns whether eviction is configured and a sweep is due.
    fn should_sweep(&self) -> bool {
        // Over budget — sweep immediately regardless of interval.
        let maxrss = self.tuning.resident_code_cache_bytes;
        if maxrss > 0 && jit_total_bytes() > maxrss {
            return true;
        }
        self.tuning.idle_evict_duration.is_some()
            && self.last_sweep.elapsed() >= self.tuning.eviction_sweep_interval
    }
}

/// Runs the backend event loop. Called on the backend thread.
pub(crate) fn run(
    inner: Arc<super::BackendShared>,
    cmd_rx: chan::Receiver<Command>,
    config: RuntimeConfig,
) {
    debug!("backend thread started");

    let (result_tx, result_rx) = chan::unbounded::<WorkerResult>();

    let workers = WorkerPool::new(result_tx, config.clone());

    let sweep_interval = config.tuning.eviction_sweep_interval;
    let event_drain_interval = config.tuning.event_drain_interval;

    // Seed resident metadata from startup-preloaded AOT entries.
    let now = Instant::now();
    let mut preload_meta = HashMap::default();
    for entry in inner.resident.iter() {
        preload_meta.insert(*entry.key(), ResidentMeta { last_hit_at: now });
    }

    let mut state = BackendState {
        inner,
        resident_meta: preload_meta,
        entries: HashMap::default(),
        workers,
        result_rx,
        store: config.store,
        tuning: config.tuning,
        pending_jobs: 0,
        generation: 0,
        last_sweep: now,
        on_compilation: config.on_compilation,
    };

    // Tick interval is min(event_drain, sweep) so we never sleep longer than
    // either. Events are drained on every wakeup regardless of cause.
    let tick = event_drain_interval.min(sweep_interval);
    let shutdown_reason;

    loop {
        chan::select! {
            recv(cmd_rx) -> msg => {
                let Ok(cmd) = msg else {
                    shutdown_reason = "channel closed";
                    break;
                };
                if state.handle(cmd).is_break() {
                    shutdown_reason = "shutdown command";
                    break;
                }
            }
            recv(state.result_rx) -> msg => {
                match msg {
                    Ok(result) => state.handle_worker_result(result),
                    Err(_) => warn!("worker unexpectedly closed"),
                }
            }
            default(tick) => {}
        }
        state.tick();
    }

    debug!(?shutdown_reason, stats = ?state.inner.stats(), "backend task shutting down");

    state.workers.shutdown();
}

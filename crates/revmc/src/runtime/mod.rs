//! Runtime JIT backend: O(1) compiled-function lookup with background compilation.
//!
//! - Startup AOT preload from [`ArtifactStore::load_all`] into an immutable in-memory map.
//! - O(1) [`JitBackend::lookup`] that only reads the resident map.
//! - Fire-and-forget lookup-observed events to the backend thread.
//! - Background JIT compilation for hot keys (threshold-based promotion).

use crate::{
    EvmCompilerFn,
    eyre::{self, WrapErr},
};
use api::LoadedLibrary;
use backend::{
    Command, CompileJitRequest, EventQueue, LookupObservedEvent, PrepareAotRequest, ResidentMap,
};
use crossbeam_channel as chan;
use crossbeam_queue::ArrayQueue;
use revm_primitives::{B256, hardfork::SpecId};
use stats::RuntimeStats;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use worker::SyncNotifier;

mod api;
pub use api::{
    AotRequest, CompiledProgram, InterpretReason, LookupDecision, LookupRequest, ProgramKind,
};

mod config;
pub use config::{CompilationEvent, CompilationKind, RuntimeConfig, RuntimeTuning};

mod backend;

mod stats;
pub use stats::RuntimeStatsSnapshot;

mod storage;
pub use storage::{
    ArtifactKey, ArtifactManifest, ArtifactStore, BackendSelection, RuntimeCacheKey, StoredArtifact,
};

mod worker;

#[cfg(test)]
mod tests;

/// Shared inner state for [`JitBackend`].
#[derive(derive_more::Debug)]
pub(crate) struct BackendInner {
    /// Shared resident compiled map.
    #[debug(skip)]
    resident: Arc<ResidentMap>,
    /// Global enable flag.
    enabled: AtomicBool,
    /// Blocking mode: every lookup synchronously compiles and never falls back.
    blocking: bool,
    /// Bounded channel for control commands (compile_jit, prepare_aot, clears,
    /// shutdown). The lookup hot path does NOT use this — see [`events`].
    #[debug(skip)]
    tx: chan::Sender<Command>,
    /// Lock-free unbounded queue of [`LookupObservedEvent`]s.
    ///
    /// The lookup hot path pushes here without signaling waiters; the backend
    /// drains on its loop iterations. Decouples lookup throughput from backend
    /// wakeup cost.
    #[debug(skip)]
    events: Arc<EventQueue>,
    /// Shared stats counters.
    #[debug(skip)]
    stats: Arc<RuntimeStats>,
    /// Backend thread + done signal. `None` after shutdown.
    #[debug(skip)]
    thread: std::sync::Mutex<Option<BackendThread>>,
    /// Shutdown timeout.
    shutdown_timeout: Duration,
    /// Tuning knobs (Copy). Cached for hot-path eligibility checks.
    tuning: crate::runtime::config::RuntimeTuning,
    /// State for lazily spawning the backend thread.
    #[debug(skip)]
    lazy_spawn: std::sync::Mutex<Option<LazySpawnState>>,
}

/// State kept around for lazily spawning the backend thread.
struct LazySpawnState {
    rx: chan::Receiver<Command>,
    config: RuntimeConfig,
}

/// Backend thread handle and its completion signal.
struct BackendThread {
    handle: std::thread::JoinHandle<()>,
    done_rx: chan::Receiver<()>,
}

/// JIT compilation backend with O(1) compiled-function lookup.
///
/// Created via [`JitBackend::new`] or [`JitBackend::disabled`].
/// This type is cheaply clonable (backed by `Arc`).
/// All clones share the same backend thread, resident map, and statistics.
#[derive(Clone, Debug)]
pub struct JitBackend {
    inner: Arc<BackendInner>,
}

impl JitBackend {
    /// Creates a disabled backend that performs no compilation and spawns no threads.
    ///
    /// All [`lookup`](Self::lookup) calls return `LookupDecision::Interpret(Disabled)`.
    /// Call [`set_enabled`](Self::set_enabled) to lazily spawn the backend thread with
    /// a default [`RuntimeConfig`].
    pub fn disabled() -> Self {
        Self::new(RuntimeConfig::default()).expect("default config cannot fail")
    }

    /// Creates a backend from the given config.
    ///
    /// If [`enabled`](RuntimeConfig::enabled) is `true`, the backend thread is spawned
    /// immediately and AOT artifacts are preloaded. Otherwise, both are deferred until the
    /// first [`set_enabled(true)`](Self::set_enabled) call.
    pub fn new(mut config: RuntimeConfig) -> eyre::Result<Self> {
        if config.blocking {
            config.enabled = true;
            config.tuning.jit_hot_threshold = 0;
        }

        let enabled = config.enabled;
        let (tx, rx) = chan::bounded::<Command>(config.tuning.channel_capacity);
        let events = Arc::new(ArrayQueue::new(config.tuning.channel_capacity));
        let tuning = config.tuning;
        let this = Self {
            inner: Arc::new(BackendInner {
                resident: Arc::new(ResidentMap::default()),
                enabled: AtomicBool::new(false),
                blocking: config.blocking,
                tx,
                events,
                stats: Arc::new(RuntimeStats::default()),
                thread: std::sync::Mutex::new(None),
                shutdown_timeout: config.tuning.shutdown_timeout,
                tuning,
                lazy_spawn: std::sync::Mutex::new(Some(LazySpawnState { rx, config })),
            }),
        };
        this.set_enabled(enabled)?;
        Ok(this)
    }

    /// Looks up a compiled function for the given request.
    ///
    /// In normal mode this never blocks. In [`blocking`](RuntimeConfig::blocking) mode,
    /// a miss triggers synchronous JIT compilation and the call blocks until it completes.
    pub fn lookup(&self, req: LookupRequest) -> LookupDecision {
        if !self.inner.enabled.load(Ordering::Relaxed) {
            return LookupDecision::Interpret(InterpretReason::Disabled);
        }

        // Blocking mode: synchronously compile on miss, never fall back.
        if self.inner.blocking {
            return match self.lookup_blocking(req) {
                Some(program) => LookupDecision::Compiled(program),
                None => LookupDecision::Interpret(InterpretReason::JitFailed),
            };
        }

        let key = RuntimeCacheKey { code_hash: req.code_hash, spec_id: req.spec_id };

        let r = self.inner.resident.get(&key);
        let was_hit = r.is_some();
        let decision = if let Some(program) = { r } {
            self.inner.stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
            LookupDecision::Compiled(Arc::clone(&program))
        } else {
            self.inner.stats.lookup_misses.fetch_add(1, Ordering::Relaxed);
            LookupDecision::Interpret(InterpretReason::NotReady)
        };

        if self
            .inner
            .events
            .push(LookupObservedEvent {
                key,
                bytecode: if was_hit { None } else { Some(req.code) },
            })
            .is_err()
        {
            self.inner.stats.events_dropped.fetch_add(1, Ordering::Relaxed);
        }

        decision
    }

    /// Checks the resident map for a compiled program without enqueuing an event.
    pub fn get_compiled(&self, code_hash: B256, spec_id: SpecId) -> Option<Arc<CompiledProgram>> {
        let key = RuntimeCacheKey { code_hash, spec_id };
        self.inner.resident.get(&key).map(|entry| Arc::clone(&entry))
    }

    /// Like [`get_compiled`](Self::get_compiled), but also records hit/miss stats.
    pub fn get_compiled_tracked(
        &self,
        code_hash: B256,
        spec_id: SpecId,
    ) -> Option<Arc<CompiledProgram>> {
        let result = self.get_compiled(code_hash, spec_id);
        if result.is_some() {
            self.inner.stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.inner.stats.lookup_misses.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Looks up a compiled function, blocking until compilation completes if not yet ready.
    ///
    /// If the function is already compiled, returns it immediately. Otherwise, enqueues
    /// a synchronous JIT compilation request and blocks until the result is available.
    /// Returns `None` if the bytecode is ineligible for compilation (see
    /// [`RuntimeTuning::should_compile`]) or compilation fails.
    pub fn lookup_blocking(&self, req: LookupRequest) -> Option<Arc<CompiledProgram>> {
        if !self.inner.tuning.should_compile(&req.code) {
            return None;
        }
        let code_hash = req.code_hash;
        let spec_id = req.spec_id;
        if let Some(program) = self.get_compiled_tracked(code_hash, spec_id) {
            return Some(program);
        }
        let _ = self.compile_jit_sync(req);
        self.get_compiled_tracked(code_hash, spec_id)
    }

    /// Enqueues an explicit JIT compilation request for the given bytecode.
    ///
    /// Blocks if the command channel is full to guarantee delivery.
    pub fn compile_jit(&self, req: LookupRequest) {
        let _ = self.ensure_started();
        let cmd = Command::CompileJit(CompileJitRequest {
            key: RuntimeCacheKey { code_hash: req.code_hash, spec_id: req.spec_id },
            bytecode: req.code,
            sync_notifier: SyncNotifier::none(),
        });
        let _ = self.inner.tx.send(cmd);
    }

    /// Enqueues a JIT compilation request and blocks until the compilation completes.
    ///
    /// Returns `Ok(())` when the compiled function is available in the resident map,
    /// or when the compilation fails. Use [`get_compiled`](Self::get_compiled) to
    /// retrieve the result after this returns.
    pub fn compile_jit_sync(&self, req: LookupRequest) -> eyre::Result<()> {
        self.ensure_started()?;
        let (tx, rx) = chan::bounded(1);
        let cmd = Command::CompileJit(CompileJitRequest {
            key: RuntimeCacheKey { code_hash: req.code_hash, spec_id: req.spec_id },
            bytecode: req.code,
            sync_notifier: SyncNotifier::new(tx),
        });
        self.inner.tx.send(cmd).map_err(|_| eyre::eyre!("backend channel closed"))?;
        rx.recv().map_err(|_| eyre::eyre!("backend shut down before compilation completed"))
    }

    /// Enqueues a single AOT preparation request.
    ///
    /// This is enqueue-only and returns immediately. The compilation happens
    /// asynchronously on the worker pool. The resulting artifact is persisted
    /// via [`ArtifactStore::store`] and loaded into the resident map.
    pub fn prepare_aot(&self, req: AotRequest) {
        self.prepare_aot_batch(vec![req]);
    }

    /// Enqueues a batch of AOT preparation requests.
    ///
    /// Blocks if the command channel is full to guarantee delivery.
    pub fn prepare_aot_batch(&self, reqs: Vec<AotRequest>) {
        let _ = self.ensure_started();
        let owned: Vec<PrepareAotRequest> = reqs
            .into_iter()
            .map(|r| PrepareAotRequest {
                key: RuntimeCacheKey { code_hash: r.code_hash, spec_id: r.spec_id },
                bytecode: r.code,
            })
            .collect();
        let cmd = Command::PrepareAot(owned);
        let _ = self.inner.tx.send(cmd);
    }

    /// Clears the in-memory resident compiled map.
    ///
    /// All compiled programs are removed from the map. Active references
    /// held by callers remain valid until dropped.
    pub fn clear_resident(&self) {
        let _ = self.inner.tx.send(Command::ClearResident);
    }

    /// Clears persisted artifacts from the artifact store.
    pub fn clear_persisted(&self) {
        let _ = self.inner.tx.send(Command::ClearPersisted);
    }

    /// Clears both the resident map and persisted artifacts.
    pub fn clear_all(&self) {
        let _ = self.inner.tx.send(Command::ClearAll);
    }

    /// Returns whether the runtime is enabled.
    pub fn enabled(&self) -> bool {
        self.inner.enabled.load(Ordering::Relaxed)
    }

    /// Sets whether the runtime is enabled, spawning the backend thread on first enable.
    ///
    /// When `enabled` is `true` and the backend thread has not been spawned yet, this
    /// lazily starts it using the config provided at construction time.
    pub fn set_enabled(&self, enabled: bool) -> eyre::Result<()> {
        debug!(enabled, "set_enabled");
        self.inner.enabled.store(enabled, Ordering::Relaxed);
        if enabled {
            self.ensure_started()?;
        }
        Ok(())
    }

    /// Returns a point-in-time snapshot of runtime statistics.
    pub fn stats(&self) -> RuntimeStatsSnapshot {
        self.inner.stats.snapshot(stats::RuntimeStatsGauges {
            resident_entries: self.inner.resident.len() as u64,
            events_queued: self.inner.events.len() as u64,
            command_queue_len: self.inner.tx.len() as u64,
        })
    }

    /// Spawns the backend thread if it hasn't been started yet.
    fn ensure_started(&self) -> eyre::Result<()> {
        let mut guard = self.inner.lazy_spawn.lock().unwrap();
        let Some(lazy) = guard.take() else {
            return Ok(());
        };

        let LazySpawnState { rx, config } = lazy;

        debug!(
            blocking = self.inner.blocking,
            workers = config.tuning.jit_worker_count,
            hot_threshold = config.tuning.jit_hot_threshold,
            channel_capacity = config.tuning.channel_capacity,
            "spawning backend thread",
        );

        // Preload AOT artifacts into the already-allocated resident map.
        match Self::preload_aot(config.store.as_deref()) {
            Ok(entries) => {
                for (key, prog) in entries {
                    self.inner.resident.insert(key, prog);
                }
            }
            Err(e) => {
                // Restore lazy_spawn so a subsequent attempt can retry.
                *guard = Some(LazySpawnState { rx, config });
                return Err(e);
            }
        }

        drop(guard);

        let (done_tx, done_rx) = chan::bounded::<()>(1);
        let inner = Arc::clone(&self.inner);

        let thread = std::thread::Builder::new()
            .name(config.thread_name.clone())
            .spawn(move || {
                backend::run(inner, rx, config);
                let _ = done_tx.send(());
            })
            .wrap_err("failed to spawn backend thread")?;

        *self.inner.thread.lock().unwrap() = Some(BackendThread { handle: thread, done_rx });
        Ok(())
    }

    /// Preloads AOT artifacts from the store as `Arc<CompiledProgram>`s ready to insert.
    fn preload_aot(
        store: Option<&dyn ArtifactStore>,
    ) -> eyre::Result<Vec<(RuntimeCacheKey, Arc<CompiledProgram>)>> {
        let Some(store) = store else {
            debug!("no artifact store configured, skipping AOT preload");
            return Ok(Vec::new());
        };

        let span = info_span!("aot_preload");
        let _enter = span.enter();

        let artifacts = store.load_all()?;
        info!(count = artifacts.len(), "loading AOT artifacts");

        let mut out: Vec<(RuntimeCacheKey, Arc<CompiledProgram>)> =
            Vec::with_capacity(artifacts.len());
        let mut seen = alloy_primitives::map::HashSet::<RuntimeCacheKey>::default();
        let mut loaded = 0u64;
        let mut failed = 0u64;

        for (artifact_key, stored) in artifacts {
            match Self::load_artifact(&artifact_key, &stored) {
                Ok(program) => {
                    let key = artifact_key.runtime.clone();
                    if !seen.insert(key.clone()) {
                        warn!(
                            code_hash = %key.code_hash,
                            spec_id = ?key.spec_id,
                            "duplicate artifact key, keeping first",
                        );
                        continue;
                    }
                    out.push((key, Arc::new(program)));
                    loaded += 1;
                }
                Err(e) => {
                    warn!(
                        code_hash = %artifact_key.runtime.code_hash,
                        error = %e,
                        "failed to load artifact, skipping",
                    );
                    failed += 1;
                }
            }
        }

        info!(loaded, failed, "AOT preload complete");
        Ok(out)
    }

    /// Loads a single artifact: `dlopen`s the dylib path and resolves the symbol.
    fn load_artifact(key: &ArtifactKey, stored: &StoredArtifact) -> eyre::Result<CompiledProgram> {
        let library = unsafe { libloading::Library::new(&stored.dylib_path) }
            .wrap_err_with(|| format!("dlopen {:?}", stored.dylib_path))?;

        let func: EvmCompilerFn = unsafe {
            let sym: libloading::Symbol<'_, EvmCompilerFn> = library
                .get(stored.manifest.symbol_name.as_bytes())
                .wrap_err_with(|| format!("symbol '{}'", stored.manifest.symbol_name))?;
            *sym
        };

        let library = Arc::new(LoadedLibrary::new(library));
        Ok(CompiledProgram::new_aot(key.runtime.clone(), func, library))
    }
}

impl BackendInner {
    fn shutdown(&self) -> eyre::Result<()> {
        debug!("shutting down JIT backend");
        if let Some(ct) = self.thread.lock().unwrap().take() {
            // Ignoring send error — backend may already be gone.
            let _ = self.tx.send(Command::Shutdown);

            // Wait for the thread to signal completion, with a timeout.
            match ct.done_rx.recv_timeout(self.shutdown_timeout) {
                Ok(()) | Err(chan::RecvTimeoutError::Disconnected) => {}
                Err(chan::RecvTimeoutError::Timeout) => {
                    eyre::bail!(
                        "backend thread did not exit within timeout ({:?})",
                        self.shutdown_timeout
                    );
                }
            }

            // Thread signaled done, join should return immediately.
            ct.handle.join().map_err(|_| eyre::eyre!("backend thread panicked"))?;
        }
        Ok(())
    }
}

impl Drop for BackendInner {
    fn drop(&mut self) {
        if let Err(err) = self.shutdown() {
            warn!(%err, "failed to shutdown JIT backend");
        }
    }
}

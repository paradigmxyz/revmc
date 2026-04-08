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
use backend::{Command, CompileJitRequest, LookupObservedEvent, PrepareAotRequest, ResidentMap};
use crossbeam_channel as chan;
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
struct BackendInner {
    /// Shared resident compiled map.
    #[debug(skip)]
    resident: Arc<ResidentMap>,
    /// Global enable flag.
    enabled: AtomicBool,
    /// Blocking mode: every lookup synchronously compiles and never falls back.
    blocking: bool,
    /// Channel for sending commands to the backend thread.
    #[debug(skip)]
    tx: chan::Sender<Command>,
    /// Shared stats counters.
    #[debug(skip)]
    stats: Arc<RuntimeStats>,
    /// Backend thread + done signal. `None` after shutdown.
    #[debug(skip)]
    thread: std::sync::Mutex<Option<BackendThread>>,
    /// Shutdown timeout.
    shutdown_timeout: Duration,
}

/// Backend thread handle and its completion signal.
struct BackendThread {
    handle: std::thread::JoinHandle<()>,
    done_rx: chan::Receiver<()>,
}

/// JIT compilation backend with O(1) compiled-function lookup.
///
/// Created via [`JitBackend::start`]. This type is cheaply clonable (backed by `Arc`).
/// All clones share the same backend thread, resident map, and statistics.
#[derive(Clone, Debug)]
pub struct JitBackend {
    inner: Arc<BackendInner>,
}

impl JitBackend {
    /// Starts the backend: loads AOT artifacts, builds the resident map, and spawns the
    /// backend thread.
    pub fn start(mut config: RuntimeConfig) -> eyre::Result<Self> {
        // Blocking mode forces enabled and zero threshold.
        if config.blocking {
            config.enabled = true;
            config.tuning.jit_hot_threshold = 0;
        }

        debug!(
            enabled = config.enabled,
            blocking = config.blocking,
            workers = config.tuning.jit_worker_count,
            hot_threshold = config.tuning.jit_hot_threshold,
            channel_capacity = config.tuning.lookup_event_channel_capacity,
            "starting JIT backend",
        );
        let resident = Self::preload_aot(config.store.as_deref())?;
        let resident = Arc::new(resident);

        let (tx, rx) = chan::bounded::<Command>(config.tuning.lookup_event_channel_capacity);
        let (done_tx, done_rx) = chan::bounded::<()>(1);

        let tuning = config.tuning;
        let store = config.store.clone();
        let dump_dir = config.dump_dir;
        let debug_assertions = config.debug_assertions;
        let on_compilation = config.on_compilation;
        let resident_for_thread = Arc::clone(&resident);
        let stats = Arc::new(RuntimeStats::default());
        let stats_for_thread = Arc::clone(&stats);

        let thread = std::thread::Builder::new()
            .name(config.thread_name)
            .spawn(move || {
                backend::run(
                    rx,
                    resident_for_thread,
                    store,
                    tuning,
                    dump_dir,
                    debug_assertions,
                    stats_for_thread,
                    on_compilation,
                );
                let _ = done_tx.send(());
            })
            .wrap_err("failed to spawn backend thread")?;

        let inner = BackendInner {
            resident,
            enabled: AtomicBool::new(config.enabled),
            blocking: config.blocking,
            tx,
            stats,
            thread: std::sync::Mutex::new(Some(BackendThread { handle: thread, done_rx })),
            shutdown_timeout: config.tuning.shutdown_timeout,
        };

        Ok(Self { inner: Arc::new(inner) })
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

        let decision = if let Some(program) = self.inner.resident.get(&key) {
            self.inner.stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
            LookupDecision::Compiled(Arc::clone(&program))
        } else {
            self.inner.stats.lookup_misses.fetch_add(1, Ordering::Relaxed);
            LookupDecision::Interpret(InterpretReason::NotReady)
        };

        // Fire-and-forget: silently drop if channel is full.
        let was_hit = matches!(decision, LookupDecision::Compiled(_));
        let event = Command::LookupObserved(LookupObservedEvent {
            key,
            was_hit,
            bytecode: if was_hit { None } else { Some(req.code) },
        });
        match self.inner.tx.try_send(event) {
            Ok(()) => {
                self.inner.stats.events_sent.fetch_add(1, Ordering::Relaxed);
            }
            Err(chan::TrySendError::Full(_) | chan::TrySendError::Disconnected(_)) => {
                self.inner.stats.events_dropped.fetch_add(1, Ordering::Relaxed);
            }
        }

        decision
    }

    /// Checks the resident map for a compiled program without sending any events.
    pub fn get_compiled(&self, code_hash: B256, spec_id: SpecId) -> Option<Arc<CompiledProgram>> {
        let key = RuntimeCacheKey { code_hash, spec_id };
        self.inner.resident.get(&key).map(|entry| Arc::clone(&entry))
    }

    /// Looks up a compiled function, blocking until compilation completes if not yet ready.
    ///
    /// If the function is already compiled, returns it immediately. Otherwise, enqueues
    /// a synchronous JIT compilation request and blocks until the result is available.
    /// Returns `None` only if the bytecode is empty or compilation fails.
    pub fn lookup_blocking(&self, req: LookupRequest) -> Option<Arc<CompiledProgram>> {
        if req.code.is_empty() {
            return None;
        }
        let code_hash = req.code_hash;
        let spec_id = req.spec_id;
        if let Some(program) = self.get_compiled(code_hash, spec_id) {
            return Some(program);
        }
        let _ = self.compile_jit_sync(req);
        self.get_compiled(code_hash, spec_id)
    }

    /// Enqueues an explicit JIT compilation request for the given bytecode.
    ///
    /// This is fire-and-forget: returns immediately and silently drops the request
    /// if the backend channel is full.
    pub fn compile_jit(&self, req: LookupRequest) {
        let cmd = Command::CompileJit(CompileJitRequest {
            key: RuntimeCacheKey { code_hash: req.code_hash, spec_id: req.spec_id },
            bytecode: req.code,
            sync_notifier: SyncNotifier::none(),
        });
        let _ = self.inner.tx.try_send(cmd);
    }

    /// Enqueues a JIT compilation request and blocks until the compilation completes.
    ///
    /// Returns `Ok(())` when the compiled function is available in the resident map,
    /// or when the compilation fails. Use [`get_compiled`](Self::get_compiled) to
    /// retrieve the result after this returns.
    pub fn compile_jit_sync(&self, req: LookupRequest) -> eyre::Result<()> {
        let (tx, rx) = chan::bounded(1);
        let cmd = Command::CompileJit(CompileJitRequest {
            key: RuntimeCacheKey { code_hash: req.code_hash, spec_id: req.spec_id },
            bytecode: req.code,
            sync_notifier: SyncNotifier::new(tx),
        });
        self.inner.tx.try_send(cmd).map_err(|_| eyre::eyre!("backend channel full or closed"))?;
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
    /// This is enqueue-only and returns immediately.
    pub fn prepare_aot_batch(&self, reqs: Vec<AotRequest>) {
        let owned: Vec<PrepareAotRequest> = reqs
            .into_iter()
            .map(|r| PrepareAotRequest {
                key: RuntimeCacheKey { code_hash: r.code_hash, spec_id: r.spec_id },
                bytecode: r.code,
            })
            .collect();
        let cmd = Command::PrepareAot(owned);
        let _ = self.inner.tx.try_send(cmd);
    }

    /// Clears the in-memory resident compiled map.
    ///
    /// All compiled programs are removed from the map. Active references
    /// held by callers remain valid until dropped.
    pub fn clear_resident(&self) {
        let _ = self.inner.tx.try_send(Command::ClearResident);
    }

    /// Clears persisted artifacts from the artifact store.
    pub fn clear_persisted(&self) {
        let _ = self.inner.tx.try_send(Command::ClearPersisted);
    }

    /// Clears both the resident map and persisted artifacts.
    pub fn clear_all(&self) {
        let _ = self.inner.tx.try_send(Command::ClearAll);
    }

    /// Sets whether the runtime is enabled.
    pub fn set_enabled(&self, enabled: bool) {
        debug!(enabled, "set_enabled");
        self.inner.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Returns a point-in-time snapshot of runtime statistics.
    pub fn stats(&self) -> RuntimeStatsSnapshot {
        self.inner.stats.snapshot(self.inner.resident.len() as u64, self.inner.tx.len() as u64)
    }

    /// Preloads AOT artifacts from the store into the resident map.
    fn preload_aot(store: Option<&dyn ArtifactStore>) -> eyre::Result<ResidentMap> {
        let store = match store {
            Some(s) => s,
            None => {
                debug!("no artifact store configured, skipping AOT preload");
                return Ok(ResidentMap::default());
            }
        };

        let span = info_span!("aot_preload");
        let _enter = span.enter();

        let artifacts = store.load_all()?;
        info!(count = artifacts.len(), "loading AOT artifacts");

        let map = ResidentMap::with_capacity(artifacts.len());
        let mut loaded = 0u64;
        let mut failed = 0u64;

        for (artifact_key, stored) in artifacts {
            match Self::load_artifact(&artifact_key, &stored) {
                Ok(program) => {
                    let key = artifact_key.runtime.clone();
                    if map.contains_key(&key) {
                        warn!(
                            code_hash = %key.code_hash,
                            spec_id = ?key.spec_id,
                            "duplicate artifact key, keeping first",
                        );
                        continue;
                    }
                    map.insert(key, Arc::new(program));
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
        Ok(map)
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

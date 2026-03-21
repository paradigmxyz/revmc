//! Runtime JIT coordinator: O(1) compiled-function lookup with background compilation.
//!
//! - Startup AOT preload from [`ArtifactStore::load_all`] into an immutable in-memory map.
//! - O(1) [`JitCoordinatorHandle::lookup`] that only reads the resident map.
//! - Fire-and-forget lookup-observed events to the coordinator thread.
//! - Background JIT compilation for hot keys (threshold-based promotion).

mod api;
mod config;
mod coordinator;
mod stats;
mod storage;
mod worker;

#[cfg(test)]
mod tests;

pub use api::{
    AotRequest, CompiledProgram, InterpretReason, LookupDecision, LookupRequest, ProgramKind,
};
pub use config::{RuntimeConfig, RuntimeTuning};
pub use stats::RuntimeStatsSnapshot;
pub use storage::{
    ArtifactKey, ArtifactManifest, ArtifactStore, BackendSelection, RuntimeCacheKey, StoredArtifact,
};

use api::LoadedLibrary;
use coordinator::{
    Command, CompileJitRequest, LookupObservedEvent, PrepareAotRequest, ResidentMap,
};
use stats::RuntimeStats;

use crate::{
    EvmCompilerFn,
    eyre::{self, WrapErr},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

/// The JIT coordinator. Owns the coordinator thread and resident map.
///
/// Created via [`JitCoordinator::start`]. Use [`JitCoordinator::handle`] to obtain a
/// clonable handle for lookups.
#[allow(missing_debug_implementations)]
pub struct JitCoordinator {
    handle: JitCoordinatorHandle,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Receives a signal when the coordinator thread exits.
    thread_done_rx: mpsc::Receiver<()>,
    shutdown_timeout: Duration,
}

impl JitCoordinator {
    /// Starts the coordinator: loads AOT artifacts, builds the resident map, and spawns the
    /// coordinator thread.
    pub fn start(config: RuntimeConfig) -> eyre::Result<Self> {
        debug!(
            enabled = config.enabled,
            workers = config.tuning.jit_worker_count,
            hot_threshold = config.tuning.jit_hot_threshold,
            channel_capacity = config.tuning.lookup_event_channel_capacity,
            "starting JIT coordinator",
        );
        let resident = Self::preload_aot(config.store.as_deref())?;
        let resident = Arc::new(resident);
        let stats = Arc::new(RuntimeStats::default());
        let enabled = Arc::new(AtomicBool::new(config.enabled));

        let (tx, rx) = mpsc::sync_channel::<Command>(config.tuning.lookup_event_channel_capacity);
        let (done_tx, done_rx) = mpsc::sync_channel::<()>(1);

        let tuning = config.tuning;
        let store = config.store.clone();
        let resident_for_coord = Arc::clone(&resident);

        let thread = std::thread::Builder::new()
            .name(config.thread_name)
            .spawn(move || {
                coordinator::run(rx, resident_for_coord, store, tuning);
                let _ = done_tx.send(());
            })
            .wrap_err("failed to spawn coordinator")?;

        let handle = JitCoordinatorHandle { resident, enabled, tx, stats };

        Ok(Self {
            handle,
            thread: Some(thread),
            thread_done_rx: done_rx,
            shutdown_timeout: config.tuning.shutdown_timeout,
        })
    }

    /// Returns a clonable handle for performing lookups.
    pub fn handle(&self) -> JitCoordinatorHandle {
        self.handle.clone()
    }

    /// Shuts down the coordinator thread and waits for it to finish.
    pub fn shutdown(mut self) -> eyre::Result<()> {
        self.shutdown_inner()
    }

    fn shutdown_inner(&mut self) -> eyre::Result<()> {
        debug!("shutting down JIT coordinator");
        if let Some(thread) = self.thread.take() {
            // Ignoring send error — coordinator may already be gone.
            let _ = self.handle.tx.send(Command::Shutdown);

            // Wait for the thread to signal completion, with a timeout.
            match self.thread_done_rx.recv_timeout(self.shutdown_timeout) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {}
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    warn!(
                        timeout = ?self.shutdown_timeout,
                        "coordinator thread did not exit within timeout",
                    );
                    eyre::bail!("coordinator thread did not exit within timeout");
                }
            }

            // Thread signaled done, join should return immediately.
            thread.join().map_err(|_| eyre::eyre!("coordinator thread panicked"))?;
        }
        Ok(())
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
        Ok(CompiledProgram::new_aot(
            key.runtime.clone(),
            func,
            stored.manifest.artifact_len,
            library,
        ))
    }
}

impl Drop for JitCoordinator {
    fn drop(&mut self) {
        let _ = self.shutdown_inner();
    }
}

/// Clonable handle for performing lookups against the runtime.
///
/// Obtained via [`JitCoordinator::handle`].
#[derive(Clone)]
#[allow(missing_debug_implementations)]
pub struct JitCoordinatorHandle {
    /// Shared resident compiled map.
    resident: Arc<ResidentMap>,
    /// Global enable flag.
    enabled: Arc<AtomicBool>,
    /// Channel for sending commands to the coordinator thread.
    tx: mpsc::SyncSender<Command>,
    /// Shared stats counters.
    stats: Arc<RuntimeStats>,
}

impl JitCoordinatorHandle {
    /// Looks up a compiled function for the given request.
    ///
    /// This never blocks, never touches storage, and never waits for compilation.
    pub fn lookup(&self, req: LookupRequest) -> LookupDecision {
        if !self.enabled.load(Ordering::Relaxed) {
            return LookupDecision::Interpret(InterpretReason::Disabled);
        }

        let key = RuntimeCacheKey { code_hash: req.code_hash, spec_id: req.spec_id };

        let decision = if let Some(program) = self.resident.get(&key) {
            self.stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
            trace!(code_hash = %req.code_hash, "lookup hit");
            LookupDecision::Compiled(Arc::clone(&program))
        } else {
            self.stats.lookup_misses.fetch_add(1, Ordering::Relaxed);
            trace!(code_hash = %req.code_hash, "lookup miss");
            LookupDecision::Interpret(InterpretReason::NotReady)
        };

        // Fire-and-forget: silently drop if channel is full.
        let was_hit = matches!(decision, LookupDecision::Compiled(_));
        let event = Command::LookupObserved(LookupObservedEvent {
            key,
            was_hit,
            bytecode: if was_hit { None } else { Some(req.code) },
        });
        match self.tx.try_send(event) {
            Ok(()) => {
                self.stats.events_sent.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::TrySendError::Full(_) | mpsc::TrySendError::Disconnected(_)) => {
                self.stats.events_dropped.fetch_add(1, Ordering::Relaxed);
            }
        }

        decision
    }

    /// Checks the resident map for a compiled program without sending any events.
    pub fn get_compiled(
        &self,
        code_hash: alloy_primitives::B256,
        spec_id: revm_primitives::hardfork::SpecId,
    ) -> Option<Arc<CompiledProgram>> {
        let key = RuntimeCacheKey { code_hash, spec_id };
        self.resident.get(&key).map(|entry| Arc::clone(&entry))
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
    /// if the coordinator channel is full.
    pub fn compile_jit(&self, req: LookupRequest) {
        let cmd = Command::CompileJit(CompileJitRequest {
            key: RuntimeCacheKey { code_hash: req.code_hash, spec_id: req.spec_id },
            bytecode: req.code,
        });
        let _ = self.tx.try_send(cmd);
    }

    /// Enqueues a JIT compilation request and blocks until the compilation completes.
    ///
    /// Returns `Ok(())` when the compiled function is available in the resident map,
    /// or when the compilation fails. Use [`get_compiled`](Self::get_compiled) to
    /// retrieve the result after this returns.
    pub fn compile_jit_sync(&self, req: LookupRequest) -> eyre::Result<()> {
        let (tx, rx) = mpsc::sync_channel(1);
        let cmd = Command::CompileJitSync(
            CompileJitRequest {
                key: RuntimeCacheKey { code_hash: req.code_hash, spec_id: req.spec_id },
                bytecode: req.code,
            },
            tx,
        );
        self.tx.try_send(cmd).map_err(|_| eyre::eyre!("coordinator channel full or closed"))?;
        rx.recv().map_err(|_| eyre::eyre!("coordinator shut down before compilation completed"))
    }

    /// Enqueues a single AOT preparation request.
    ///
    /// This is enqueue-only and returns immediately. The compilation happens
    /// asynchronously on the worker pool. The resulting artifact is persisted
    /// via [`ArtifactStore::store`] and loaded into the resident map.
    pub fn prepare_aot(&self, req: AotRequest) -> eyre::Result<()> {
        self.prepare_aot_batch(vec![req])
    }

    /// Enqueues a batch of AOT preparation requests.
    ///
    /// This is enqueue-only and returns immediately.
    pub fn prepare_aot_batch(&self, reqs: Vec<AotRequest>) -> eyre::Result<()> {
        let owned: Vec<PrepareAotRequest> = reqs
            .into_iter()
            .map(|r| PrepareAotRequest {
                key: RuntimeCacheKey { code_hash: r.code_hash, spec_id: r.spec_id },
                bytecode: r.code,
            })
            .collect();
        let cmd = Command::PrepareAot(owned);
        self.tx.try_send(cmd).map_err(|_| eyre::eyre!("coordinator channel full or closed"))
    }

    /// Clears the in-memory resident compiled map.
    ///
    /// All compiled programs are removed from the map. Active references
    /// held by callers remain valid until dropped.
    pub fn clear_resident(&self) -> eyre::Result<()> {
        self.tx
            .try_send(Command::ClearResident)
            .map_err(|_| eyre::eyre!("coordinator channel full or closed"))
    }

    /// Clears persisted artifacts from the artifact store.
    pub fn clear_persisted(&self) -> eyre::Result<()> {
        self.tx
            .try_send(Command::ClearPersisted)
            .map_err(|_| eyre::eyre!("coordinator channel full or closed"))
    }

    /// Clears both the resident map and persisted artifacts.
    pub fn clear_all(&self) -> eyre::Result<()> {
        self.tx
            .try_send(Command::ClearAll)
            .map_err(|_| eyre::eyre!("coordinator channel full or closed"))
    }

    /// Sets whether the runtime is enabled.
    pub fn set_enabled(&self, enabled: bool) {
        debug!(enabled, "set_enabled");
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Returns a point-in-time snapshot of runtime statistics.
    pub fn stats(&self) -> RuntimeStatsSnapshot {
        self.stats.snapshot(self.resident.len() as u64)
    }
}

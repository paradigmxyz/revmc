//! Runtime JIT coordinator: O(1) compiled-function lookup with background compilation.
//!
//! # Phase 0
//!
//! - Startup AOT preload from [`ArtifactStore::load_all`] into an immutable in-memory map.
//! - O(1) [`JitCoordinatorHandle::lookup`] that only reads the resident map.
//! - Fire-and-forget lookup-observed events to the coordinator thread.

mod api;
mod config;
mod coordinator;
mod error;
mod stats;
mod storage;

#[cfg(test)]
mod tests;

pub use api::{CompiledProgram, InterpretReason, LookupDecision, LookupRequest, ProgramKind};
pub use config::{RuntimeConfig, RuntimeTuning};
pub use error::{RuntimeError, StorageError};
pub use stats::RuntimeStatsSnapshot;
pub use storage::{
    ArtifactKey, ArtifactManifest, ArtifactStore, BackendSelection, RuntimeCacheKey, StoredArtifact,
};

use api::LoadedLibraryOwner;
use coordinator::{Command, LookupObservedEvent};
use stats::RuntimeStats;

use crate::EvmCompilerFn;
use rustc_hash::FxHashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

/// The JIT coordinator. Owns the coordinator thread and resident map.
///
/// Created via [`JitCoordinator::start`]. Use [`JitCoordinator::handle`] to obtain a
/// clonable handle for lookups.
#[allow(missing_debug_implementations)]
pub struct JitCoordinator {
    handle: JitCoordinatorHandle,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl JitCoordinator {
    /// Starts the coordinator: loads AOT artifacts, builds the resident map, and spawns the
    /// coordinator thread.
    pub fn start(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        let resident = Self::preload_aot(config.store.as_deref())?;
        let resident = Arc::new(resident);
        let stats = Arc::new(RuntimeStats::default());
        let enabled = Arc::new(AtomicBool::new(config.enabled));

        let (tx, rx) = mpsc::sync_channel::<Command>(config.tuning.lookup_event_channel_capacity);

        let thread = std::thread::Builder::new()
            .name(config.thread_name)
            .spawn(move || coordinator::run(rx))
            .map_err(|e| RuntimeError::ArtifactLoad(format!("failed to spawn coordinator: {e}")))?;

        let handle = JitCoordinatorHandle { resident, enabled, tx, stats };

        Ok(Self { handle, thread: Some(thread) })
    }

    /// Returns a clonable handle for performing lookups.
    pub fn handle(&self) -> JitCoordinatorHandle {
        self.handle.clone()
    }

    /// Shuts down the coordinator thread and waits for it to finish.
    pub fn shutdown(mut self) -> Result<(), RuntimeError> {
        self.shutdown_inner()
    }

    fn shutdown_inner(&mut self) -> Result<(), RuntimeError> {
        if let Some(thread) = self.thread.take() {
            // Ignoring send error — coordinator may already be gone.
            let _ = self.handle.tx.send(Command::Shutdown);
            thread.join().map_err(|_| RuntimeError::Shutdown)?;
        }
        Ok(())
    }

    /// Preloads AOT artifacts from the store into the resident map.
    fn preload_aot(
        store: Option<&dyn ArtifactStore>,
    ) -> Result<FxHashMap<RuntimeCacheKey, Arc<CompiledProgram>>, RuntimeError> {
        let store = match store {
            Some(s) => s,
            None => {
                debug!("no artifact store configured, skipping AOT preload");
                return Ok(FxHashMap::default());
            }
        };

        let span = tracing::info_span!("aot_preload");
        let _enter = span.enter();

        let artifacts = store.load_all()?;
        info!(count = artifacts.len(), "loading AOT artifacts");

        let mut map = FxHashMap::with_capacity_and_hasher(artifacts.len(), Default::default());
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

    /// Loads a single artifact, resolves the symbol, and returns a `CompiledProgram`.
    ///
    /// On Linux, uses `memfd_create` to load entirely from memory (no filesystem I/O).
    /// On other platforms, writes to a temp directory and loads via `libloading`.
    fn load_artifact(
        key: &ArtifactKey,
        stored: &StoredArtifact,
    ) -> Result<CompiledProgram, RuntimeError> {
        let owner = Self::load_library(&stored.dylib_bytes)?;

        let func: EvmCompilerFn = unsafe {
            let sym: libloading::Symbol<'_, EvmCompilerFn> =
                owner.library().get(stored.manifest.symbol_name.as_bytes()).map_err(|e| {
                    RuntimeError::ArtifactLoad(format!(
                        "symbol '{}': {e}",
                        stored.manifest.symbol_name,
                    ))
                })?;
            *sym
        };

        let owner = Arc::new(owner);
        Ok(CompiledProgram::new_aot(key.runtime.clone(), func, stored.manifest.artifact_len, owner))
    }

    /// Loads a shared library from raw bytes.
    ///
    /// On Linux, uses `memfd_create` + `dlopen("/proc/self/fd/{fd}")` — entirely in memory.
    /// On other platforms, writes to a temp directory.
    fn load_library(dylib_bytes: &[u8]) -> Result<LoadedLibraryOwner, RuntimeError> {
        #[cfg(target_os = "linux")]
        {
            Self::load_library_memfd(dylib_bytes)
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self::load_library_tempfile(dylib_bytes)
        }
    }

    /// Linux: load via `memfd_create` + `dlopen("/proc/self/fd/{fd}")` — no filesystem I/O.
    #[cfg(target_os = "linux")]
    fn load_library_memfd(dylib_bytes: &[u8]) -> Result<LoadedLibraryOwner, RuntimeError> {
        use rustix::fd::IntoRawFd;
        use std::{
            io::Write,
            os::unix::io::{AsRawFd, FromRawFd},
        };

        // Create an anonymous in-memory file.
        let memfd = rustix::fs::memfd_create(c"revmc-artifact", rustix::fs::MemfdFlags::CLOEXEC)
            .map_err(|e| RuntimeError::ArtifactLoad(format!("memfd_create: {e}")))?;

        // Convert rustix OwnedFd → std File for writing.
        let mut file = unsafe { std::fs::File::from_raw_fd(memfd.into_raw_fd()) };
        file.write_all(dylib_bytes)
            .map_err(|e| RuntimeError::ArtifactLoad(format!("write memfd: {e}")))?;

        // dlopen via /proc/self/fd/{fd}.
        let fd: std::os::unix::io::OwnedFd = file.into();
        let path = format!("/proc/self/fd/{}", fd.as_raw_fd());
        let library = unsafe { libloading::Library::new(&path) }
            .map_err(|e| RuntimeError::ArtifactLoad(format!("dlopen memfd: {e}")))?;

        Ok(LoadedLibraryOwner::new_memfd(library, fd))
    }

    /// Non-Linux fallback: write to temp directory and load.
    #[cfg(not(target_os = "linux"))]
    fn load_library_tempfile(dylib_bytes: &[u8]) -> Result<LoadedLibraryOwner, RuntimeError> {
        let tmp_dir =
            tempfile::tempdir().map_err(|e| RuntimeError::ArtifactLoad(format!("tempdir: {e}")))?;

        let lib_ext = if cfg!(target_os = "macos") {
            "dylib"
        } else if cfg!(target_os = "windows") {
            "dll"
        } else {
            "so"
        };
        let lib_path = tmp_dir.path().join(format!("artifact.{lib_ext}"));

        std::fs::write(&lib_path, dylib_bytes)
            .map_err(|e| RuntimeError::ArtifactLoad(format!("write dylib: {e}")))?;

        let library = unsafe { libloading::Library::new(&lib_path) }
            .map_err(|e| RuntimeError::ArtifactLoad(format!("load library: {e}")))?;

        Ok(LoadedLibraryOwner::new_tempdir(library, tmp_dir))
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
    /// Immutable resident compiled map.
    resident: Arc<FxHashMap<RuntimeCacheKey, Arc<CompiledProgram>>>,
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
    pub fn lookup(&self, req: LookupRequest<'_>) -> LookupDecision {
        if !self.enabled.load(Ordering::Relaxed) {
            self.stats.lookup_disabled.fetch_add(1, Ordering::Relaxed);
            return LookupDecision::Interpret(InterpretReason::Disabled);
        }

        let key = RuntimeCacheKey { code_hash: req.code_hash, spec_id: req.spec_id };

        let decision = if let Some(program) = self.resident.get(&key) {
            self.stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
            LookupDecision::Compiled(Arc::clone(program))
        } else {
            self.stats.lookup_misses.fetch_add(1, Ordering::Relaxed);
            LookupDecision::Interpret(InterpretReason::NotReady)
        };

        // Fire-and-forget: silently drop if channel is full.
        let was_hit = matches!(decision, LookupDecision::Compiled(_));
        let event = Command::LookupObserved(LookupObservedEvent { key, was_hit });
        match self.tx.try_send(event) {
            Ok(()) => {
                self.stats.events_sent.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::TrySendError::Full(_)) => {
                self.stats.events_dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.stats.events_dropped.fetch_add(1, Ordering::Relaxed);
            }
        }

        decision
    }

    /// Sets whether the runtime is enabled.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Returns a point-in-time snapshot of runtime statistics.
    pub fn stats(&self) -> RuntimeStatsSnapshot {
        self.stats.snapshot(self.resident.len() as u64)
    }
}

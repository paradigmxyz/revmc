//! Tests for the runtime module.

use super::*;
use alloy_primitives::{B256, Bytes};
use revm_primitives::hardfork::SpecId;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Test bytecodes.
// ---------------------------------------------------------------------------

/// PUSH1 0x42 PUSH0 MSTORE PUSH1 0x20 PUSH0 RETURN — returns 0x42.
const BYTECODE_RET42: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];

/// PUSH1 1 PUSH1 1 ADD PUSH0 MSTORE PUSH1 0x20 PUSH0 RETURN — returns 2.
const BYTECODE_ADD: &[u8] = &[0x60, 0x01, 0x60, 0x01, 0x01, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];

/// Returns a simple bytecode that varies by index (for generating distinct code hashes).
fn indexed_bytecode(i: u8) -> Vec<u8> {
    vec![0x60, i, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3]
}

// ---------------------------------------------------------------------------
// Test harness.
// ---------------------------------------------------------------------------

/// Helper: polls `f` until it returns `Some(T)` or the deadline is reached.
fn poll_until<T>(timeout: std::time::Duration, mut f: impl FnMut() -> Option<T>) -> T {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(std::time::Instant::now() < deadline, "timed out polling");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// Test harness wrapping a [`JitBackend`] with convenience methods.
///
/// Automatically shuts down the backend on drop.
struct TestBackend {
    backend: JitBackend,
}

impl TestBackend {
    /// Creates a backend with the given config.
    fn new(config: RuntimeConfig) -> Self {
        Self { backend: JitBackend::start(config).unwrap() }
    }

    /// Creates a backend with `enabled: true` and the given tuning overrides.
    fn with_tuning(tuning: RuntimeTuning) -> Self {
        Self::new(RuntimeConfig { enabled: true, tuning, ..Default::default() })
    }

    /// Creates a backend with `enabled: true`, 1 worker, and the given tuning overrides.
    fn with_tuning_1w(tuning: RuntimeTuning) -> Self {
        Self::with_tuning(RuntimeTuning { jit_worker_count: 1, ..tuning })
    }

    /// Creates a backend with `enabled: true`, 1 worker, and default tuning.
    fn default_1w() -> Self {
        Self::with_tuning_1w(RuntimeTuning::default())
    }

    /// Builds a [`LookupRequest`] for the given bytecode and spec.
    fn req(bytecode: &[u8], spec_id: SpecId) -> LookupRequest {
        LookupRequest {
            code_hash: alloy_primitives::keccak256(bytecode),
            code: Bytes::copy_from_slice(bytecode),
            spec_id,
        }
    }

    /// Builds a [`LookupRequest`] with [`SpecId::CANCUN`].
    fn req_cancun(bytecode: &[u8]) -> LookupRequest {
        Self::req(bytecode, SpecId::CANCUN)
    }

    /// Sends lookups until JIT promotion triggers and the program appears in the resident map.
    /// Returns the compiled program.
    fn trigger_jit(&self, bytecode: &[u8], spec_id: SpecId) -> Arc<CompiledProgram> {
        for _ in 0..10 {
            let _ = self.backend.lookup(Self::req(bytecode, spec_id));
        }
        self.wait_compiled(bytecode, spec_id)
    }

    /// Like [`trigger_jit`](Self::trigger_jit) but with [`SpecId::CANCUN`].
    fn trigger_jit_cancun(&self, bytecode: &[u8]) -> Arc<CompiledProgram> {
        self.trigger_jit(bytecode, SpecId::CANCUN)
    }

    /// Polls until the given bytecode is compiled and available in the resident map.
    fn wait_compiled(&self, bytecode: &[u8], spec_id: SpecId) -> Arc<CompiledProgram> {
        poll_until(std::time::Duration::from_secs(30), || {
            match self.backend.lookup(Self::req(bytecode, spec_id)) {
                LookupDecision::Compiled(p) => Some(p),
                _ => None,
            }
        })
    }

    /// Polls until the resident map reaches the expected entry count.
    fn wait_resident_count(&self, expected: u64) {
        poll_until(std::time::Duration::from_secs(5), || {
            let n = self.backend.stats().resident_entries;
            if n == expected { Some(()) } else { None }
        });
    }

    fn stats(&self) -> RuntimeStatsSnapshot {
        self.backend.stats()
    }
}

impl std::ops::Deref for TestBackend {
    type Target = JitBackend;
    fn deref(&self) -> &JitBackend {
        &self.backend
    }
}

impl Drop for TestBackend {
    fn drop(&mut self) {
        self.backend.shutdown().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Artifact stores for testing.
// ---------------------------------------------------------------------------

/// A no-op artifact store that returns no artifacts.
struct EmptyStore;

impl ArtifactStore for EmptyStore {
    fn load_all(&self) -> eyre::Result<Vec<(ArtifactKey, StoredArtifact)>> {
        Ok(vec![])
    }

    fn load(&self, _key: &ArtifactKey) -> eyre::Result<Option<StoredArtifact>> {
        Ok(None)
    }

    fn store(
        &self,
        _key: &ArtifactKey,
        _manifest: &ArtifactManifest,
        _dylib_bytes: &[u8],
    ) -> eyre::Result<()> {
        Ok(())
    }

    fn delete(&self, _key: &ArtifactKey) -> eyre::Result<()> {
        Ok(())
    }

    fn clear(&self) -> eyre::Result<()> {
        Ok(())
    }
}

/// A store that fails on load_all.
struct FailingStore;

impl ArtifactStore for FailingStore {
    fn load_all(&self) -> eyre::Result<Vec<(ArtifactKey, StoredArtifact)>> {
        Err(std::io::Error::other("boom").into())
    }

    fn load(&self, _key: &ArtifactKey) -> eyre::Result<Option<StoredArtifact>> {
        Ok(None)
    }

    fn store(
        &self,
        _key: &ArtifactKey,
        _manifest: &ArtifactManifest,
        _dylib_bytes: &[u8],
    ) -> eyre::Result<()> {
        Ok(())
    }

    fn delete(&self, _key: &ArtifactKey) -> eyre::Result<()> {
        Ok(())
    }

    fn clear(&self) -> eyre::Result<()> {
        Ok(())
    }
}

/// An artifact store backed by a temp directory on disk.
///
/// `store()` writes the dylib bytes to a file. `load()` returns the path to it.
struct TempDirStore {
    dir: std::path::PathBuf,
    artifacts: Mutex<std::collections::HashMap<String, (ArtifactManifest, std::path::PathBuf)>>,
}

impl TempDirStore {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("revmc-test-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir, artifacts: Mutex::new(std::collections::HashMap::new()) }
    }

    fn artifact_file_key(key: &ArtifactKey) -> String {
        format!("{:x}_{:?}", key.runtime.code_hash, key.runtime.spec_id)
    }

    fn stored_count(&self) -> usize {
        self.artifacts.lock().unwrap().len()
    }
}

impl Drop for TempDirStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl ArtifactStore for TempDirStore {
    fn load_all(&self) -> eyre::Result<Vec<(ArtifactKey, StoredArtifact)>> {
        let map = self.artifacts.lock().unwrap();
        Ok(map
            .values()
            .map(|(manifest, path)| {
                (
                    manifest.artifact_key.clone(),
                    StoredArtifact { manifest: manifest.clone(), dylib_path: path.clone() },
                )
            })
            .collect())
    }

    fn load(&self, key: &ArtifactKey) -> eyre::Result<Option<StoredArtifact>> {
        let map = self.artifacts.lock().unwrap();
        let file_key = Self::artifact_file_key(key);
        Ok(map.get(&file_key).map(|(manifest, path)| StoredArtifact {
            manifest: manifest.clone(),
            dylib_path: path.clone(),
        }))
    }

    fn store(
        &self,
        key: &ArtifactKey,
        manifest: &ArtifactManifest,
        dylib_bytes: &[u8],
    ) -> eyre::Result<()> {
        let file_key = Self::artifact_file_key(key);
        let path = self.dir.join(format!("{file_key}.so"));
        std::fs::write(&path, dylib_bytes)?;
        self.artifacts.lock().unwrap().insert(file_key, (manifest.clone(), path));
        Ok(())
    }

    fn delete(&self, key: &ArtifactKey) -> eyre::Result<()> {
        let file_key = Self::artifact_file_key(key);
        let mut map = self.artifacts.lock().unwrap();
        if let Some((_, path)) = map.remove(&file_key) {
            let _ = std::fs::remove_file(path);
        }
        Ok(())
    }

    fn clear(&self) -> eyre::Result<()> {
        let mut map = self.artifacts.lock().unwrap();
        for (_, path) in map.values() {
            let _ = std::fs::remove_file(path);
        }
        map.clear();
        Ok(())
    }
}

// ===========================================================================
// Tests: startup / basic.
// ===========================================================================

#[test]
fn start_no_store() {
    let tb = TestBackend::new(RuntimeConfig::default());
    assert_eq!(tb.stats().resident_entries, 0);
}

#[test]
fn start_empty_store() {
    let config =
        RuntimeConfig { enabled: true, store: Some(Arc::new(EmptyStore)), ..Default::default() };
    let tb = TestBackend::new(config);
    assert_eq!(tb.stats().resident_entries, 0);
}

#[test]
fn startup_store_failure() {
    let config =
        RuntimeConfig { enabled: true, store: Some(Arc::new(FailingStore)), ..Default::default() };
    let result = JitBackend::start(config);
    assert!(result.is_err());
}

// ===========================================================================
// Tests: lookup behavior.
// ===========================================================================

#[test]
fn lookup_disabled() {
    let tb = TestBackend::new(RuntimeConfig { enabled: false, ..Default::default() });

    let req = TestBackend::req_cancun(&[0x00]);
    let decision = tb.lookup(req);
    assert!(matches!(decision, LookupDecision::Interpret(InterpretReason::Disabled)));

    let stats = tb.stats();
    assert_eq!(stats.lookup_hits, 0);
    assert_eq!(stats.lookup_misses, 0);
}

#[test]
fn lookup_miss_when_enabled() {
    let tb = TestBackend::new(RuntimeConfig { enabled: true, ..Default::default() });

    let decision = tb.lookup(TestBackend::req_cancun(&[0x00]));
    assert!(matches!(decision, LookupDecision::Interpret(InterpretReason::NotReady)));

    assert_eq!(tb.stats().lookup_misses, 1);
    assert_eq!(tb.stats().lookup_hits, 0);
}

#[test]
fn set_enabled_toggle() {
    let tb = TestBackend::new(RuntimeConfig { enabled: false, ..Default::default() });
    let req = TestBackend::req_cancun(&[0x00]);

    assert!(matches!(tb.lookup(req.clone()), LookupDecision::Interpret(InterpretReason::Disabled)));

    tb.set_enabled(true);
    assert!(matches!(tb.lookup(req.clone()), LookupDecision::Interpret(InterpretReason::NotReady)));

    tb.set_enabled(false);
    assert!(matches!(tb.lookup(req), LookupDecision::Interpret(InterpretReason::Disabled)));
}

#[test]
fn events_sent_on_lookup() {
    let tb = TestBackend::new(RuntimeConfig { enabled: true, ..Default::default() });

    for _ in 0..10 {
        let _ = tb.lookup(TestBackend::req_cancun(&[]));
    }

    assert_eq!(tb.stats().events_sent, 10);
    assert_eq!(tb.stats().events_dropped, 0);
}

#[test]
fn drop_shuts_down_backend() {
    let backend = JitBackend::start(RuntimeConfig::default()).unwrap();
    let backend2 = backend.clone();
    drop(backend);

    // Lookups still work (no panic) — backend is still running because backend2 holds a ref.
    let _ = backend2.lookup(TestBackend::req_cancun(&[]));
}

#[test]
fn backend_clone() {
    let tb = TestBackend::new(RuntimeConfig { enabled: true, ..Default::default() });
    let b2 = tb.backend.clone();

    let _ = tb.lookup(TestBackend::req_cancun(&[]));
    let _ = b2.lookup(TestBackend::req_cancun(&[]));

    assert_eq!(tb.stats().lookup_misses, 2);
}

// ===========================================================================
// Tests: enqueue APIs (fire-and-forget, no LLVM needed).
// ===========================================================================

#[test]
fn compile_jit_enqueue() {
    let tb = TestBackend::new(RuntimeConfig { enabled: true, ..Default::default() });
    tb.compile_jit(TestBackend::req_cancun(BYTECODE_RET42));
}

#[test]
fn prepare_aot_enqueue() {
    let tb = TestBackend::new(RuntimeConfig { enabled: true, ..Default::default() });
    tb.prepare_aot(AotRequest {
        code_hash: alloy_primitives::keccak256(BYTECODE_RET42),
        code: Bytes::copy_from_slice(BYTECODE_RET42),
        spec_id: SpecId::CANCUN,
    });
}

#[test]
fn prepare_aot_batch_enqueue() {
    let tb = TestBackend::new(RuntimeConfig { enabled: true, ..Default::default() });
    let reqs = vec![
        AotRequest {
            code_hash: alloy_primitives::keccak256(BYTECODE_RET42),
            code: Bytes::copy_from_slice(BYTECODE_RET42),
            spec_id: SpecId::CANCUN,
        },
        AotRequest {
            code_hash: alloy_primitives::keccak256(BYTECODE_ADD),
            code: Bytes::copy_from_slice(BYTECODE_ADD),
            spec_id: SpecId::CANCUN,
        },
    ];
    tb.prepare_aot_batch(reqs);
}

#[test]
fn clear_resident() {
    let tb = TestBackend::new(RuntimeConfig { enabled: true, ..Default::default() });
    tb.clear_resident();

    let decision = tb.lookup(TestBackend::req_cancun(&[0x00]));
    assert!(matches!(decision, LookupDecision::Interpret(InterpretReason::NotReady)));
}

#[test]
fn channel_saturation_drops_events() {
    let tb = TestBackend::with_tuning_1w(RuntimeTuning {
        lookup_event_channel_capacity: 2,
        jit_hot_threshold: 1000,
        ..Default::default()
    });

    for _ in 0..200 {
        let _ = tb.lookup(TestBackend::req_cancun(&[0x00]));
    }

    let stats = tb.stats();
    assert!(stats.events_dropped > 0, "expected some dropped events with tiny channel");
    assert_eq!(stats.events_sent + stats.events_dropped, 200);
}

// ===========================================================================
// Tests: JIT compilation (require LLVM).
// ===========================================================================

#[test]
#[cfg(feature = "llvm")]
fn blocking_mode() {
    let tb = TestBackend::new(RuntimeConfig {
        blocking: true,
        tuning: RuntimeTuning { jit_worker_count: 1, ..Default::default() },
        ..Default::default()
    });

    let req = || TestBackend::req_cancun(BYTECODE_RET42);

    // First lookup should block until JIT finishes, then return Compiled.
    let decision = tb.lookup(req());
    assert!(
        matches!(&decision, LookupDecision::Compiled(p) if p.kind == ProgramKind::Jit),
        "blocking mode should return Compiled on first lookup, got: {decision:?}",
    );

    // Second lookup should hit the resident map immediately.
    assert!(matches!(tb.lookup(req()), LookupDecision::Compiled(_)));

    // Empty bytecodes return JitFailed (nothing to compile).
    let decision = tb.lookup(TestBackend::req_cancun(&[]));
    assert!(matches!(decision, LookupDecision::Interpret(InterpretReason::JitFailed)));
}

#[test]
#[cfg(feature = "llvm")]
fn jit_hotness_promotion() {
    let tb =
        TestBackend::with_tuning_1w(RuntimeTuning { jit_hot_threshold: 3, ..Default::default() });

    let req = || TestBackend::req_cancun(BYTECODE_RET42);

    // Below threshold: should remain NotReady.
    for _ in 0..2 {
        assert!(matches!(tb.lookup(req()), LookupDecision::Interpret(InterpretReason::NotReady)));
    }

    // Hit threshold and beyond: JIT should eventually compile.
    let p = tb.trigger_jit_cancun(BYTECODE_RET42);
    assert_eq!(p.kind, ProgramKind::Jit);

    let stats = tb.stats();
    assert!(stats.resident_entries >= 1);
    assert!(stats.lookup_hits >= 1);
}

#[test]
#[cfg(feature = "llvm")]
fn jit_max_bytecode_len_prevents_promotion() {
    let tb = TestBackend::with_tuning_1w(RuntimeTuning {
        jit_hot_threshold: 1,
        jit_max_bytecode_len: 4,
        ..Default::default()
    });

    let req = || TestBackend::req_cancun(BYTECODE_RET42);

    // Send many lookups to exceed any threshold.
    for _ in 0..20 {
        let _ = tb.lookup(req());
    }

    // Give the backend time to process events.
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Should still be NotReady — never promoted due to bytecode length.
    assert!(matches!(tb.lookup(req()), LookupDecision::Interpret(InterpretReason::NotReady)));
}

#[test]
#[cfg(feature = "llvm")]
fn clear_resident_discards_inflight_jit() {
    let tb =
        TestBackend::with_tuning_1w(RuntimeTuning { jit_hot_threshold: 1, ..Default::default() });

    let req = || TestBackend::req_cancun(BYTECODE_RET42);

    // Trigger JIT promotion.
    for _ in 0..5 {
        let _ = tb.lookup(req());
    }

    // Immediately clear — in-flight JIT results should be discarded.
    tb.clear_resident();

    // Give time for worker to finish and backend to process.
    std::thread::sleep(std::time::Duration::from_millis(500));

    // The result from the old generation should have been discarded.
    assert!(
        matches!(tb.lookup(req()), LookupDecision::Interpret(InterpretReason::NotReady)),
        "stale JIT result should not appear after clear_resident",
    );
}

#[test]
#[cfg(feature = "llvm")]
fn jit_memory_tracks_compilation() {
    let tb =
        TestBackend::with_tuning_1w(RuntimeTuning { jit_hot_threshold: 1, ..Default::default() });

    let baseline = tb.stats().jit_code_bytes;

    tb.trigger_jit_cancun(BYTECODE_RET42);

    let stats = tb.stats();
    assert!(stats.jit_code_bytes > baseline, "jit_code_bytes should increase after JIT");
    assert_eq!(stats.resident_entries, 1);
}

#[test]
#[cfg(feature = "llvm")]
fn compile_jit_sync_blocks() {
    let tb = TestBackend::default_1w();

    let code_hash = alloy_primitives::keccak256(BYTECODE_RET42);
    tb.compile_jit_sync(TestBackend::req_cancun(BYTECODE_RET42)).unwrap();

    let program = tb.get_compiled(code_hash, SpecId::CANCUN);
    assert!(program.is_some(), "program should be resident after compile_jit_sync");
    assert_eq!(program.unwrap().kind, ProgramKind::Jit);
}

#[test]
#[cfg(feature = "llvm")]
fn compile_jit_sync_deduplicates() {
    let compiled_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let compiled_count2 = compiled_count.clone();

    let tb = TestBackend::new(RuntimeConfig {
        enabled: true,
        tuning: RuntimeTuning { jit_worker_count: 1, ..Default::default() },
        on_compilation: Some(Arc::new(move |event| {
            if event.success {
                compiled_count2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        })),
        ..Default::default()
    });

    let code_hash = alloy_primitives::keccak256(BYTECODE_RET42);

    // Multiple sync compiles from different threads.
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let b = tb.backend.clone();
            let r = TestBackend::req_cancun(BYTECODE_RET42);
            std::thread::spawn(move || {
                let _ = b.compile_jit_sync(r);
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }

    assert!(tb.get_compiled(code_hash, SpecId::CANCUN).is_some());
    assert_eq!(
        compiled_count.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "should compile exactly once even with concurrent sync requests",
    );
}

// ===========================================================================
// Tests: multiple SpecIds.
// ===========================================================================

#[test]
#[cfg(feature = "llvm")]
fn multiple_spec_ids() {
    let tb = TestBackend::with_tuning(RuntimeTuning {
        jit_hot_threshold: 1,
        jit_worker_count: 2,
        ..Default::default()
    });

    let code_hash = alloy_primitives::keccak256(BYTECODE_RET42);

    for &spec_id in &[SpecId::CANCUN, SpecId::PRAGUE] {
        tb.trigger_jit(BYTECODE_RET42, spec_id);
    }

    assert!(tb.stats().resident_entries >= 2);

    let p1 = tb.get_compiled(code_hash, SpecId::CANCUN).unwrap();
    let p2 = tb.get_compiled(code_hash, SpecId::PRAGUE).unwrap();
    assert_eq!(p1.key.spec_id, SpecId::CANCUN);
    assert_eq!(p2.key.spec_id, SpecId::PRAGUE);
}

// ===========================================================================
// Tests: eviction.
// ===========================================================================

#[test]
#[cfg(feature = "llvm")]
fn idle_eviction() {
    let tb = TestBackend::with_tuning_1w(RuntimeTuning {
        jit_hot_threshold: 1,
        idle_evict_duration: Some(std::time::Duration::from_millis(100)),
        eviction_sweep_interval: std::time::Duration::from_millis(50),
        ..Default::default()
    });

    tb.trigger_jit_cancun(BYTECODE_RET42);
    assert_eq!(tb.stats().resident_entries, 1);

    // Stop hitting the entry and wait for idle eviction.
    tb.wait_resident_count(0);
    assert_eq!(tb.stats().resident_entries, 0);
}

#[test]
#[cfg(feature = "llvm")]
fn eviction_frees_jit_memory() {
    let tb = TestBackend::with_tuning_1w(RuntimeTuning {
        jit_hot_threshold: 1,
        idle_evict_duration: Some(std::time::Duration::from_millis(100)),
        eviction_sweep_interval: std::time::Duration::from_millis(50),
        ..Default::default()
    });

    let baseline = tb.stats().jit_code_bytes;

    let compiled = tb.trigger_jit_cancun(BYTECODE_RET42);
    let after_compile = tb.stats().jit_code_bytes;
    assert!(
        after_compile > baseline,
        "jit_code_bytes should increase: baseline={baseline}, after={after_compile}",
    );

    // Drop our reference so eviction can free it.
    drop(compiled);

    tb.wait_resident_count(0);
    std::thread::sleep(std::time::Duration::from_millis(100));

    let after_eviction = tb.stats().jit_code_bytes;
    assert!(
        after_eviction < after_compile,
        "jit_code_bytes should decrease: before={after_compile}, after={after_eviction}",
    );
}

#[test]
#[cfg(feature = "llvm")]
fn memory_budget_eviction() {
    let tb = TestBackend::with_tuning_1w(RuntimeTuning {
        jit_hot_threshold: 1,
        // Tiny budget: 1 byte — any entry should be evicted.
        resident_code_cache_bytes: 1,
        eviction_sweep_interval: std::time::Duration::from_millis(50),
        ..Default::default()
    });

    tb.trigger_jit_cancun(BYTECODE_RET42);

    // Wait for budget eviction to kick in.
    tb.wait_resident_count(0);
}

// ===========================================================================
// Tests: concurrency.
// ===========================================================================

#[test]
#[cfg(feature = "llvm")]
fn concurrent_lookup_same_key() {
    let tb = TestBackend::with_tuning(RuntimeTuning {
        jit_hot_threshold: 1,
        jit_worker_count: 2,
        ..Default::default()
    });

    tb.trigger_jit_cancun(BYTECODE_RET42);

    // Hammer from many threads.
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let b = tb.backend.clone();
            std::thread::spawn(move || {
                for _ in 0..100 {
                    let d = b.lookup(TestBackend::req_cancun(BYTECODE_RET42));
                    assert!(
                        matches!(d, LookupDecision::Compiled(_)),
                        "expected Compiled, got: {d:?}"
                    );
                }
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }

    assert!(tb.stats().lookup_hits >= 800, "expected ≥800 hits, got {}", tb.stats().lookup_hits);
}

#[test]
#[cfg(feature = "llvm")]
fn single_jit_admission_per_key() {
    let compiled_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let compiled_count2 = compiled_count.clone();

    let tb = TestBackend::new(RuntimeConfig {
        enabled: true,
        tuning: RuntimeTuning { jit_hot_threshold: 1, jit_worker_count: 2, ..Default::default() },
        on_compilation: Some(Arc::new(move |event| {
            if event.success {
                compiled_count2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        })),
        ..Default::default()
    });

    // Many lookups to trigger admission multiple times.
    for _ in 0..50 {
        let _ = tb.lookup(TestBackend::req_cancun(BYTECODE_RET42));
    }

    tb.wait_compiled(BYTECODE_RET42, SpecId::CANCUN);

    // Give extra time for any spurious duplicate compilations.
    std::thread::sleep(std::time::Duration::from_millis(200));

    assert_eq!(
        compiled_count.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "key should be compiled exactly once",
    );
}

#[test]
#[cfg(feature = "llvm")]
fn stats_accuracy_concurrent() {
    let tb =
        TestBackend::with_tuning_1w(RuntimeTuning { jit_hot_threshold: 1, ..Default::default() });

    tb.trigger_jit_cancun(BYTECODE_RET42);

    let before = tb.stats();

    let n_threads = 4u64;
    let n_lookups = 50u64;
    let threads: Vec<_> = (0..n_threads)
        .map(|_| {
            let b = tb.backend.clone();
            std::thread::spawn(move || {
                for _ in 0..n_lookups {
                    let _ = b.lookup(TestBackend::req_cancun(BYTECODE_RET42));
                }
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }

    let after = tb.stats();
    let total_lookups = n_threads * n_lookups;
    let new_hits = after.lookup_hits - before.lookup_hits;
    let new_events =
        (after.events_sent - before.events_sent) + (after.events_dropped - before.events_dropped);

    assert_eq!(new_hits, total_lookups, "all lookups should be hits");
    assert_eq!(new_events, total_lookups, "events sent + dropped should equal lookups");
}

// ===========================================================================
// Tests: lifecycle (clear, set_enabled, shutdown).
// ===========================================================================

#[test]
#[cfg(feature = "llvm")]
fn clear_during_active_work() {
    let tb =
        TestBackend::with_tuning_1w(RuntimeTuning { jit_hot_threshold: 1, ..Default::default() });

    // Fire many JIT requests to keep workers busy.
    for i in 0u8..20 {
        let bytecode = indexed_bytecode(i);
        let _ = tb.lookup(TestBackend::req_cancun(&bytecode));
    }

    // Clear while work is in-flight.
    tb.clear_resident();
    tb.clear_all();

    // Should not panic.
    std::thread::sleep(std::time::Duration::from_millis(500));
    let _stats = tb.stats();
}

#[test]
#[cfg(feature = "llvm")]
fn set_enabled_during_compilation() {
    let tb =
        TestBackend::with_tuning_1w(RuntimeTuning { jit_hot_threshold: 1, ..Default::default() });

    // Trigger JIT.
    for _ in 0..5 {
        let _ = tb.lookup(TestBackend::req_cancun(BYTECODE_RET42));
    }

    // Disable while compilation may be in progress.
    tb.set_enabled(false);
    assert!(matches!(
        tb.lookup(TestBackend::req_cancun(BYTECODE_RET42)),
        LookupDecision::Interpret(InterpretReason::Disabled)
    ));

    // Re-enable and check compilation eventually completes.
    tb.set_enabled(true);
    tb.wait_compiled(BYTECODE_RET42, SpecId::CANCUN);
}

// ===========================================================================
// Tests: on_compilation callback.
// ===========================================================================

#[test]
#[cfg(feature = "llvm")]
fn on_compilation_callback() {
    let events: Arc<Mutex<Vec<CompilationEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events2 = events.clone();

    let tb = TestBackend::new(RuntimeConfig {
        enabled: true,
        tuning: RuntimeTuning { jit_hot_threshold: 1, jit_worker_count: 1, ..Default::default() },
        on_compilation: Some(Arc::new(move |event| {
            events2.lock().unwrap().push(event);
        })),
        ..Default::default()
    });

    let code_hash = alloy_primitives::keccak256(BYTECODE_RET42);
    tb.trigger_jit_cancun(BYTECODE_RET42);

    let captured = events.lock().unwrap();
    assert_eq!(captured.len(), 1, "expected exactly one compilation event");
    let event = &captured[0];
    assert_eq!(event.code_hash, code_hash);
    assert_eq!(event.spec_id, SpecId::CANCUN);
    assert!(event.success);
    assert_eq!(event.kind, CompilationKind::Jit);
    assert!(!event.duration.is_zero());
}

// ===========================================================================
// Tests: AOT prepare and persistence.
// ===========================================================================

#[test]
#[cfg(feature = "llvm")]
fn prepare_aot_persist_and_load() {
    let store = Arc::new(TempDirStore::new());
    let tb = TestBackend::new(RuntimeConfig {
        enabled: true,
        store: Some(store.clone()),
        tuning: RuntimeTuning { jit_worker_count: 1, ..Default::default() },
        ..Default::default()
    });

    let code_hash = alloy_primitives::keccak256(BYTECODE_RET42);
    tb.prepare_aot(AotRequest {
        code_hash,
        code: Bytes::copy_from_slice(BYTECODE_RET42),
        spec_id: SpecId::CANCUN,
    });

    let p = tb.wait_compiled(BYTECODE_RET42, SpecId::CANCUN);
    assert_eq!(p.kind, ProgramKind::Aot);
    assert_eq!(store.stored_count(), 1);
}

#[test]
#[cfg(feature = "llvm")]
fn prepare_aot_batch_persist_and_load() {
    let store = Arc::new(TempDirStore::new());
    let tb = TestBackend::new(RuntimeConfig {
        enabled: true,
        store: Some(store.clone()),
        tuning: RuntimeTuning { jit_worker_count: 1, ..Default::default() },
        ..Default::default()
    });

    let bytecodes: &[&[u8]] = &[BYTECODE_RET42, BYTECODE_ADD];
    let hashes: Vec<B256> = bytecodes.iter().map(alloy_primitives::keccak256).collect();

    let reqs: Vec<AotRequest> = bytecodes
        .iter()
        .zip(&hashes)
        .map(|(code, hash)| AotRequest {
            code_hash: *hash,
            code: Bytes::copy_from_slice(code),
            spec_id: SpecId::CANCUN,
        })
        .collect();
    tb.prepare_aot_batch(reqs);

    for bc in bytecodes {
        tb.wait_compiled(bc, SpecId::CANCUN);
    }
    assert_eq!(store.stored_count(), 2);
}

#[test]
#[cfg(feature = "llvm")]
fn aot_artifacts_survive_restart() {
    let store = Arc::new(TempDirStore::new());
    let code_hash = alloy_primitives::keccak256(BYTECODE_RET42);

    // First backend: compile and persist an AOT artifact.
    {
        let tb = TestBackend::new(RuntimeConfig {
            enabled: true,
            store: Some(store.clone()),
            tuning: RuntimeTuning { jit_worker_count: 1, ..Default::default() },
            ..Default::default()
        });

        tb.prepare_aot(AotRequest {
            code_hash,
            code: Bytes::copy_from_slice(BYTECODE_RET42),
            spec_id: SpecId::CANCUN,
        });
        tb.wait_compiled(BYTECODE_RET42, SpecId::CANCUN);
    }

    assert_eq!(store.stored_count(), 1);

    // Second backend: should preload the artifact at startup.
    {
        let tb = TestBackend::new(RuntimeConfig {
            enabled: true,
            store: Some(store),
            ..Default::default()
        });

        let decision = tb.lookup(TestBackend::req_cancun(BYTECODE_RET42));
        assert!(
            matches!(&decision, LookupDecision::Compiled(p) if p.kind == ProgramKind::Aot),
            "expected AOT hit after restart, got: {decision:?}",
        );
        assert_eq!(tb.stats().resident_entries, 1);
    }
}

#[test]
#[cfg(feature = "llvm")]
fn preload_aot_seeds_resident() {
    let store = Arc::new(TempDirStore::new());
    let code_hash = alloy_primitives::keccak256(BYTECODE_RET42);

    // First backend: compile and persist.
    {
        let tb = TestBackend::new(RuntimeConfig {
            enabled: true,
            store: Some(store.clone()),
            tuning: RuntimeTuning { jit_worker_count: 1, ..Default::default() },
            ..Default::default()
        });

        tb.prepare_aot(AotRequest {
            code_hash,
            code: Bytes::copy_from_slice(BYTECODE_RET42),
            spec_id: SpecId::CANCUN,
        });
        tb.wait_compiled(BYTECODE_RET42, SpecId::CANCUN);
    }

    // Second backend: preloaded AOT should be available immediately.
    {
        let tb = TestBackend::new(RuntimeConfig {
            enabled: true,
            store: Some(store),
            ..Default::default()
        });
        assert_eq!(tb.stats().resident_entries, 1);
    }
}

//! Tests for the runtime module.

use super::*;
use alloy_primitives::{B256, Bytes};
use revm_primitives::hardfork::SpecId;
use std::sync::{Arc, Mutex};

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

#[test]
fn start_no_store() {
    let backend = JitBackend::start(RuntimeConfig::default()).unwrap();
    assert_eq!(backend.stats().resident_entries, 0);
    backend.shutdown().unwrap();
}

#[test]
fn start_empty_store() {
    let config =
        RuntimeConfig { enabled: true, store: Some(Arc::new(EmptyStore)), ..Default::default() };
    let backend = JitBackend::start(config).unwrap();
    assert_eq!(backend.stats().resident_entries, 0);
    backend.shutdown().unwrap();
}

#[test]
fn lookup_disabled() {
    let config = RuntimeConfig { enabled: false, ..Default::default() };
    let backend = JitBackend::start(config).unwrap();

    let req = LookupRequest { code_hash: B256::ZERO, code: Bytes::new(), spec_id: SpecId::CANCUN };
    let decision = backend.lookup(req);
    assert!(matches!(decision, LookupDecision::Interpret(InterpretReason::Disabled)));

    let stats = backend.stats();
    assert_eq!(stats.lookup_hits, 0);
    assert_eq!(stats.lookup_misses, 0);

    backend.shutdown().unwrap();
}

#[test]
fn lookup_miss_when_enabled() {
    let config = RuntimeConfig { enabled: true, ..Default::default() };
    let backend = JitBackend::start(config).unwrap();

    let req = LookupRequest {
        code_hash: B256::ZERO,
        code: Bytes::from_static(&[0x00]),
        spec_id: SpecId::CANCUN,
    };
    let decision = backend.lookup(req);
    assert!(matches!(decision, LookupDecision::Interpret(InterpretReason::NotReady)));

    let stats = backend.stats();
    assert_eq!(stats.lookup_misses, 1);
    assert_eq!(stats.lookup_hits, 0);

    backend.shutdown().unwrap();
}

#[test]
fn set_enabled_toggle() {
    let config = RuntimeConfig { enabled: false, ..Default::default() };
    let backend = JitBackend::start(config).unwrap();

    let req = LookupRequest { code_hash: B256::ZERO, code: Bytes::new(), spec_id: SpecId::CANCUN };

    // Initially disabled.
    assert!(matches!(
        backend.lookup(req.clone()),
        LookupDecision::Interpret(InterpretReason::Disabled)
    ));

    // Enable.
    backend.set_enabled(true);
    assert!(matches!(
        backend.lookup(req.clone()),
        LookupDecision::Interpret(InterpretReason::NotReady)
    ));

    // Disable again.
    backend.set_enabled(false);
    assert!(matches!(backend.lookup(req), LookupDecision::Interpret(InterpretReason::Disabled)));

    backend.shutdown().unwrap();
}

#[test]
fn events_sent_on_lookup() {
    let config = RuntimeConfig { enabled: true, ..Default::default() };
    let backend = JitBackend::start(config).unwrap();

    for _ in 0..10 {
        let req =
            LookupRequest { code_hash: B256::ZERO, code: Bytes::new(), spec_id: SpecId::CANCUN };
        let _ = backend.lookup(req);
    }

    let stats = backend.stats();
    assert_eq!(stats.events_sent, 10);
    assert_eq!(stats.events_dropped, 0);

    backend.shutdown().unwrap();
}

#[test]
fn drop_shuts_down_backend() {
    let backend = JitBackend::start(RuntimeConfig::default()).unwrap();
    let backend2 = backend.clone();
    drop(backend);

    // Lookups still work (no panic) — coordinator is still running because backend2 holds a ref.
    let req = LookupRequest { code_hash: B256::ZERO, code: Bytes::new(), spec_id: SpecId::CANCUN };
    let _ = backend2.lookup(req);
}

#[test]
fn backend_clone() {
    let config = RuntimeConfig { enabled: true, ..Default::default() };
    let b1 = JitBackend::start(config).unwrap();
    let b2 = b1.clone();

    let req = LookupRequest { code_hash: B256::ZERO, code: Bytes::new(), spec_id: SpecId::CANCUN };
    let _ = b1.lookup(req.clone());
    let _ = b2.lookup(req);

    // Both share the same stats.
    let stats = b1.stats();
    assert_eq!(stats.lookup_misses, 2);

    b1.shutdown().unwrap();
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

#[test]
fn startup_store_failure() {
    let config =
        RuntimeConfig { enabled: true, store: Some(Arc::new(FailingStore)), ..Default::default() };
    let result = JitBackend::start(config);
    assert!(result.is_err());
}

#[test]
fn compile_jit_enqueue() {
    let config = RuntimeConfig { enabled: true, ..Default::default() };
    let backend = JitBackend::start(config).unwrap();

    let req = LookupRequest {
        code_hash: B256::repeat_byte(0x01),
        code: Bytes::from_static(&[0x60, 0x00]),
        spec_id: SpecId::CANCUN,
    };
    backend.compile_jit(req);

    backend.shutdown().unwrap();
}

#[test]
fn prepare_aot_enqueue() {
    let config = RuntimeConfig { enabled: true, ..Default::default() };
    let backend = JitBackend::start(config).unwrap();

    let req = super::AotRequest {
        code_hash: B256::repeat_byte(0x02),
        code: Bytes::from_static(&[0x60, 0x00]),
        spec_id: SpecId::CANCUN,
    };
    backend.prepare_aot(req).unwrap();

    backend.shutdown().unwrap();
}

#[test]
fn prepare_aot_batch_enqueue() {
    let config = RuntimeConfig { enabled: true, ..Default::default() };
    let backend = JitBackend::start(config).unwrap();

    let reqs = vec![
        super::AotRequest {
            code_hash: B256::repeat_byte(0x03),
            code: Bytes::from_static(&[0x60, 0x00]),
            spec_id: SpecId::CANCUN,
        },
        super::AotRequest {
            code_hash: B256::repeat_byte(0x04),
            code: Bytes::from_static(&[0x60, 0x01]),
            spec_id: SpecId::CANCUN,
        },
    ];
    backend.prepare_aot_batch(reqs).unwrap();

    backend.shutdown().unwrap();
}

#[test]
fn clear_resident() {
    let config = RuntimeConfig { enabled: true, ..Default::default() };
    let backend = JitBackend::start(config).unwrap();

    backend.clear_resident().unwrap();

    // After clear, lookups should miss.
    let req = LookupRequest {
        code_hash: B256::ZERO,
        code: Bytes::from_static(&[0x00]),
        spec_id: SpecId::CANCUN,
    };
    let decision = backend.lookup(req);
    assert!(matches!(decision, LookupDecision::Interpret(InterpretReason::NotReady)));

    backend.shutdown().unwrap();
}

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

#[test]
#[cfg(feature = "llvm")]
fn jit_hotness_promotion() {
    let config = RuntimeConfig {
        enabled: true,
        tuning: RuntimeTuning { jit_hot_threshold: 3, jit_worker_count: 1, ..Default::default() },
        ..Default::default()
    };
    let backend = JitBackend::start(config).unwrap();

    // Simple bytecode: PUSH1 0x42 PUSH0 MSTORE PUSH1 0x20 PUSH0 RETURN.
    let bytecode: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];
    let code_hash = alloy_primitives::keccak256(bytecode);
    let req = || LookupRequest {
        code_hash,
        code: Bytes::copy_from_slice(bytecode),
        spec_id: SpecId::CANCUN,
    };

    // Below threshold: should remain NotReady.
    for _ in 0..2 {
        assert!(matches!(
            backend.lookup(req()),
            LookupDecision::Interpret(InterpretReason::NotReady)
        ));
    }

    // Hit threshold and beyond: JIT should eventually compile.
    for _ in 0..5 {
        let _ = backend.lookup(req());
    }

    poll_until(std::time::Duration::from_secs(30), || match backend.lookup(req()) {
        LookupDecision::Compiled(p) => {
            assert_eq!(p.kind, ProgramKind::Jit);
            Some(())
        }
        _ => None,
    });

    let stats = backend.stats();
    assert!(stats.resident_entries >= 1);
    assert!(stats.lookup_hits >= 1);

    backend.shutdown().unwrap();
}

#[test]
#[cfg(feature = "llvm")]
fn jit_max_bytecode_len_prevents_promotion() {
    let config = RuntimeConfig {
        enabled: true,
        tuning: RuntimeTuning {
            jit_hot_threshold: 1,
            jit_max_bytecode_len: 4,
            jit_worker_count: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let backend = JitBackend::start(config).unwrap();

    // 8-byte bytecode exceeds the 4-byte limit.
    let bytecode: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];
    let code_hash = alloy_primitives::keccak256(bytecode);
    let req = || LookupRequest {
        code_hash,
        code: Bytes::copy_from_slice(bytecode),
        spec_id: SpecId::CANCUN,
    };

    // Send many lookups to exceed any threshold.
    for _ in 0..20 {
        let _ = backend.lookup(req());
    }

    // Give the coordinator time to process events.
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Should still be NotReady — never promoted due to bytecode length.
    assert!(matches!(backend.lookup(req()), LookupDecision::Interpret(InterpretReason::NotReady)));

    backend.shutdown().unwrap();
}

#[test]
#[cfg(feature = "llvm")]
fn clear_resident_discards_inflight_jit() {
    let config = RuntimeConfig {
        enabled: true,
        tuning: RuntimeTuning { jit_hot_threshold: 1, jit_worker_count: 1, ..Default::default() },
        ..Default::default()
    };
    let backend = JitBackend::start(config).unwrap();

    // Simple bytecode: PUSH1 0x42 PUSH0 MSTORE PUSH1 0x20 PUSH0 RETURN.
    let bytecode: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];
    let code_hash = alloy_primitives::keccak256(bytecode);
    let req = || LookupRequest {
        code_hash,
        code: Bytes::copy_from_slice(bytecode),
        spec_id: SpecId::CANCUN,
    };

    // Trigger JIT promotion.
    for _ in 0..5 {
        let _ = backend.lookup(req());
    }

    // Immediately clear — in-flight JIT results should be discarded.
    backend.clear_resident().unwrap();

    // Give time for worker to finish and coordinator to process.
    std::thread::sleep(std::time::Duration::from_millis(500));

    // The result from the old generation should have been discarded.
    // The entry should not be in the resident map.
    assert!(
        matches!(backend.lookup(req()), LookupDecision::Interpret(InterpretReason::NotReady)),
        "stale JIT result should not appear after clear_resident",
    );

    backend.shutdown().unwrap();
}

#[test]
#[cfg(feature = "llvm")]
fn resident_bytes_tracks_jit() {
    let config = RuntimeConfig {
        enabled: true,
        tuning: RuntimeTuning { jit_hot_threshold: 1, jit_worker_count: 1, ..Default::default() },
        ..Default::default()
    };
    let backend = JitBackend::start(config).unwrap();

    assert_eq!(backend.stats().resident_bytes, 0);

    let bytecode: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];
    let code_hash = alloy_primitives::keccak256(bytecode);
    let req = || LookupRequest {
        code_hash,
        code: Bytes::copy_from_slice(bytecode),
        spec_id: SpecId::CANCUN,
    };

    // Trigger JIT and wait for it.
    for _ in 0..5 {
        let _ = backend.lookup(req());
    }
    poll_until(std::time::Duration::from_secs(30), || match backend.lookup(req()) {
        LookupDecision::Compiled(_) => Some(()),
        _ => None,
    });

    let stats = backend.stats();
    assert!(stats.resident_bytes > 0, "resident_bytes should be non-zero after JIT");
    assert_eq!(stats.resident_entries, 1);

    // Clear and verify bytes go back to zero.
    backend.clear_resident().unwrap();
    // Give coordinator time to process the clear.
    std::thread::sleep(std::time::Duration::from_millis(100));
    let stats = backend.stats();
    assert_eq!(stats.resident_bytes, 0);
    assert_eq!(stats.resident_entries, 0);

    backend.shutdown().unwrap();
}

#[test]
#[cfg(feature = "llvm")]
fn idle_eviction() {
    let config = RuntimeConfig {
        enabled: true,
        tuning: RuntimeTuning {
            jit_hot_threshold: 1,
            jit_worker_count: 1,
            // Evict after 100ms idle.
            idle_evict_duration: Some(std::time::Duration::from_millis(100)),
            // Sweep every 50ms.
            eviction_sweep_interval: std::time::Duration::from_millis(50),
            ..Default::default()
        },
        ..Default::default()
    };
    let backend = JitBackend::start(config).unwrap();

    let bytecode: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];
    let code_hash = alloy_primitives::keccak256(bytecode);
    let req = || LookupRequest {
        code_hash,
        code: Bytes::copy_from_slice(bytecode),
        spec_id: SpecId::CANCUN,
    };

    // Trigger JIT and wait for compilation.
    for _ in 0..5 {
        let _ = backend.lookup(req());
    }
    poll_until(std::time::Duration::from_secs(30), || match backend.lookup(req()) {
        LookupDecision::Compiled(_) => Some(()),
        _ => None,
    });

    assert_eq!(backend.stats().resident_entries, 1);

    // Stop hitting the entry and wait for idle eviction.
    poll_until(std::time::Duration::from_secs(5), || {
        let stats = backend.stats();
        if stats.resident_entries == 0 { Some(()) } else { None }
    });

    assert_eq!(backend.stats().resident_entries, 0);
    assert_eq!(backend.stats().resident_bytes, 0);

    backend.shutdown().unwrap();
}

#[test]
#[cfg(feature = "llvm")]
fn memory_budget_eviction() {
    let config = RuntimeConfig {
        enabled: true,
        tuning: RuntimeTuning {
            jit_hot_threshold: 1,
            jit_worker_count: 1,
            // Tiny budget: 1 byte — any entry should be evicted.
            resident_code_cache_bytes: 1,
            // Sweep frequently.
            eviction_sweep_interval: std::time::Duration::from_millis(50),
            ..Default::default()
        },
        ..Default::default()
    };
    let backend = JitBackend::start(config).unwrap();

    let bytecode: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];
    let code_hash = alloy_primitives::keccak256(bytecode);
    let req = || LookupRequest {
        code_hash,
        code: Bytes::copy_from_slice(bytecode),
        spec_id: SpecId::CANCUN,
    };

    // Trigger JIT and wait for compilation.
    for _ in 0..5 {
        let _ = backend.lookup(req());
    }
    poll_until(std::time::Duration::from_secs(30), || match backend.lookup(req()) {
        LookupDecision::Compiled(_) => Some(()),
        _ => None,
    });

    // Wait for budget eviction to kick in.
    poll_until(std::time::Duration::from_secs(5), || {
        // Don't send lookup events so the entry doesn't get re-promoted.
        let stats = backend.stats();
        if stats.resident_entries == 0 { Some(()) } else { None }
    });

    assert_eq!(backend.stats().resident_entries, 0);
    assert_eq!(backend.stats().resident_bytes, 0);

    backend.shutdown().unwrap();
}

#[test]
#[cfg(feature = "llvm")]
fn preload_aot_seeds_resident_bytes() {
    let store = Arc::new(TempDirStore::new());

    // First backend: compile and persist an AOT artifact.
    let bytecode: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];
    let code_hash = alloy_primitives::keccak256(bytecode);
    {
        let config = RuntimeConfig {
            enabled: true,
            store: Some(store.clone()),
            tuning: RuntimeTuning { jit_worker_count: 1, ..Default::default() },
            ..Default::default()
        };
        let backend = JitBackend::start(config).unwrap();

        backend
            .prepare_aot(AotRequest {
                code_hash,
                code: Bytes::copy_from_slice(bytecode),
                spec_id: SpecId::CANCUN,
            })
            .unwrap();

        poll_until(std::time::Duration::from_secs(30), || {
            match backend.lookup(LookupRequest {
                code_hash,
                code: Bytes::copy_from_slice(bytecode),
                spec_id: SpecId::CANCUN,
            }) {
                LookupDecision::Compiled(_) => Some(()),
                _ => None,
            }
        });

        backend.shutdown().unwrap();
    }

    // Second backend: preloaded AOT should have resident_bytes > 0 immediately.
    {
        let config = RuntimeConfig { enabled: true, store: Some(store), ..Default::default() };
        let backend = JitBackend::start(config).unwrap();

        let stats = backend.stats();
        assert_eq!(stats.resident_entries, 1);
        assert!(stats.resident_bytes > 0, "preloaded AOT should seed resident_bytes");

        backend.shutdown().unwrap();
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

#[test]
#[cfg(feature = "llvm")]
fn prepare_aot_persist_and_load() {
    let store = Arc::new(TempDirStore::new());
    let config = RuntimeConfig {
        enabled: true,
        store: Some(store.clone()),
        tuning: RuntimeTuning { jit_worker_count: 1, ..Default::default() },
        ..Default::default()
    };
    let backend = JitBackend::start(config).unwrap();

    // Simple bytecode: PUSH1 0x42 PUSH0 MSTORE PUSH1 0x20 PUSH0 RETURN.
    let bytecode: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];
    let code_hash = alloy_primitives::keccak256(bytecode);

    let req =
        AotRequest { code_hash, code: Bytes::copy_from_slice(bytecode), spec_id: SpecId::CANCUN };
    backend.prepare_aot(req).unwrap();

    // Poll until the artifact appears in the resident map.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let req = LookupRequest {
            code_hash,
            code: Bytes::copy_from_slice(bytecode),
            spec_id: SpecId::CANCUN,
        };
        if let LookupDecision::Compiled(program) = backend.lookup(req) {
            assert_eq!(program.kind, ProgramKind::Aot);
            break;
        }
        assert!(std::time::Instant::now() < deadline, "timed out waiting for AOT compilation");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // Verify the artifact was persisted to the store.
    assert_eq!(store.stored_count(), 1);

    backend.shutdown().unwrap();
}

#[test]
#[cfg(feature = "llvm")]
fn prepare_aot_batch_persist_and_load() {
    let store = Arc::new(TempDirStore::new());
    let config = RuntimeConfig {
        enabled: true,
        store: Some(store.clone()),
        tuning: RuntimeTuning { jit_worker_count: 1, ..Default::default() },
        ..Default::default()
    };
    let backend = JitBackend::start(config).unwrap();

    let bytecodes: &[&[u8]] = &[
        &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3],
        &[0x60, 0x01, 0x60, 0x01, 0x01, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3],
    ];
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
    backend.prepare_aot_batch(reqs).unwrap();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let all_ready = hashes.iter().zip(bytecodes.iter()).all(|(hash, code)| {
            let req = LookupRequest {
                code_hash: *hash,
                code: Bytes::copy_from_slice(code),
                spec_id: SpecId::CANCUN,
            };
            matches!(backend.lookup(req), LookupDecision::Compiled(_))
        });
        if all_ready {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for AOT batch compilation",
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    assert_eq!(store.stored_count(), 2);

    backend.shutdown().unwrap();
}

#[test]
#[cfg(feature = "llvm")]
fn aot_artifacts_survive_restart() {
    let store = Arc::new(TempDirStore::new());

    // First backend: compile and persist an AOT artifact.
    {
        let config = RuntimeConfig {
            enabled: true,
            store: Some(store.clone()),
            tuning: RuntimeTuning { jit_worker_count: 1, ..Default::default() },
            ..Default::default()
        };
        let backend = JitBackend::start(config).unwrap();

        let bytecode: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];
        let code_hash = alloy_primitives::keccak256(bytecode);

        backend
            .prepare_aot(AotRequest {
                code_hash,
                code: Bytes::copy_from_slice(bytecode),
                spec_id: SpecId::CANCUN,
            })
            .unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let req = LookupRequest {
                code_hash,
                code: Bytes::copy_from_slice(bytecode),
                spec_id: SpecId::CANCUN,
            };
            if matches!(backend.lookup(req), LookupDecision::Compiled(_)) {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "timed out waiting for AOT compilation",);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        backend.shutdown().unwrap();
    }

    assert_eq!(store.stored_count(), 1);

    // Second backend: should preload the artifact at startup.
    {
        let config = RuntimeConfig { enabled: true, store: Some(store), ..Default::default() };
        let backend = JitBackend::start(config).unwrap();

        let bytecode: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];
        let code_hash = alloy_primitives::keccak256(bytecode);

        // Should be available immediately from AOT preload — no waiting.
        let req = LookupRequest {
            code_hash,
            code: Bytes::copy_from_slice(bytecode),
            spec_id: SpecId::CANCUN,
        };
        let decision = backend.lookup(req);
        assert!(
            matches!(&decision, LookupDecision::Compiled(p) if p.kind == ProgramKind::Aot),
            "expected AOT hit after restart, got: {decision:?}",
        );

        assert_eq!(backend.stats().resident_entries, 1);

        backend.shutdown().unwrap();
    }
}

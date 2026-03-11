//! Tests for the runtime module.

use super::*;
use alloy_primitives::B256;
use revm_primitives::hardfork::SpecId;
use std::sync::Arc;

/// A no-op artifact store that returns no artifacts.
struct EmptyStore;

impl ArtifactStore for EmptyStore {
    fn load_all(&self) -> Result<Vec<(ArtifactKey, StoredArtifact)>, StorageError> {
        Ok(vec![])
    }

    fn load(&self, _key: &ArtifactKey) -> Result<Option<StoredArtifact>, StorageError> {
        Ok(None)
    }

    fn store(&self, _key: &ArtifactKey, _artifact: &StoredArtifact) -> Result<(), StorageError> {
        Ok(())
    }

    fn delete(&self, _key: &ArtifactKey) -> Result<(), StorageError> {
        Ok(())
    }

    fn clear(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

#[test]
fn start_no_store() {
    let coord = JitCoordinator::start(RuntimeConfig::default()).unwrap();
    let handle = coord.handle();
    assert_eq!(handle.stats().resident_entries, 0);
    coord.shutdown().unwrap();
}

#[test]
fn start_empty_store() {
    let config = RuntimeConfig {
        enabled: true,
        store: Some(Arc::new(EmptyStore)),
        ..Default::default()
    };
    let coord = JitCoordinator::start(config).unwrap();
    let handle = coord.handle();
    assert_eq!(handle.stats().resident_entries, 0);
    coord.shutdown().unwrap();
}

#[test]
fn lookup_disabled() {
    let config = RuntimeConfig { enabled: false, ..Default::default() };
    let coord = JitCoordinator::start(config).unwrap();
    let handle = coord.handle();

    let req = LookupRequest {
        code_hash: B256::ZERO,
        code: &[],
        spec_id: SpecId::CANCUN,
    };
    let decision = handle.lookup(req);
    assert!(matches!(decision, LookupDecision::Interpret(InterpretReason::Disabled)));

    let stats = handle.stats();
    assert_eq!(stats.lookup_disabled, 1);
    assert_eq!(stats.lookup_hits, 0);
    assert_eq!(stats.lookup_misses, 0);

    coord.shutdown().unwrap();
}

#[test]
fn lookup_miss_when_enabled() {
    let config = RuntimeConfig { enabled: true, ..Default::default() };
    let coord = JitCoordinator::start(config).unwrap();
    let handle = coord.handle();

    let req = LookupRequest {
        code_hash: B256::ZERO,
        code: &[0x00],
        spec_id: SpecId::CANCUN,
    };
    let decision = handle.lookup(req);
    assert!(matches!(decision, LookupDecision::Interpret(InterpretReason::NotReady)));

    let stats = handle.stats();
    assert_eq!(stats.lookup_misses, 1);
    assert_eq!(stats.lookup_hits, 0);

    coord.shutdown().unwrap();
}

#[test]
fn set_enabled_toggle() {
    let config = RuntimeConfig { enabled: false, ..Default::default() };
    let coord = JitCoordinator::start(config).unwrap();
    let handle = coord.handle();

    let req = LookupRequest {
        code_hash: B256::ZERO,
        code: &[],
        spec_id: SpecId::CANCUN,
    };

    // Initially disabled.
    assert!(matches!(handle.lookup(req.clone()), LookupDecision::Interpret(InterpretReason::Disabled)));

    // Enable.
    handle.set_enabled(true);
    assert!(matches!(handle.lookup(req.clone()), LookupDecision::Interpret(InterpretReason::NotReady)));

    // Disable again.
    handle.set_enabled(false);
    assert!(matches!(handle.lookup(req), LookupDecision::Interpret(InterpretReason::Disabled)));

    coord.shutdown().unwrap();
}

#[test]
fn events_sent_on_lookup() {
    let config = RuntimeConfig { enabled: true, ..Default::default() };
    let coord = JitCoordinator::start(config).unwrap();
    let handle = coord.handle();

    for _ in 0..10 {
        let req = LookupRequest {
            code_hash: B256::ZERO,
            code: &[],
            spec_id: SpecId::CANCUN,
        };
        let _ = handle.lookup(req);
    }

    let stats = handle.stats();
    assert_eq!(stats.events_sent, 10);
    assert_eq!(stats.events_dropped, 0);

    coord.shutdown().unwrap();
}

#[test]
fn drop_shuts_down_coordinator() {
    let coord = JitCoordinator::start(RuntimeConfig::default()).unwrap();
    let handle = coord.handle();
    drop(coord);

    // Lookups still work (no panic), events will be dropped since coordinator is gone.
    let req = LookupRequest {
        code_hash: B256::ZERO,
        code: &[],
        spec_id: SpecId::CANCUN,
    };
    let _ = handle.lookup(req);
}

#[test]
fn handle_clone() {
    let config = RuntimeConfig { enabled: true, ..Default::default() };
    let coord = JitCoordinator::start(config).unwrap();
    let h1 = coord.handle();
    let h2 = h1.clone();

    let req = LookupRequest {
        code_hash: B256::ZERO,
        code: &[],
        spec_id: SpecId::CANCUN,
    };
    let _ = h1.lookup(req.clone());
    let _ = h2.lookup(req);

    // Both share the same stats.
    let stats = h1.stats();
    assert_eq!(stats.lookup_misses, 2);

    coord.shutdown().unwrap();
}

/// A store that fails on load_all.
struct FailingStore;

impl ArtifactStore for FailingStore {
    fn load_all(&self) -> Result<Vec<(ArtifactKey, StoredArtifact)>, StorageError> {
        Err(StorageError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "boom",
        )))
    }

    fn load(&self, _key: &ArtifactKey) -> Result<Option<StoredArtifact>, StorageError> {
        Ok(None)
    }

    fn store(&self, _key: &ArtifactKey, _artifact: &StoredArtifact) -> Result<(), StorageError> {
        Ok(())
    }

    fn delete(&self, _key: &ArtifactKey) -> Result<(), StorageError> {
        Ok(())
    }

    fn clear(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

#[test]
fn startup_store_failure() {
    let config = RuntimeConfig {
        enabled: true,
        store: Some(Arc::new(FailingStore)),
        ..Default::default()
    };
    let result = JitCoordinator::start(config);
    assert!(result.is_err());
}

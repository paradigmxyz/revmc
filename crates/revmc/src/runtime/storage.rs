//! Artifact storage trait and data model.

use crate::runtime::error::StorageError;
use alloy_primitives::B256;
use revm_primitives::hardfork::SpecId;
use revmc_backend::{OptimizationLevel, Target};

/// Runtime cache key: the minimal identity for a compiled program at runtime.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RuntimeCacheKey {
    /// The code hash of the contract bytecode.
    pub code_hash: B256,
    /// The EVM spec (hardfork) the code was compiled against.
    pub spec_id: SpecId,
}

/// Full artifact identity for persisted artifacts.
///
/// Persisted artifacts must match all fields of this key to be loaded.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactKey {
    /// The runtime cache key (code_hash + spec_id).
    pub runtime: RuntimeCacheKey,
    /// The compiler backend used.
    pub backend: BackendSelection,
    /// The compilation target.
    pub target: Target,
    /// The optimization level used.
    pub opt_level: OptimizationLevel,
    /// The revmc semver at compile time.
    pub revmc_semver: String,
    /// A fingerprint of the compiler configuration.
    pub compiler_fingerprint: String,
    /// The ABI version of the compiled artifact.
    pub abi_version: u32,
}

/// A stored artifact consisting of a manifest and the compiled dylib bytes.
#[derive(Clone, Debug)]
pub struct StoredArtifact {
    /// Metadata about the artifact.
    pub manifest: ArtifactManifest,
    /// The raw shared-library bytes.
    pub dylib_bytes: Vec<u8>,
}

/// Metadata for a stored artifact.
#[derive(Clone, Debug)]
pub struct ArtifactManifest {
    /// The full artifact key.
    pub artifact_key: ArtifactKey,
    /// The symbol name to look up in the loaded library.
    pub symbol_name: String,
    /// Length of the original bytecode.
    pub bytecode_len: usize,
    /// Length of the compiled artifact in bytes.
    pub artifact_len: usize,
    /// Creation timestamp (unix seconds).
    pub created_at_unix_secs: u64,
    /// SHA-256 digest of the dylib bytes.
    pub sha256: [u8; 32],
}

/// Backend selection for compilation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BackendSelection {
    /// Automatically select the best available backend.
    #[default]
    Auto,
    /// Use the LLVM backend.
    Llvm,
    /// Use the Cranelift backend.
    Cranelift,
}

/// Trait for loading and storing compiled artifacts.
pub trait ArtifactStore: Send + Sync + 'static {
    /// Loads all available artifacts from storage.
    fn load_all(&self) -> Result<Vec<(ArtifactKey, StoredArtifact)>, StorageError>;

    /// Loads a single artifact by key.
    fn load(&self, key: &ArtifactKey) -> Result<Option<StoredArtifact>, StorageError>;

    /// Stores an artifact.
    fn store(&self, key: &ArtifactKey, artifact: &StoredArtifact) -> Result<(), StorageError>;

    /// Deletes an artifact by key.
    fn delete(&self, key: &ArtifactKey) -> Result<(), StorageError>;

    /// Clears all stored artifacts.
    fn clear(&self) -> Result<(), StorageError>;
}

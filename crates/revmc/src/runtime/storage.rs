//! Artifact storage trait and data model.

use crate::eyre;
use alloy_primitives::B256;
use revm_primitives::hardfork::SpecId;
use revmc_backend::OptimizationLevel;
use std::path::PathBuf;

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
    /// The optimization level used.
    pub opt_level: OptimizationLevel,
}

/// A stored artifact consisting of a manifest and a path to the compiled dylib.
#[derive(Clone, Debug)]
pub struct StoredArtifact {
    /// Metadata about the artifact.
    pub manifest: ArtifactManifest,
    /// Path to the shared library on disk. The store owns and manages these files.
    pub dylib_path: PathBuf,
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
    /// Keccak-256 digest of the dylib bytes.
    pub content_hash: [u8; 32],
}

/// Backend selection for compilation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BackendSelection {
    /// Automatically select the best available backend.
    #[default]
    Auto,
    /// Use the LLVM backend.
    Llvm,
}

/// Trait for loading and storing compiled artifacts.
///
/// Implementations manage artifact files on the filesystem. The store owns the dylib files and
/// returns paths to them. The backend loads shared libraries directly from these paths.
pub trait ArtifactStore: Send + Sync + 'static {
    /// Loads all available artifacts from storage.
    fn load_all(&self) -> eyre::Result<Vec<(ArtifactKey, StoredArtifact)>>;

    /// Loads a single artifact by key.
    fn load(&self, key: &ArtifactKey) -> eyre::Result<Option<StoredArtifact>>;

    /// Stores an artifact. The `dylib_bytes` are the raw shared-library bytes to persist.
    /// The store writes them to disk and the returned path (via subsequent `load`) points there.
    fn store(
        &self,
        key: &ArtifactKey,
        manifest: &ArtifactManifest,
        dylib_bytes: &[u8],
    ) -> eyre::Result<()>;

    /// Deletes an artifact by key.
    fn delete(&self, key: &ArtifactKey) -> eyre::Result<()>;

    /// Clears all stored artifacts.
    fn clear(&self) -> eyre::Result<()>;
}

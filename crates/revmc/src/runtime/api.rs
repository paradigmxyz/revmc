//! Public types and handle methods.

use crate::{EvmCompilerFn, runtime::storage::RuntimeCacheKey};
use alloy_primitives::B256;
use revm_primitives::hardfork::SpecId;
use std::sync::Arc;

/// Request to look up a compiled function.
#[derive(Clone, Debug)]
pub struct LookupRequest<'a> {
    /// The code hash of the contract bytecode.
    pub code_hash: B256,
    /// The raw contract bytecode.
    pub code: &'a [u8],
    /// The EVM spec (hardfork) for this execution.
    pub spec_id: SpecId,
}

/// Result of a lookup.
#[derive(Clone, Debug)]
pub enum LookupDecision {
    /// A compiled function is available.
    Compiled(Arc<CompiledProgram>),
    /// The caller should interpret instead.
    Interpret(InterpretReason),
}

/// Reason why the runtime returned "interpret" instead of a compiled function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpretReason {
    /// The runtime is disabled.
    Disabled,
    /// No compiled program is ready for this key.
    NotReady,
    /// The JIT compile queue is saturated.
    QueueSaturated,
    /// JIT compilation failed for this contract.
    JitFailed,
    /// The requested backend is not available.
    UnsupportedBackend,
    /// The compiled artifact was invalidated.
    Invalidated,
}

/// A compiled EVM program kept alive in the resident map.
pub struct CompiledProgram {
    /// The cache key this program was compiled for.
    pub key: RuntimeCacheKey,
    /// Whether this is an AOT or JIT program.
    pub kind: ProgramKind,
    /// The callable compiled function.
    pub func: EvmCompilerFn,
    /// Approximate size of the compiled code in bytes.
    pub approx_size_bytes: usize,
    /// Keeps the backing memory (shared library / JIT module) alive.
    _backing: ProgramBacking,
}

impl std::fmt::Debug for CompiledProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledProgram")
            .field("key", &self.key)
            .field("kind", &self.kind)
            .field("func", &self.func)
            .field("approx_size_bytes", &self.approx_size_bytes)
            .finish_non_exhaustive()
    }
}

impl CompiledProgram {
    /// Creates a new compiled program backed by a loaded shared library.
    pub(crate) fn new_aot(
        key: RuntimeCacheKey,
        func: EvmCompilerFn,
        approx_size_bytes: usize,
        owner: Arc<LoadedLibraryOwner>,
    ) -> Self {
        Self {
            key,
            kind: ProgramKind::Aot,
            func,
            approx_size_bytes,
            _backing: ProgramBacking::LoadedLibrary(owner),
        }
    }
}

/// Whether this program was compiled AOT or JIT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramKind {
    /// Loaded from a pre-compiled artifact at startup.
    Aot,
    /// Compiled at runtime by the JIT.
    Jit,
}

/// Backing storage that keeps compiled code alive.
#[expect(dead_code, reason = "variant fields are held alive for Drop")]
enum ProgramBacking {
    /// A dynamically loaded shared library.
    LoadedLibrary(Arc<LoadedLibraryOwner>),
}

/// Owns a loaded shared library and its backing temp directory.
///
/// Dropping this unloads the library and cleans up the temp files.
pub(crate) struct LoadedLibraryOwner {
    /// The loaded library. Must be dropped before `_tmp_dir`.
    _library: libloading::Library,
    /// Temp directory holding the shared library file.
    _tmp_dir: tempfile::TempDir,
}

impl LoadedLibraryOwner {
    /// Creates a new owner from a loaded library and its temp directory.
    pub(crate) fn new(library: libloading::Library, tmp_dir: tempfile::TempDir) -> Self {
        Self { _library: library, _tmp_dir: tmp_dir }
    }
}

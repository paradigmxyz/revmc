//! Public types and handle methods.

use crate::{
    EvmCompilerFn,
    runtime::{storage::RuntimeCacheKey, worker::WorkerBacking},
};
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
        library: Arc<LoadedLibrary>,
    ) -> Self {
        Self {
            key,
            kind: ProgramKind::Aot,
            func,
            approx_size_bytes,
            _backing: ProgramBacking::LoadedLibrary(library),
        }
    }

    /// Creates a new compiled program backed by a JIT worker's compiler.
    pub(crate) fn new_jit(
        key: RuntimeCacheKey,
        func: EvmCompilerFn,
        approx_size_bytes: usize,
        backing: Arc<WorkerBacking>,
    ) -> Self {
        Self {
            key,
            kind: ProgramKind::Jit,
            func,
            approx_size_bytes,
            _backing: ProgramBacking::JitModule(backing),
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
    /// A dynamically loaded shared library (AOT).
    LoadedLibrary(Arc<LoadedLibrary>),
    /// A JIT worker's backing compiler (keeps JIT code alive).
    JitModule(Arc<WorkerBacking>),
}

/// Owns a loaded shared library.
///
/// Dropping this unloads the library.
pub(crate) struct LoadedLibrary {
    /// The loaded library.
    _library: libloading::Library,
}

impl LoadedLibrary {
    /// Creates a new loaded library wrapper.
    pub(crate) fn new(library: libloading::Library) -> Self {
        Self { _library: library }
    }
}

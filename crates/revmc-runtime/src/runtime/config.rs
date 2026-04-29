//! Runtime configuration.

use crate::{CompileTimings, runtime::storage::ArtifactStore};
use alloy_primitives::B256;
use revm_context_interface::cfg::GasParams;
use revm_primitives::hardfork::SpecId;
use std::{path::PathBuf, sync::Arc, time::Duration};

/// Runtime configuration.
#[derive(Clone, derive_more::Debug)]
pub struct RuntimeConfig {
    /// Whether compiled-code lookup is enabled.
    ///
    /// Defaults to `false` (safe rollout default).
    pub enabled: bool,

    /// Name for the backend thread.
    ///
    /// Defaults to `"revmc-backend"`.
    pub thread_name: String,

    /// Artifact store for loading precompiled AOT artifacts.
    ///
    /// `None` means no AOT preload—only JIT will populate the map (in later phases).
    #[debug(skip)]
    pub store: Option<Arc<dyn ArtifactStore>>,

    /// Tuning knobs.
    pub tuning: RuntimeTuning,

    /// Base directory for compiler debug dumps.
    ///
    /// When set, the compiler dumps IR, assembly, and bytecode for each compiled contract
    /// to `{dump_dir}/{spec_id}/{code_hash}/`.
    ///
    /// Defaults to `None` (no dumps).
    pub dump_dir: Option<PathBuf>,

    /// Enable debug assertions in compiled code.
    ///
    /// When `true`, the compiler inserts runtime checks (e.g. stack bounds)
    /// that `panic!` on violation. Useful for diagnosing JIT correctness bugs.
    ///
    /// Defaults to `false`.
    pub debug_assertions: bool,

    /// Disable the block deduplication pass.
    ///
    /// When `true`, the dedup pass that merges identical basic blocks is skipped.
    /// Useful for isolating dedup-related JIT correctness bugs.
    ///
    /// Defaults to `false`.
    pub no_dedup: bool,

    /// Disable the dead store elimination pass.
    ///
    /// When `true`, DSE is skipped. Useful for debugging JIT correctness issues
    /// where DSE incorrectly eliminates live stack operations.
    ///
    /// Defaults to `false`.
    pub no_dse: bool,

    /// Custom gas parameters for compile-time gas folding.
    ///
    /// Overrides the default gas schedule derived from `spec_id` when compiling
    /// bytecode. Useful for custom chains with non-standard gas costs (e.g.
    /// modified SSTORE, CREATE, or EXP costs).
    ///
    /// When `None`, the compiler uses `GasParams::new_spec(spec_id)`.
    ///
    /// Defaults to `None`.
    pub gas_params: Option<GasParams>,

    /// AOT mode: observed misses are promoted to AOT compilation instead of JIT.
    ///
    /// Defaults to `false`.
    pub aot: bool,

    /// Where JIT compilation work runs.
    ///
    /// Defaults to [`JitProcessMode::InProcess`].
    pub jit_process_mode: JitProcessMode,

    /// Helper executable used when [`jit_process_mode`](Self::jit_process_mode)
    /// is [`JitProcessMode::OutOfProcess`].
    ///
    /// When `None`, the runtime spawns `std::env::current_exe()` and expects it
    /// to call [`super::maybe_run_jit_helper`] during startup.
    pub jit_helper_path: Option<PathBuf>,

    /// Blocking mode: every lookup synchronously JIT-compiles on miss and never
    /// falls back to the interpreter.
    ///
    /// When `true`, [`lookup()`](super::JitBackend::lookup) behaves like
    /// [`lookup_blocking()`](super::JitBackend::lookup_blocking): if the
    /// compiled function is not already resident, the calling thread blocks
    /// until JIT compilation completes. This implies `enabled = true` and
    /// `jit_hot_threshold = 0`.
    ///
    /// Intended for debugging and testing only — not for production use.
    ///
    /// Defaults to `false`.
    pub blocking: bool,

    /// Callback invoked after each compilation completes (success or failure).
    ///
    /// Defaults to `None`.
    #[debug(skip)]
    pub on_compilation: Option<Arc<dyn Fn(CompilationEvent) + Send + Sync>>,
}

/// Event emitted after a compilation attempt completes.
#[derive(Clone, Debug)]
pub struct CompilationEvent {
    /// The code hash of the compiled bytecode.
    pub code_hash: B256,
    /// The hardfork spec the bytecode was compiled for.
    pub spec_id: SpecId,
    /// Wall-clock time spent compiling.
    pub duration: Duration,
    /// Whether this was a JIT or AOT compilation.
    pub kind: CompilationKind,
    /// Whether compilation succeeded.
    pub success: bool,
    /// Per-phase timing breakdown (translate, optimize, codegen).
    pub timings: CompileTimings,
}

/// The kind of compilation that was performed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompilationKind {
    /// JIT compilation (in-memory function pointer).
    Jit,
    /// AOT compilation (shared library artifact).
    Aot,
}

/// Where JIT compilation work runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JitProcessMode {
    /// Compile on background threads in this process.
    #[default]
    InProcess,
    /// Compile in a helper process and link the result into this process.
    ///
    /// This is reserved for the out-of-process JIT implementation and is
    /// disabled by default.
    OutOfProcess,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            thread_name: "revmc-backend".into(),
            store: None,
            tuning: RuntimeTuning::default(),
            dump_dir: None,
            debug_assertions: false,
            no_dedup: false,
            no_dse: false,
            gas_params: None,
            aot: false,
            jit_process_mode: JitProcessMode::default(),
            jit_helper_path: None,
            blocking: false,
            on_compilation: None,
        }
    }
}

/// Tuning knobs for the runtime.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeTuning {
    /// Capacity of the channel between API callers and the backend.
    ///
    /// Defaults to `4096`.
    pub channel_capacity: usize,

    /// Maximum lookup events processed per backend wakeup.
    ///
    /// Defaults to `4096`.
    pub max_events_per_drain: usize,

    /// Maximum delay between lookup observation and hotness accounting.
    ///
    /// Defaults to `100ms`.
    pub event_drain_interval: Duration,

    /// Timeout for joining the backend thread during shutdown.
    ///
    /// Defaults to `5s`.
    pub shutdown_timeout: Duration,

    /// Number of observed misses before a key is promoted to JIT compilation.
    ///
    /// Defaults to `8`.
    pub jit_hot_threshold: usize,

    /// Maximum bytecode length eligible for compilation. `0` = no limit.
    ///
    /// Defaults to `0`.
    pub jit_max_bytecode_len: usize,

    /// Maximum number of JIT compilation jobs in flight.
    ///
    /// Defaults to `2048`.
    pub jit_max_pending_jobs: usize,

    /// Number of JIT compilation worker threads.
    ///
    /// Defaults to `min(max(1, cpus/2), 4)`.
    pub jit_worker_count: usize,

    /// Timeout for a single out-of-process JIT compilation job.
    ///
    /// When exceeded, the helper process is killed and a fresh helper is spawned for
    /// the next job. Only applies to [`JitProcessMode::OutOfProcess`].
    ///
    /// Defaults to `5s`.
    pub jit_helper_timeout: Duration,

    /// Capacity of the per-worker job queue.
    ///
    /// Defaults to `64`.
    pub jit_worker_queue_capacity: usize,

    /// Optimization level for JIT compilation.
    ///
    /// Defaults to [`OptimizationLevel::Default`](crate::OptimizationLevel::Default).
    pub jit_opt_level: crate::OptimizationLevel,

    /// Optimization level for AOT compilation.
    ///
    /// Defaults to [`OptimizationLevel::Default`](crate::OptimizationLevel::Default).
    pub aot_opt_level: crate::OptimizationLevel,

    /// Maximum total resident compiled code size in bytes. `0` = no limit.
    ///
    /// When exceeded, least-recently-used entries are evicted.
    ///
    /// Defaults to `0`.
    pub resident_code_cache_bytes: usize,

    /// Duration after which a resident program with no lookup hits is evicted.
    /// `None` disables idle eviction.
    ///
    /// Defaults to `None`.
    pub idle_evict_duration: Option<Duration>,

    /// How often the backend runs eviction sweeps, if `idle_evict_duration` is set.
    ///
    /// Defaults to `60s`.
    pub eviction_sweep_interval: Duration,

    /// Number of compilations before recycling the compiler to reclaim
    /// accumulated memory. `0` = never recycle.
    ///
    /// Defaults to `1000`.
    pub compiler_recycle_threshold: usize,
}

impl RuntimeTuning {
    /// Returns whether `bytecode` is eligible for JIT/AOT compilation.
    #[inline]
    pub fn should_compile(&self, bytecode: &[u8]) -> bool {
        if bytecode.is_empty() {
            return false;
        }
        if self.jit_max_bytecode_len > 0 && bytecode.len() > self.jit_max_bytecode_len {
            return false;
        }
        true
    }
}

impl Default for RuntimeTuning {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let worker_count = cpus.div_ceil(2).clamp(1, 4);

        Self {
            channel_capacity: 4096,
            max_events_per_drain: 4096,
            event_drain_interval: Duration::from_millis(100),
            shutdown_timeout: Duration::from_secs(5),
            jit_hot_threshold: 8,
            jit_max_bytecode_len: 0,
            jit_max_pending_jobs: 2048,
            jit_worker_count: worker_count,
            jit_helper_timeout: Duration::from_secs(5),
            jit_worker_queue_capacity: 64,
            jit_opt_level: crate::OptimizationLevel::default(),
            aot_opt_level: crate::OptimizationLevel::default(),
            resident_code_cache_bytes: 1024 * 1024 * 1024,
            idle_evict_duration: Some(Duration::from_secs(600)),
            eviction_sweep_interval: Duration::from_secs(60),
            compiler_recycle_threshold: 1000,
        }
    }
}

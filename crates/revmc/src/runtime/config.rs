//! Runtime configuration.

use crate::runtime::storage::ArtifactStore;
use std::{path::PathBuf, sync::Arc, time::Duration};

/// Runtime configuration.
#[allow(missing_debug_implementations)]
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
            blocking: false,
        }
    }
}

/// Tuning knobs for the runtime.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeTuning {
    /// Capacity of the bounded channel for lookup-observed events.
    ///
    /// When the channel is full, events are silently dropped.
    ///
    /// Defaults to `4096`.
    pub lookup_event_channel_capacity: usize,

    /// Timeout for joining the backend thread during shutdown.
    ///
    /// Defaults to `5s`.
    pub shutdown_timeout: Duration,

    /// Number of observed misses before a key is promoted to JIT compilation.
    ///
    /// Defaults to `8`.
    pub jit_hot_threshold: u32,

    /// Maximum bytecode length eligible for JIT compilation.
    ///
    /// Contracts with bytecode larger than this are never promoted to JIT.
    /// Large contracts have diminishing JIT returns and long compile times.
    ///
    /// `0` means no limit.
    ///
    /// Defaults to `0` (no limit).
    pub jit_max_bytecode_len: usize,

    /// Maximum number of JIT compilation jobs in flight.
    ///
    /// Defaults to `2048`.
    pub max_pending_jit_jobs: usize,

    /// Number of JIT compilation worker threads.
    ///
    /// Defaults to `min(max(1, cpus/2), 4)`.
    pub jit_worker_count: usize,

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
    /// Defaults to [`OptimizationLevel::Aggressive`](crate::OptimizationLevel::Aggressive).
    pub aot_opt_level: crate::OptimizationLevel,

    /// Maximum total resident compiled code size in bytes.
    ///
    /// When the total `approx_size_bytes` of all resident programs exceeds this limit,
    /// the backend evicts the least-recently-used entries until under budget.
    ///
    /// `0` means no limit.
    ///
    /// Defaults to `0` (no limit).
    pub resident_code_cache_bytes: usize,

    /// Duration after which a resident program with no lookup hits is evicted.
    ///
    /// When a compiled program has not been hit for this duration, the backend
    /// removes it from the resident map. This naturally cleans up stale entries
    /// after hardfork transitions (old `spec_id` contracts stop being looked up).
    ///
    /// `None` means idle eviction is disabled.
    ///
    /// Defaults to `None` (disabled).
    pub idle_evict_duration: Option<Duration>,

    /// How often the backend runs eviction sweeps.
    ///
    /// Defaults to `60s`.
    pub eviction_sweep_interval: Duration,
}

impl Default for RuntimeTuning {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let worker_count = cpus.div_ceil(2).clamp(1, 4);

        Self {
            lookup_event_channel_capacity: 4096,
            shutdown_timeout: Duration::from_secs(5),
            jit_hot_threshold: 8,
            jit_max_bytecode_len: 0,
            max_pending_jit_jobs: 2048,
            jit_worker_count: worker_count,
            jit_worker_queue_capacity: 64,
            jit_opt_level: crate::OptimizationLevel::Default,
            aot_opt_level: crate::OptimizationLevel::Aggressive,
            resident_code_cache_bytes: 0,
            idle_evict_duration: None,
            eviction_sweep_interval: Duration::from_secs(60),
        }
    }
}

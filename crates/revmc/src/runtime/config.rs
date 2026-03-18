//! Runtime configuration.

use crate::runtime::storage::ArtifactStore;
use std::{sync::Arc, time::Duration};

/// Runtime configuration.
#[allow(missing_debug_implementations)]
pub struct RuntimeConfig {
    /// Whether compiled-code lookup is enabled.
    ///
    /// Defaults to `false` (safe rollout default).
    pub enabled: bool,

    /// Name for the coordinator thread.
    ///
    /// Defaults to `"revmc-coordinator"`.
    pub thread_name: String,

    /// Artifact store for loading precompiled AOT artifacts.
    ///
    /// `None` means no AOT preload—only JIT will populate the map (in later phases).
    pub store: Option<Arc<dyn ArtifactStore>>,

    /// Tuning knobs.
    pub tuning: RuntimeTuning,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            thread_name: "revmc-coordinator".into(),
            store: None,
            tuning: RuntimeTuning::default(),
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

    /// Timeout for joining the coordinator thread during shutdown.
    ///
    /// Defaults to `5s`.
    pub shutdown_timeout: Duration,

    /// Number of observed misses before a key is promoted to JIT compilation.
    ///
    /// Defaults to `8`.
    pub jit_hot_threshold: u32,

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
}

impl Default for RuntimeTuning {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let worker_count = cpus.div_ceil(2).clamp(1, 4);

        Self {
            lookup_event_channel_capacity: 4096,
            shutdown_timeout: Duration::from_secs(5),
            jit_hot_threshold: 8,
            max_pending_jit_jobs: 2048,
            jit_worker_count: worker_count,
            jit_worker_queue_capacity: 64,
            jit_opt_level: crate::OptimizationLevel::Default,
            aot_opt_level: crate::OptimizationLevel::Aggressive,
        }
    }
}

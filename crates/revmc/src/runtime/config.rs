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
    /// If the thread does not exit within this duration, shutdown returns an error.
    ///
    /// Defaults to `5s`.
    pub shutdown_timeout: Duration,
}

impl Default for RuntimeTuning {
    fn default() -> Self {
        Self { lookup_event_channel_capacity: 4096, shutdown_timeout: Duration::from_secs(5) }
    }
}

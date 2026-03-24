//! Runtime statistics.

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic counters for runtime observability.
#[derive(Debug, Default)]
pub(crate) struct RuntimeStats {
    /// Total lookups that returned a compiled function.
    pub(crate) lookup_hits: AtomicU64,
    /// Total lookups that returned interpret (not ready).
    pub(crate) lookup_misses: AtomicU64,
    /// Lookup-observed events successfully enqueued.
    pub(crate) events_sent: AtomicU64,
    /// Lookup-observed events dropped (channel full).
    pub(crate) events_dropped: AtomicU64,
}

/// A point-in-time snapshot of runtime stats.
#[derive(Clone, Copy, Debug, Default)]
pub struct RuntimeStatsSnapshot {
    /// Total lookups that returned a compiled function.
    pub lookup_hits: u64,
    /// Total lookups that returned interpret (not ready).
    pub lookup_misses: u64,
    /// Lookup-observed events successfully enqueued.
    pub events_sent: u64,
    /// Lookup-observed events dropped (channel full).
    pub events_dropped: u64,
    /// Number of entries in the resident compiled map.
    pub resident_entries: u64,
    /// Approximate total bytes of compiled code in the resident map.
    pub resident_bytes: u64,
    /// Number of commands pending in the backend command queue.
    pub jit_queue_len: u64,
}

impl RuntimeStats {
    pub(crate) fn snapshot(
        &self,
        resident_entries: u64,
        resident_bytes: u64,
        jit_queue_len: u64,
    ) -> RuntimeStatsSnapshot {
        RuntimeStatsSnapshot {
            lookup_hits: self.lookup_hits.load(Ordering::Relaxed),
            lookup_misses: self.lookup_misses.load(Ordering::Relaxed),
            events_sent: self.events_sent.load(Ordering::Relaxed),
            events_dropped: self.events_dropped.load(Ordering::Relaxed),
            resident_entries,
            resident_bytes,
            jit_queue_len,
        }
    }
}

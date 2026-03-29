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
    /// Total number of entries evicted (idle + budget).
    pub(crate) evictions: AtomicU64,
    /// Total number of compilations dispatched (JIT promotions + AOT requests).
    pub(crate) compilations_dispatched: AtomicU64,
    /// Total number of successful compilations (JIT + AOT).
    pub(crate) compilations_succeeded: AtomicU64,
    /// Total number of failed compilations (JIT + AOT).
    pub(crate) compilations_failed: AtomicU64,
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
    /// Number of commands pending in the backend command queue.
    pub jit_queue_len: u64,
    /// Bytes allocated for executable JIT code sections.
    ///
    /// Sourced from the LLVM JIT memory usage plugin. Reflects live memory:
    /// bytes are added on compilation and subtracted when JIT code is freed.
    /// `0` if the LLVM backend is not initialized.
    pub jit_code_bytes: u64,
    /// Bytes allocated for non-executable JIT data sections.
    ///
    /// Sourced from the LLVM JIT memory usage plugin. `0` if not initialized.
    pub jit_data_bytes: u64,
    /// Total number of entries evicted (idle + budget).
    pub evictions: u64,
    /// Total number of compilations dispatched (JIT promotions + AOT requests).
    pub compilations_dispatched: u64,
    /// Total number of successful compilations (JIT + AOT).
    pub compilations_succeeded: u64,
    /// Total number of failed compilations (JIT + AOT).
    pub compilations_failed: u64,
}

impl RuntimeStatsSnapshot {
    /// Total bytes allocated by the JIT engine (code + data).
    pub fn jit_total_bytes(&self) -> u64 {
        self.jit_code_bytes + self.jit_data_bytes
    }
}

impl RuntimeStats {
    pub(crate) fn snapshot(
        &self,
        resident_entries: u64,
        jit_queue_len: u64,
    ) -> RuntimeStatsSnapshot {
        #[cfg(feature = "llvm")]
        let (jit_code_bytes, jit_data_bytes) = crate::llvm::jit_memory_usage()
            .map(|u| (u.code_bytes as u64, u.data_bytes as u64))
            .unwrap_or((0, 0));
        #[cfg(not(feature = "llvm"))]
        let (jit_code_bytes, jit_data_bytes) = (0, 0);

        RuntimeStatsSnapshot {
            lookup_hits: self.lookup_hits.load(Ordering::Relaxed),
            lookup_misses: self.lookup_misses.load(Ordering::Relaxed),
            events_sent: self.events_sent.load(Ordering::Relaxed),
            events_dropped: self.events_dropped.load(Ordering::Relaxed),
            resident_entries,
            jit_queue_len,
            jit_code_bytes,
            jit_data_bytes,
            evictions: self.evictions.load(Ordering::Relaxed),
            compilations_dispatched: self.compilations_dispatched.load(Ordering::Relaxed),
            compilations_succeeded: self.compilations_succeeded.load(Ordering::Relaxed),
            compilations_failed: self.compilations_failed.load(Ordering::Relaxed),
        }
    }
}

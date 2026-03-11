//! Coordinator thread: single-threaded event loop for runtime state management.
//!
//! The coordinator is the sole writer of mutable state. It processes lookup-observed
//! events, tracks hotness, admits JIT compilation jobs, and inserts results into
//! the shared resident `DashMap`.

use crate::runtime::{
    api::CompiledProgram,
    config::RuntimeTuning,
    storage::RuntimeCacheKey,
    worker::{JitJob, WorkerPool, WorkerResult},
};
use dashmap::DashMap;
use rustc_hash::FxHashMap;
use std::sync::{Arc, mpsc};

/// The resident map type: code_hash+spec_id → compiled program.
pub(crate) type ResidentMap = DashMap<RuntimeCacheKey, Arc<CompiledProgram>>;

/// Commands sent to the coordinator thread.
pub(crate) enum Command {
    /// A lookup was observed on the hot path.
    LookupObserved(LookupObservedEvent),
    /// Shut down the coordinator.
    Shutdown,
}

/// A lookup-observed event.
pub(crate) struct LookupObservedEvent {
    /// The key that was looked up.
    pub(crate) key: RuntimeCacheKey,
    /// Whether the lookup was a hit (compiled found).
    pub(crate) was_hit: bool,
    /// The bytecode, present only on misses.
    pub(crate) bytecode: Option<Arc<[u8]>>,
}

/// Per-key state tracked by the coordinator.
struct EntryState {
    /// Number of observed misses.
    hotness: u32,
    /// Current phase.
    phase: EntryPhase,
    /// The bytecode for this key (captured from a miss event).
    bytecode: Arc<[u8]>,
}

/// Phase of a coordinator entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryPhase {
    /// Not yet hot enough for JIT.
    Cold,
    /// JIT compilation in progress on a worker.
    Working,
    /// JIT compilation failed.
    Failed,
}

/// All coordinator-owned mutable state.
struct CoordinatorState {
    /// The shared resident map (handles read, coordinator writes).
    resident: Arc<ResidentMap>,
    /// Per-key tracking state (coordinator-only).
    entries: FxHashMap<RuntimeCacheKey, EntryState>,
    /// Worker pool for JIT compilation.
    workers: WorkerPool,
    /// Receiver for worker results.
    result_rx: mpsc::Receiver<WorkerResult>,
    /// Tuning knobs.
    tuning: RuntimeTuning,
    /// Number of keys currently in Working phase.
    pending_jobs: usize,
    /// JIT stats.
    jit_promotions: u64,
    jit_successes: u64,
    jit_failures: u64,
}

impl CoordinatorState {
    fn handle_lookup_observed(&mut self, event: LookupObservedEvent) {
        // Hits don't need any action.
        if event.was_hit {
            return;
        }

        // Already in the resident map (may have been inserted since the event was emitted).
        if self.resident.contains_key(&event.key) {
            return;
        }

        let bytecode = match event.bytecode {
            Some(b) => b,
            None => return,
        };

        // Skip empty bytecodes.
        if bytecode.is_empty() {
            return;
        }

        let entry = self.entries.entry(event.key.clone()).or_insert_with(|| EntryState {
            hotness: 0,
            phase: EntryPhase::Cold,
            bytecode: Arc::clone(&bytecode),
        });

        // Only increment hotness for cold entries.
        if entry.phase != EntryPhase::Cold {
            return;
        }

        entry.hotness = entry.hotness.saturating_add(1);

        if entry.hotness >= self.tuning.jit_hot_threshold
            && self.pending_jobs < self.tuning.max_pending_jit_jobs
        {
            let symbol = format!("jit_{:x}_{:?}", event.key.code_hash, event.key.spec_id);
            let job = JitJob {
                key: event.key,
                bytecode: Arc::clone(&entry.bytecode),
                symbol_name: symbol,
            };

            if self.workers.try_send(job) {
                entry.phase = EntryPhase::Working;
                self.pending_jobs += 1;
                self.jit_promotions += 1;
            }
        }
    }

    fn handle_worker_result(&mut self, result: WorkerResult) {
        self.pending_jobs = self.pending_jobs.saturating_sub(1);

        match result.outcome {
            Ok(success) => {
                let backing = self.workers.backing(result.worker_id);
                let program = Arc::new(CompiledProgram::new_jit(
                    result.key.clone(),
                    success.func,
                    success.approx_size_bytes,
                    backing,
                ));

                self.resident.insert(result.key.clone(), program);
                self.entries.remove(&result.key);
                self.jit_successes += 1;

                debug!(
                    code_hash = %result.key.code_hash,
                    spec_id = ?result.key.spec_id,
                    "JIT program published to resident map",
                );
            }
            Err(err) => {
                if let Some(entry) = self.entries.get_mut(&result.key) {
                    entry.phase = EntryPhase::Failed;
                }
                self.jit_failures += 1;

                warn!(
                    code_hash = %result.key.code_hash,
                    error = %err,
                    "JIT compilation failed",
                );
            }
        }
    }

    /// Drains all pending worker results (non-blocking).
    fn drain_worker_results(&mut self) {
        while let Ok(result) = self.result_rx.try_recv() {
            self.handle_worker_result(result);
        }
    }
}

/// Runs the coordinator event loop. Called on the coordinator thread.
pub(crate) fn run(rx: mpsc::Receiver<Command>, resident: Arc<ResidentMap>, tuning: RuntimeTuning) {
    debug!("coordinator thread started");

    let (result_tx, result_rx) = mpsc::channel::<WorkerResult>();

    let workers = WorkerPool::new(
        tuning.jit_worker_count,
        tuning.jit_worker_queue_capacity,
        result_tx,
        tuning.jit_opt_level,
    );

    let mut state = CoordinatorState {
        resident,
        entries: FxHashMap::default(),
        workers,
        result_rx,
        tuning,
        pending_jobs: 0,
        jit_promotions: 0,
        jit_successes: 0,
        jit_failures: 0,
    };

    loop {
        // Drain pending worker results first (non-blocking).
        state.drain_worker_results();

        // Block on the command channel.
        match rx.recv() {
            Ok(Command::LookupObserved(event)) => {
                state.handle_lookup_observed(event);
            }
            Ok(Command::Shutdown) => {
                debug!("coordinator shutting down");
                break;
            }
            Err(_) => {
                debug!("coordinator channel closed, shutting down");
                break;
            }
        }
    }

    // Drain remaining worker results before shutdown.
    state.drain_worker_results();

    info!(
        jit_promotions = state.jit_promotions,
        jit_successes = state.jit_successes,
        jit_failures = state.jit_failures,
        resident_entries = state.resident.len(),
        "coordinator stats at shutdown",
    );

    state.workers.shutdown();
}

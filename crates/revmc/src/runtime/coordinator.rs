//! Coordinator thread: single-threaded event loop for runtime state management.

use crate::runtime::storage::RuntimeCacheKey;
use std::sync::mpsc;

/// Commands sent to the coordinator thread.
pub(crate) enum Command {
    /// A lookup was observed on the hot path.
    LookupObserved(LookupObservedEvent),
    /// Shut down the coordinator.
    Shutdown,
}

/// A lookup-observed event.
#[derive(Debug)]
pub(crate) struct LookupObservedEvent {
    /// The key that was looked up.
    pub(crate) key: RuntimeCacheKey,
    /// Whether the lookup was a hit (compiled found).
    pub(crate) was_hit: bool,
}

/// Runs the coordinator event loop. Called on the coordinator thread.
///
/// In Phase 0, this only logs events. Future phases will add hotness tracking and JIT admission.
pub(crate) fn run(rx: mpsc::Receiver<Command>) {
    debug!("coordinator thread started");
    loop {
        match rx.recv() {
            Ok(Command::LookupObserved(event)) => {
                trace!(
                    code_hash = %event.key.code_hash,
                    spec_id = ?event.key.spec_id,
                    was_hit = event.was_hit,
                    "lookup observed",
                );
            }
            Ok(Command::Shutdown) => {
                debug!("coordinator shutting down");
                break;
            }
            Err(_) => {
                // All senders dropped — handle is gone.
                debug!("coordinator channel closed, shutting down");
                break;
            }
        }
    }
}

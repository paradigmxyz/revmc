use alloy_primitives::B256;
use revmc::primitives::{Bytes, SpecId};
use std::sync::{Arc, RwLock};
use tokio::{sync::Semaphore, task::JoinHandle};

use super::{
    runtime::{get_runtime, JitConfig, JitRuntime},
    sleddb::SledDB,
};

fn ivec_to_u64(ivec: &sled::IVec) -> Option<u64> {
    ivec.as_ref().try_into().ok().map(u64::from_be_bytes)
}

/// A worker responsible for compiling bytecode in machine code.
#[derive(Debug)]
pub struct CompileWorker {
    pub threshold: u64,
    sled_db: Arc<RwLock<SledDB<B256>>>,
    jit_runtime: Arc<JitRuntime>,
    semaphore: Arc<Semaphore>,
}

impl CompileWorker {
    /// Creates a new `CompileWorker`.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold for the number of times a bytecode must be seen before it is
    ///   compiled.
    /// * `sled_db` - A reference-counted, thread-safe handle to the sled database.
    /// * `max_concurrent_tasks` - The maximum number of concurrent jit tasks allowed.
    pub(crate) fn new(
        threshold: u64,
        sled_db: Arc<RwLock<SledDB<B256>>>,
        max_concurrent_tasks: usize,
    ) -> Self {
        Self {
            threshold,
            sled_db,
            jit_runtime: Arc::new(JitRuntime::new(JitConfig::default())),
            semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
        }
    }

    /// Processes the given bytecode with the specified specId.
    ///
    /// # Arguments
    ///
    /// * `spec_id` - The specification ID for the EVM.
    /// * `code_hash` - The hash of the bytecode to be compiled.
    /// * `bytecode` - The bytecode to be compiled.
    pub(crate) fn work(&self, spec_id: SpecId, code_hash: B256, bytecode: Bytes) -> JoinHandle<()> {
        // Read the current count of the bytecode hash from the embedded database
        let count = {
            let db_read = match self.sled_db.read() {
                Ok(lock) => lock,
                Err(poisoned) => poisoned.into_inner(),
            };
            let count_bytes = db_read.get(code_hash).unwrap_or(None);
            count_bytes.and_then(|v| ivec_to_u64(&v)).unwrap_or(0)
        };
        let new_count = count + 1;

        let sled_db = Arc::clone(&self.sled_db);
        let rt = self.jit_runtime.clone();
        let threshold = self.threshold;
        let semaphore = Arc::clone(&self.semaphore);

        let runtime = get_runtime();

        runtime.spawn(async move {
            let _permit = semaphore.acquire().await.unwrap();
            // Check if the bytecode is all zeros
            if code_hash.is_zero() {
                return;
            }
            // Check if the bytecode should be compiled
            if new_count == threshold {
                // Compile the bytecode
                match rt.compile(code_hash, bytecode, spec_id) {
                    Ok(_) => {
                        tracing::info!("Compiled: bytecode hash {code_hash}");
                    }
                    Err(err) => {
                        tracing::error!(
                            "Failed to Compile: with bytecode hash {code_hash} {err:#?}"
                        );
                        return;
                    }
                }
            }
            // Commit the new count to the database
            {
                let db_write = match sled_db.write() {
                    Ok(lock) => lock,
                    Err(poisoned) => poisoned.into_inner(),
                };
                db_write.put(code_hash, &new_count.to_be_bytes()).unwrap();
            }
        })
    }
}

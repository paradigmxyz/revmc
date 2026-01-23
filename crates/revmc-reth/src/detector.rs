//! Hot contract detection based on cumulative gas usage.

use alloy_primitives::B256;
use parking_lot::RwLock;
use revm_primitives::hardfork::SpecId;
use rustc_hash::FxHashMap;

/// Tracks gas consumption per bytecode hash to detect hot contracts.
#[derive(Debug)]
pub struct HotDetector {
    /// Cumulative gas used per bytecode hash.
    gas_usage: RwLock<FxHashMap<B256, GasStats>>,
    /// Gas threshold to consider a contract "hot".
    threshold: u64,
}

/// Statistics for a single bytecode.
#[derive(Debug, Clone, Default)]
struct GasStats {
    /// Total gas consumed across all executions.
    total_gas: u64,
    /// Number of times this bytecode was executed.
    execution_count: u64,
    /// Whether this has been queued for compilation.
    queued: bool,
}

/// Information about a bytecode that became hot.
#[derive(Debug, Clone)]
pub struct HotBytecode {
    /// The bytecode hash.
    pub hash: B256,
    /// The raw bytecode bytes.
    pub bytecode: Vec<u8>,
    /// The spec ID to compile for.
    pub spec_id: SpecId,
    /// Total gas consumed.
    pub total_gas: u64,
    /// Execution count.
    pub execution_count: u64,
}

impl HotDetector {
    /// Create a new hot detector with the given threshold.
    pub fn new(threshold: u64) -> Self {
        Self { gas_usage: RwLock::new(FxHashMap::default()), threshold }
    }

    /// Record an execution of a bytecode.
    ///
    /// Returns `Some(HotBytecode)` if this bytecode just crossed the hot threshold
    /// and hasn't been queued for compilation yet.
    pub fn record_execution(
        &self,
        hash: B256,
        bytecode: &[u8],
        gas_used: u64,
        spec_id: SpecId,
    ) -> Option<HotBytecode> {
        let mut usage = self.gas_usage.write();
        let stats = usage.entry(hash).or_default();

        stats.total_gas = stats.total_gas.saturating_add(gas_used);
        stats.execution_count += 1;

        // Check if we just crossed the threshold and haven't queued yet
        if stats.total_gas >= self.threshold && !stats.queued {
            stats.queued = true;
            Some(HotBytecode {
                hash,
                bytecode: bytecode.to_vec(),
                spec_id,
                total_gas: stats.total_gas,
                execution_count: stats.execution_count,
            })
        } else {
            None
        }
    }

    /// Check if a bytecode hash is known to be hot.
    pub fn is_hot(&self, hash: &B256) -> bool {
        self.gas_usage.read().get(hash).map(|s| s.total_gas >= self.threshold).unwrap_or(false)
    }

    /// Get the current gas usage for a bytecode.
    pub fn get_gas_usage(&self, hash: &B256) -> Option<u64> {
        self.gas_usage.read().get(hash).map(|s| s.total_gas)
    }

    /// Get statistics about the detector.
    pub fn stats(&self) -> DetectorStats {
        let usage = self.gas_usage.read();
        let total_contracts = usage.len();
        let hot_contracts = usage.values().filter(|s| s.total_gas >= self.threshold).count();
        let queued_contracts = usage.values().filter(|s| s.queued).count();

        DetectorStats {
            total_contracts,
            hot_contracts,
            queued_contracts,
            threshold: self.threshold,
        }
    }

    /// Clear all tracking data.
    pub fn clear(&self) {
        self.gas_usage.write().clear();
    }
}

/// Statistics about the hot detector.
#[derive(Debug, Clone)]
pub struct DetectorStats {
    /// Total number of unique contracts seen.
    pub total_contracts: usize,
    /// Number of contracts that have crossed the hot threshold.
    pub hot_contracts: usize,
    /// Number of contracts queued for compilation.
    pub queued_contracts: usize,
    /// The gas threshold.
    pub threshold: u64,
}

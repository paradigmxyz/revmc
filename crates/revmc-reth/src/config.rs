//! Configuration for the revmc runtime.

use revm_primitives::hardfork::SpecId;

/// Configuration for the revmc JIT compilation runtime.
#[derive(Debug, Clone)]
pub struct RevmcConfig {
    /// Enable JIT compilation. When false, all calls fall back to interpreter.
    pub enabled: bool,

    /// Gas threshold before a contract is considered "hot" and queued for compilation.
    /// Default: 1_000_000 gas (roughly 1M gas consumed across all executions)
    pub hot_threshold: u64,

    /// Maximum number of compiled functions to keep in the LRU cache.
    /// Default: 1024 contracts
    pub cache_size: usize,

    /// Number of background compilation workers.
    /// Default: 2 threads
    pub worker_threads: usize,

    /// Maximum bytecode size to compile. Larger contracts are skipped.
    /// Default: 24KB (EIP-170 limit)
    pub max_bytecode_size: usize,

    /// Optimization level for LLVM.
    /// 0 = None, 1 = Less, 2 = Default, 3 = Aggressive
    pub optimization_level: u8,

    /// Default spec ID for compilation when not specified.
    pub default_spec_id: SpecId,
}

impl Default for RevmcConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hot_threshold: 1_000_000,
            cache_size: 1024,
            worker_threads: 2,
            max_bytecode_size: 24 * 1024,
            optimization_level: 3,
            default_spec_id: SpecId::CANCUN,
        }
    }
}

impl RevmcConfig {
    /// Create a new config with JIT disabled.
    pub fn disabled() -> Self {
        Self { enabled: false, ..Default::default() }
    }

    /// Create a config optimized for low memory usage.
    pub fn low_memory() -> Self {
        Self { cache_size: 256, worker_threads: 1, optimization_level: 1, ..Default::default() }
    }

    /// Create a config optimized for maximum performance.
    pub fn high_performance() -> Self {
        Self {
            cache_size: 4096,
            worker_threads: 4,
            hot_threshold: 500_000,
            optimization_level: 3,
            ..Default::default()
        }
    }

    /// Set enabled state.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set the hot threshold.
    pub fn with_hot_threshold(mut self, threshold: u64) -> Self {
        self.hot_threshold = threshold;
        self
    }

    /// Set the cache size.
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }
}

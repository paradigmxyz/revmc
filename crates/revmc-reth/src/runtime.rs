//! Main runtime that ties everything together.

use crate::{
    cache::CompiledCache, config::RevmcConfig, detector::HotDetector, worker::CompilationWorker,
};
use alloy_primitives::B256;
use revm_interpreter::{Interpreter, InterpreterAction, SharedMemory};
use revm_primitives::hardfork::SpecId;
use revmc_context::{EvmCompilerFn, HostExt};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use tracing::{debug, trace};

/// The main revmc runtime for hot contract JIT compilation.
///
/// This struct is designed to be shared across threads (e.g., via `Arc<RevmcRuntime>`).
/// It handles:
/// - Hot contract detection based on gas usage
/// - Background JIT compilation
/// - Caching of compiled functions
/// - Statistics tracking
#[derive(Debug)]
pub struct RevmcRuntime {
    /// Configuration.
    config: RevmcConfig,
    /// Hot contract detector.
    detector: HotDetector,
    /// Compiled function cache.
    cache: Arc<CompiledCache>,
    /// Background compilation worker.
    worker: Option<CompilationWorker>,
    /// Whether the runtime is enabled.
    enabled: AtomicBool,
    /// Statistics: total executions.
    total_executions: AtomicU64,
    /// Statistics: JIT hits (used compiled code).
    jit_hits: AtomicU64,
    /// Statistics: interpreter fallbacks.
    interpreter_fallbacks: AtomicU64,
}

impl RevmcRuntime {
    /// Create a new runtime with the given configuration.
    pub fn new(config: RevmcConfig) -> Self {
        let cache = Arc::new(CompiledCache::new(config.cache_size));
        let detector = HotDetector::new(config.hot_threshold);

        let worker = if config.enabled {
            Some(CompilationWorker::new(&config, Arc::clone(&cache)))
        } else {
            None
        };

        Self {
            enabled: AtomicBool::new(config.enabled),
            config,
            detector,
            cache,
            worker,
            total_executions: AtomicU64::new(0),
            jit_hits: AtomicU64::new(0),
            interpreter_fallbacks: AtomicU64::new(0),
        }
    }

    /// Create a disabled runtime that always falls back to the interpreter.
    pub fn disabled() -> Self {
        Self::new(RevmcConfig::disabled())
    }

    /// Check if the runtime is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Enable or disable the runtime.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get a compiled function for the given bytecode hash.
    ///
    /// Returns `None` if:
    /// - The runtime is disabled
    /// - No compiled version exists
    /// - The bytecode hasn't been detected as hot yet
    #[inline]
    pub fn get_compiled(&self, bytecode_hash: B256) -> Option<EvmCompilerFn> {
        if !self.is_enabled() {
            return None;
        }

        // Process any pending compilation results first
        if let Some(ref worker) = self.worker {
            worker.process_completions();
        }

        self.cache.get(&bytecode_hash)
    }

    /// Record an execution of a bytecode.
    ///
    /// This tracks gas usage and queues the bytecode for compilation if it becomes hot.
    /// Should be called after each bytecode execution (whether JIT or interpreted).
    pub fn record_execution(
        &self,
        bytecode_hash: B256,
        bytecode: &[u8],
        gas_used: u64,
        spec_id: SpecId,
    ) {
        self.total_executions.fetch_add(1, Ordering::Relaxed);

        if !self.is_enabled() {
            return;
        }

        // Check if we already have this compiled
        if self.cache.contains(&bytecode_hash) {
            return;
        }

        // Record execution and check if it became hot
        if let Some(hot) =
            self.detector.record_execution(bytecode_hash, bytecode, gas_used, spec_id)
        {
            debug!(
                hash = %bytecode_hash,
                total_gas = hot.total_gas,
                executions = hot.execution_count,
                "Contract became hot, queueing for compilation"
            );

            if let Some(ref worker) = self.worker {
                worker.queue(hot);
            }
        }
    }

    /// Execute bytecode using compiled code if available, otherwise fall back to interpreter.
    ///
    /// This is the main entry point for integrating with revm handlers.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the interpreter and memory are valid and properly initialized.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn execute_or_fallback<F>(
        &self,
        bytecode_hash: B256,
        bytecode: &[u8],
        interpreter: &mut Interpreter,
        memory: &mut SharedMemory,
        host: &mut dyn HostExt,
        spec_id: SpecId,
        fallback: F,
    ) -> InterpreterAction
    where
        F: FnOnce(&mut Interpreter, &mut SharedMemory) -> InterpreterAction,
    {
        // Try to get compiled version
        if let Some(compiled) = self.get_compiled(bytecode_hash) {
            self.jit_hits.fetch_add(1, Ordering::Relaxed);
            trace!(hash = %bytecode_hash, "Using JIT compiled code");

            let result = compiled.call_with_interpreter_and_memory(interpreter, memory, host);

            // Record execution for statistics (gas used is in interpreter.gas)
            let gas_used = interpreter.gas.spent();
            self.record_execution(bytecode_hash, bytecode, gas_used, spec_id);

            result
        } else {
            self.interpreter_fallbacks.fetch_add(1, Ordering::Relaxed);
            trace!(hash = %bytecode_hash, "Falling back to interpreter");

            let result = fallback(interpreter, memory);

            // Record execution for hot detection
            let gas_used = interpreter.gas.spent();
            self.record_execution(bytecode_hash, bytecode, gas_used, spec_id);

            result
        }
    }

    /// Get runtime statistics.
    pub fn stats(&self) -> RuntimeStats {
        let cache_stats = self.cache.stats();
        let detector_stats = self.detector.stats();

        RuntimeStats {
            enabled: self.is_enabled(),
            total_executions: self.total_executions.load(Ordering::Relaxed),
            jit_hits: self.jit_hits.load(Ordering::Relaxed),
            interpreter_fallbacks: self.interpreter_fallbacks.load(Ordering::Relaxed),
            cached_functions: cache_stats.entries,
            cache_capacity: cache_stats.capacity,
            total_code_size: cache_stats.total_code_size,
            contracts_tracked: detector_stats.total_contracts,
            hot_contracts: detector_stats.hot_contracts,
            pending_compilations: self.worker.as_ref().map(|w| w.pending_count()).unwrap_or(0),
        }
    }

    /// Get the JIT hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.jit_hits.load(Ordering::Relaxed);
        let total = self.total_executions.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            (hits as f64 / total as f64) * 100.0
        }
    }

    /// Shutdown the runtime and all workers.
    pub fn shutdown(&self) {
        if let Some(ref worker) = self.worker {
            worker.shutdown();
        }
    }

    /// Get the underlying cache (for advanced use cases).
    pub fn cache(&self) -> &CompiledCache {
        &self.cache
    }

    /// Get the underlying detector (for advanced use cases).
    pub fn detector(&self) -> &HotDetector {
        &self.detector
    }

    /// Get the configuration.
    pub fn config(&self) -> &RevmcConfig {
        &self.config
    }
}

/// Runtime statistics.
#[derive(Debug, Clone)]
pub struct RuntimeStats {
    /// Whether JIT is enabled.
    pub enabled: bool,
    /// Total bytecode executions.
    pub total_executions: u64,
    /// Executions using JIT compiled code.
    pub jit_hits: u64,
    /// Executions falling back to interpreter.
    pub interpreter_fallbacks: u64,
    /// Number of cached compiled functions.
    pub cached_functions: usize,
    /// Cache capacity.
    pub cache_capacity: usize,
    /// Approximate total compiled code size.
    pub total_code_size: usize,
    /// Number of unique contracts tracked.
    pub contracts_tracked: usize,
    /// Number of hot contracts detected.
    pub hot_contracts: usize,
    /// Number of pending compilations.
    pub pending_compilations: usize,
}

impl std::fmt::Display for RuntimeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hit_rate = if self.total_executions > 0 {
            (self.jit_hits as f64 / self.total_executions as f64) * 100.0
        } else {
            0.0
        };

        write!(
            f,
            "RevmcRuntime: enabled={}, hit_rate={:.1}%, cached={}/{}, hot={}, pending={}",
            self.enabled,
            hit_rate,
            self.cached_functions,
            self.cache_capacity,
            self.hot_contracts,
            self.pending_compilations
        )
    }
}

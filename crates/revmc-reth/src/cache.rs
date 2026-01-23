//! LRU cache for compiled functions.

use alloy_primitives::B256;
use lru::LruCache;
use parking_lot::Mutex;
use revmc_context::EvmCompilerFn;
use std::num::NonZeroUsize;

/// Thread-safe LRU cache for compiled EVM functions.
#[derive(Debug)]
pub struct CompiledCache {
    cache: Mutex<LruCache<B256, CachedFunction>>,
}

/// A cached compiled function with metadata.
#[derive(Debug, Clone, Copy)]
struct CachedFunction {
    /// The compiled function pointer.
    function: EvmCompilerFn,
    /// Size of the compiled code in bytes (approximate).
    code_size: usize,
}

impl CompiledCache {
    /// Create a new cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let capacity = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self { cache: Mutex::new(LruCache::new(capacity)) }
    }

    /// Get a compiled function for the given bytecode hash.
    ///
    /// This also marks the entry as recently used.
    pub fn get(&self, hash: &B256) -> Option<EvmCompilerFn> {
        self.cache.lock().get(hash).map(|cf| cf.function)
    }

    /// Insert a compiled function into the cache.
    ///
    /// Returns the evicted entry if the cache was full.
    pub fn insert(&self, hash: B256, function: EvmCompilerFn, code_size: usize) -> Option<B256> {
        let mut cache = self.cache.lock();

        // Check if we need to evict
        let evicted =
            if cache.len() >= cache.cap().get() { cache.peek_lru().map(|(k, _)| *k) } else { None };

        cache.put(hash, CachedFunction { function, code_size });
        evicted
    }

    /// Check if a hash is in the cache without affecting LRU order.
    pub fn contains(&self, hash: &B256) -> bool {
        self.cache.lock().peek(hash).is_some()
    }

    /// Get the current number of cached functions.
    pub fn len(&self) -> usize {
        self.cache.lock().len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.lock().is_empty()
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.lock();
        let total_code_size: usize = cache.iter().map(|(_, cf)| cf.code_size).sum();

        CacheStats { entries: cache.len(), capacity: cache.cap().get(), total_code_size }
    }

    /// Clear the cache.
    ///
    /// # Safety
    /// The caller must ensure no compiled functions are currently executing.
    pub unsafe fn clear(&self) {
        self.cache.lock().clear();
    }
}

/// Statistics about the compiled cache.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of entries in the cache.
    pub entries: usize,
    /// Maximum capacity.
    pub capacity: usize,
    /// Approximate total size of compiled code.
    pub total_code_size: usize,
}

use std::{
    fmt::Debug,
    num::NonZeroUsize,
    sync::{Arc, RwLock, TryLockError},
};

use crate::{
    error::ExtError,
    worker::{store_path, CompileWorker, SledDB},
};
use alloy_primitives::B256;
use lru::LruCache;
use once_cell::sync::OnceCell;
use revm_primitives::{Bytes, SpecId};
use revmc::EvmCompilerFn;

pub(crate) static SLED_DB: OnceCell<Arc<RwLock<SledDB<B256>>>> = OnceCell::new();

#[derive(PartialEq, Debug)]
pub enum FetchedFnResult {
    Found(EvmCompilerFn),
    NotFound,
}
/// Compiler Worker as an external context.
///
/// External function fetching is optimized by using an LRU Cache.
/// In many cases, a contract that is called will likely be called again,
/// so the cache helps reduce library loading cost.
#[derive(Debug)]
pub struct EXTCompileWorker {
    compile_worker: CompileWorker,
    pub cache: RwLock<LruCache<B256, (EvmCompilerFn, libloading::Library)>>,
}

impl EXTCompileWorker {
    pub fn new(threshold: u64, max_concurrent_tasks: usize, cache_size_words: usize) -> Self {
        let sled_db = SLED_DB.get_or_init(|| Arc::new(RwLock::new(SledDB::init())));
        let compile_worker =
            CompileWorker::new(threshold, Arc::clone(sled_db), max_concurrent_tasks);

        Self {
            compile_worker,
            cache: RwLock::new(LruCache::new(NonZeroUsize::new(cache_size_words).unwrap())),
        }
    }

    /// Fetches the compiled function from disk, if exists
    pub fn get_function(&self, code_hash: B256) -> Result<FetchedFnResult, ExtError> {
        if code_hash.is_zero() {
            return Ok(FetchedFnResult::NotFound);
        }

        // Write locks are required for reading from LRU Cache
        {
            let mut acq = true;

            let cache = match self.cache.try_write() {
                Ok(c) => Some(c),
                Err(err) => match err {
                    /* in this case, read from file instead of cache */
                    TryLockError::WouldBlock => {
                        acq = false;
                        None
                    }
                    TryLockError::Poisoned(err) => Some(err.into_inner()),
                },
            };

            if acq {
                if let Some((f, _)) = cache.unwrap().get(&code_hash) {
                    return Ok(FetchedFnResult::Found(*f));
                }
            }
        }

        let so_file_path = store_path().join(code_hash.to_string()).join("a.so");
        if so_file_path.try_exists().unwrap_or(false) {
            {
                let lib = (unsafe { libloading::Library::new(&so_file_path) })
                    .map_err(|err| ExtError::LibLoadingError { err: err.to_string() })?;

                let f: EvmCompilerFn = unsafe {
                    *lib.get(code_hash.to_string().as_ref())
                        .map_err(|err| ExtError::GetSymbolError { err: err.to_string() })?
                };

                let mut cache = self
                    .cache
                    .write()
                    .map_err(|err| ExtError::RwLockPoison { err: err.to_string() })?;
                cache.put(code_hash, (f, lib));

                return Ok(FetchedFnResult::Found(f));
            }
        }

        Ok(FetchedFnResult::NotFound)
    }

    /// Starts compile routine JIT-ing the code referred by code_hash
    pub fn work(&self, spec_id: SpecId, code_hash: B256, bytecode: Bytes) -> Result<(), ExtError> {
        self.compile_worker.work(spec_id, code_hash, bytecode);

        Ok(())
    }

    /// Preloads cache upfront for the specified code hashes
    ///
    /// Improve runtime performance by reducing the overhead for library loading
    /// Available when the corresponding code hash is called definitively
    /// ex. AccesslistType Transaction (tx type: 2, 3)
    pub fn preload_cache(&self, code_hashes: Vec<B256>) -> Result<(), ExtError> {
        for code_hash in code_hashes.into_iter() {
            self.get_function(code_hash)?;
        }

        Ok(())
    }
}

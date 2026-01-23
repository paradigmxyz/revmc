//! Background compilation worker.

use crate::{cache::CompiledCache, detector::HotBytecode, RevmcConfig};
use alloy_primitives::B256;
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use revmc::llvm::{inkwell::context::Context, EvmLlvmBackend};
use revmc::{EvmCompiler, OptimizationLevel};
use revmc_context::EvmCompilerFn;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use tracing::{debug, error, info, warn};

/// Message sent to compilation workers.
#[derive(Debug)]
enum WorkerMessage {
    /// Compile a hot bytecode.
    Compile(HotBytecode),
    /// Shutdown the worker.
    Shutdown,
}

/// Result of a compilation.
#[derive(Debug)]
struct CompilationResult {
    /// The bytecode hash.
    hash: B256,
    /// The compiled function, or None if compilation failed.
    function: Option<EvmCompilerFn>,
    /// Approximate size of compiled code.
    code_size: usize,
}

/// Background compilation worker pool.
pub struct CompilationWorker {
    /// Channel to send work to workers.
    sender: Sender<WorkerMessage>,
    /// Channel to receive results.
    result_receiver: Receiver<CompilationResult>,
    /// Worker thread handles.
    handles: Mutex<Vec<JoinHandle<()>>>,
    /// Shared cache reference.
    cache: Arc<CompiledCache>,
    /// Number of pending compilations.
    pending: std::sync::atomic::AtomicUsize,
}

impl std::fmt::Debug for CompilationWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompilationWorker")
            .field("pending", &self.pending.load(std::sync::atomic::Ordering::Relaxed))
            .finish()
    }
}

impl CompilationWorker {
    /// Create a new worker pool.
    pub fn new(config: &RevmcConfig, cache: Arc<CompiledCache>) -> Self {
        let (sender, receiver) = bounded::<WorkerMessage>(256);
        let (result_sender, result_receiver) = bounded::<CompilationResult>(256);

        let mut handles = Vec::with_capacity(config.worker_threads);

        for worker_id in 0..config.worker_threads {
            let receiver = receiver.clone();
            let result_sender = result_sender.clone();
            let opt_level = config.optimization_level;
            let max_size = config.max_bytecode_size;

            let handle = thread::Builder::new()
                .name(format!("revmc-worker-{worker_id}"))
                .spawn(move || {
                    Self::worker_loop(worker_id, receiver, result_sender, opt_level, max_size);
                })
                .expect("failed to spawn worker thread");

            handles.push(handle);
        }

        info!(workers = config.worker_threads, "Started revmc compilation workers");

        Self {
            sender,
            result_receiver,
            handles: Mutex::new(handles),
            cache,
            pending: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Worker thread main loop.
    fn worker_loop(
        worker_id: usize,
        receiver: Receiver<WorkerMessage>,
        result_sender: Sender<CompilationResult>,
        opt_level: u8,
        max_size: usize,
    ) {
        debug!(worker_id, "Compilation worker started");

        // Create LLVM context for this thread (LLVM contexts are not thread-safe)
        let context = Context::create();

        while let Ok(msg) = receiver.recv() {
            match msg {
                WorkerMessage::Compile(hot) => {
                    let result = Self::compile_bytecode(&context, &hot, opt_level, max_size);
                    if result_sender.send(result).is_err() {
                        warn!(worker_id, "Result channel closed, worker exiting");
                        break;
                    }
                }
                WorkerMessage::Shutdown => {
                    debug!(worker_id, "Worker received shutdown signal");
                    break;
                }
            }
        }

        debug!(worker_id, "Compilation worker exiting");
    }

    /// Compile a single bytecode.
    fn compile_bytecode(
        context: &Context,
        hot: &HotBytecode,
        opt_level: u8,
        max_size: usize,
    ) -> CompilationResult {
        let start = std::time::Instant::now();
        let hash = hot.hash;

        // Skip if bytecode is too large
        if hot.bytecode.len() > max_size {
            debug!(
                hash = %hash,
                size = hot.bytecode.len(),
                max = max_size,
                "Skipping oversized bytecode"
            );
            return CompilationResult { hash, function: None, code_size: 0 };
        }

        // Skip empty bytecode
        if hot.bytecode.is_empty() {
            return CompilationResult { hash, function: None, code_size: 0 };
        }

        let opt = match opt_level {
            0 => OptimizationLevel::None,
            1 => OptimizationLevel::Less,
            2 => OptimizationLevel::Default,
            _ => OptimizationLevel::Aggressive,
        };

        let result = (|| -> Result<(EvmCompilerFn, usize), revmc::Error> {
            let backend = EvmLlvmBackend::new(context, false, opt)?;
            let mut compiler = EvmCompiler::new(backend);

            // Generate a unique name for this function
            let name = format!("jit_{hash:x}");

            // Compile the bytecode
            let function = unsafe { compiler.jit(&name, &hot.bytecode, hot.spec_id)? };

            // Estimate code size (rough approximation)
            let code_size = hot.bytecode.len() * 10; // JIT code is typically ~10x bytecode size

            Ok((function, code_size))
        })();

        let compile_time_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok((function, code_size)) => {
                info!(
                    hash = %hash,
                    bytecode_len = hot.bytecode.len(),
                    code_size,
                    compile_time_ms,
                    "Compiled hot contract"
                );
                CompilationResult { hash, function: Some(function), code_size }
            }
            Err(e) => {
                error!(
                    hash = %hash,
                    error = %e,
                    "Failed to compile bytecode"
                );
                CompilationResult { hash, function: None, code_size: 0 }
            }
        }
    }

    /// Queue a hot bytecode for compilation.
    pub fn queue(&self, hot: HotBytecode) -> bool {
        match self.sender.try_send(WorkerMessage::Compile(hot)) {
            Ok(()) => {
                self.pending.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                warn!("Compilation queue full, dropping request");
                false
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                error!("Compilation workers disconnected");
                false
            }
        }
    }

    /// Process any completed compilations and add them to the cache.
    ///
    /// Returns the number of new functions added to the cache.
    pub fn process_completions(&self) -> usize {
        let mut count = 0;
        while let Ok(result) = self.result_receiver.try_recv() {
            self.pending.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            if let Some(function) = result.function {
                self.cache.insert(result.hash, function, result.code_size);
                count += 1;
            }
        }
        count
    }

    /// Get the number of pending compilations.
    pub fn pending_count(&self) -> usize {
        self.pending.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Shutdown all workers gracefully.
    pub fn shutdown(&self) {
        let handles = std::mem::take(&mut *self.handles.lock());
        let worker_count = handles.len();

        // Send shutdown signal to all workers
        for _ in 0..worker_count {
            let _ = self.sender.send(WorkerMessage::Shutdown);
        }

        // Wait for workers to finish
        for handle in handles {
            let _ = handle.join();
        }

        info!("All compilation workers shut down");
    }
}

impl Drop for CompilationWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

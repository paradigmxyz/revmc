//! revmc integration for reth, OP-reth, and Tempo.
//!
//! This crate provides hot contract detection and background JIT compilation
//! for EVM bytecode execution in blockchain nodes.
//!
//! # Features
//!
//! - **Hot contract detection**: Tracks gas consumed per bytecode hash
//! - **Background JIT compilation**: Compiles hot contracts asynchronously
//! - **LRU cache**: Limits memory usage with configurable cache size
//! - **Thread-safe**: Uses proper synchronization primitives
//!
//! # Usage
//!
//! ```ignore
//! use revmc_reth::{RevmcConfig, RevmcRuntime};
//!
//! // Create runtime with default config
//! let runtime = RevmcRuntime::new(RevmcConfig::default())?;
//!
//! // Start background compilation worker
//! runtime.start_worker();
//!
//! // In your EVM execution loop:
//! if let Some(compiled) = runtime.get_compiled(bytecode_hash) {
//!     // Use compiled function
//! } else {
//!     // Track execution for hot detection
//!     runtime.record_execution(bytecode_hash, &bytecode, gas_used, spec_id);
//!     // Fall back to interpreter
//! }
//! ```

#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod cache;
mod config;
mod detector;
mod runtime;
mod worker;

pub use cache::{CacheStats, CompiledCache};
pub use config::RevmcConfig;
pub use detector::{DetectorStats, HotBytecode, HotDetector};
pub use runtime::{RevmcRuntime, RuntimeStats};
pub use worker::CompilationWorker;

// Re-export key types from revmc-context
pub use revmc_context::EvmCompilerFn;

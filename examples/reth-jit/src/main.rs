//! Example demonstrating revmc-reth hot contract JIT compilation.
//!
//! This example shows how to:
//! 1. Create a `RevmcRuntime` with custom configuration
//! 2. Record bytecode executions to trigger hot detection
//! 3. Query statistics and hit rates
//!
//! Run with: `cargo run -p revmc-examples-reth-jit`

use alloy_primitives::{hex, B256};
use revm_primitives::hardfork::SpecId;
use revmc_reth::{RevmcConfig, RevmcRuntime};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

/// Fibonacci bytecode (calculates fib(n) for input n).
const FIBONACCI_CODE: &[u8] =
    &hex!("5f355f60015b8215601a578181019150909160019003916005565b9150505f5260205ff3");

/// Simple counter bytecode (PUSH1 0x01, ADD, STOP).
const COUNTER_CODE: &[u8] = &hex!("60016001016000");

/// Compute the keccak256 hash of bytecode.
fn bytecode_hash(code: &[u8]) -> B256 {
    use alloy_primitives::keccak256;
    keccak256(code)
}

fn main() {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("revmc=debug".parse().unwrap()),
        )
        .init();

    println!("=== revmc-reth Hot Contract JIT Demo ===\n");

    // Create runtime with custom configuration:
    // - Low hot threshold (10,000 gas) for demo purposes
    // - Small cache for demonstration
    let config = RevmcConfig::default()
        .with_hot_threshold(10_000) // Mark as hot after 10k gas
        .with_cache_size(64);

    let runtime = Arc::new(RevmcRuntime::new(config));

    println!("Runtime created:");
    println!("  Enabled: {}", runtime.is_enabled());
    println!("  Hot threshold: {} gas", runtime.config().hot_threshold);
    println!("  Cache size: {}", runtime.config().cache_size);
    println!();

    // Get bytecode hashes
    let fib_hash = bytecode_hash(FIBONACCI_CODE);
    let counter_hash = bytecode_hash(COUNTER_CODE);

    println!("Bytecode hashes:");
    println!("  Fibonacci: {}", fib_hash);
    println!("  Counter:   {}", counter_hash);
    println!();

    // Simulate executions - fibonacci becomes hot, counter doesn't
    println!("Simulating executions...\n");

    // Execute fibonacci multiple times with high gas usage
    for i in 0..5 {
        let gas_used = 5_000; // 5k gas per execution
        runtime.record_execution(fib_hash, FIBONACCI_CODE, gas_used, SpecId::CANCUN);
        println!(
            "  Fibonacci execution {}: {} gas (total: {} gas)",
            i + 1,
            gas_used,
            (i + 1) * gas_used
        );

        // Check if it's been detected as hot
        if runtime.detector().is_hot(&fib_hash) {
            println!("  -> Fibonacci became HOT! Queued for compilation.");
        }
    }
    println!();

    // Execute counter just once - won't become hot
    runtime.record_execution(counter_hash, COUNTER_CODE, 1_000, SpecId::CANCUN);
    println!("  Counter execution 1: 1000 gas");
    println!();

    // Give background worker time to compile
    println!("Waiting for background compilation...");
    std::thread::sleep(std::time::Duration::from_millis(500));
    println!();

    // Check compilation status
    println!("Compilation status:");
    if runtime.get_compiled(fib_hash).is_some() {
        println!("  Fibonacci: COMPILED (JIT ready)");
    } else {
        println!("  Fibonacci: pending or interpreter-only");
    }
    if runtime.get_compiled(counter_hash).is_some() {
        println!("  Counter:   COMPILED (JIT ready)");
    } else {
        println!("  Counter:   interpreter-only (not hot)");
    }
    println!();

    // Print final statistics
    let stats = runtime.stats();
    println!("=== Runtime Statistics ===");
    println!("{}", stats);
    println!();
    println!("Detailed stats:");
    println!("  Total executions:      {}", stats.total_executions);
    println!("  JIT hits:              {}", stats.jit_hits);
    println!("  Interpreter fallbacks: {}", stats.interpreter_fallbacks);
    println!("  Hit rate:              {:.1}%", runtime.hit_rate());
    println!("  Cached functions:      {}/{}", stats.cached_functions, stats.cache_capacity);
    println!("  Hot contracts:         {}", stats.hot_contracts);
    println!("  Contracts tracked:     {}", stats.contracts_tracked);
    println!("  Pending compilations:  {}", stats.pending_compilations);

    // Shutdown worker threads cleanly
    runtime.shutdown();
    println!("\nRuntime shutdown complete.");
}

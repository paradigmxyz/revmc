# revmc-reth

Hot contract JIT compilation for reth, OP-reth, and Tempo.

## Overview

This crate provides automatic JIT compilation of frequently-executed ("hot") EVM bytecode.
It integrates with revm's handler system to transparently accelerate execution without 
requiring manual precompilation.

## Features

- **Hot contract detection**: Tracks cumulative gas usage per bytecode hash
- **Background compilation**: JIT compiles hot contracts in worker threads
- **LRU cache**: Limits memory usage with configurable cache size
- **Thread-safe**: Uses proper synchronization (no unsafe statics)
- **Zero-config fallback**: Always falls back to interpreter for uncompiled bytecode

## Usage

### Basic Integration

```rust
use revmc_reth::{RevmcConfig, RevmcRuntime};
use std::sync::Arc;

// Create runtime
let runtime = Arc::new(RevmcRuntime::new(RevmcConfig::default()));

// In your revm handler:
// 1. Check for compiled version
if let Some(compiled) = runtime.get_compiled(bytecode_hash) {
    // Use compiled function
    unsafe { compiled.call_with_interpreter_and_memory(interpreter, memory, host) }
} else {
    // Fall back to interpreter and record execution
    let result = interpreter.run(memory, instruction_table, host);
    runtime.record_execution(bytecode_hash, &bytecode, gas_used, spec_id);
    result
}
```

### CLI Flag Support

The runtime can be enabled/disabled at runtime:

```rust
// Parse CLI flag
let jit_enabled = std::env::args().any(|arg| arg == "--jit");

let config = RevmcConfig::default().with_enabled(jit_enabled);
let runtime = RevmcRuntime::new(config);

// Or toggle at runtime
runtime.set_enabled(false);
```

### Configuration Presets

```rust
// Low memory usage (smaller cache, single worker)
let config = RevmcConfig::low_memory();

// High performance (larger cache, more workers, aggressive optimization)
let config = RevmcConfig::high_performance();

// Custom configuration
let config = RevmcConfig::default()
    .with_hot_threshold(500_000)  // Gas threshold for "hot" detection
    .with_cache_size(2048);       // Max compiled functions to cache
```

## Integration with Different Clients

### reth / OP-reth

```rust
use reth_evm::handler::EvmHandler;

fn register_jit_handler<DB>(handler: &mut EvmHandler<DB>, runtime: Arc<RevmcRuntime>) {
    let prev = handler.execution.execute_frame.clone();
    handler.execution.execute_frame = Arc::new(move |frame, memory, tables, context| {
        let hash = frame.interpreter().contract.hash.unwrap_or_default();
        
        if let Some(compiled) = runtime.get_compiled(hash) {
            // JIT path
            unsafe { compiled.call_with_interpreter_and_memory(...) }
        } else {
            // Interpreter fallback
            prev(frame, memory, tables, context)
        }
    });
}
```

### Tempo (AA Multi-call)

For Tempo's Account Abstraction with multi-call batches, each call target is 
tracked independently:

```rust
for call in aa_batch.calls {
    let target_hash = call.target_bytecode_hash;
    // JIT compilation applies per-contract, not per-transaction
    runtime.record_execution(target_hash, &call.bytecode, gas_used, spec_id);
}
```

## Statistics

```rust
let stats = runtime.stats();
println!("JIT hit rate: {:.1}%", runtime.hit_rate());
println!("Cached: {}/{}", stats.cached_functions, stats.cache_capacity);
println!("Hot contracts: {}", stats.hot_contracts);
```

## License

MIT OR Apache-2.0

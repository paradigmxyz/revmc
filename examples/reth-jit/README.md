# revmc-reth JIT Example

Demonstrates hot contract JIT compilation using `revmc-reth`.

## What This Shows

1. **Creating a `RevmcRuntime`** with custom configuration
2. **Hot detection** triggering compilation when gas threshold is reached
3. **Background compilation** by worker threads
4. **Statistics output** including hit rate, cache usage, and pending compilations

## Prerequisites

LLVM 18 must be installed:

```bash
# macOS
brew install llvm@18

# Set environment variables
export LLVM_SYS_180_PREFIX=$(brew --prefix llvm@18)
export LIBRARY_PATH=$(brew --prefix)/lib
```

## Running

```bash
cargo run -p revmc-examples-reth-jit
```

## Example Output

```
=== revmc-reth Hot Contract JIT Demo ===

Runtime created:
  Enabled: true
  Hot threshold: 10000 gas
  Cache size: 64

Bytecode hashes:
  Fibonacci: 0xab1ad...
  Counter:   0x509c9...

Simulating executions...

  Fibonacci execution 1: 5000 gas (total: 5000 gas)
  Fibonacci execution 2: 5000 gas (total: 10000 gas)
  -> Fibonacci became HOT! Queued for compilation.
  ...

Waiting for background compilation...

Compilation status:
  Fibonacci: COMPILED (JIT ready)
  Counter:   interpreter-only (not hot)

=== Runtime Statistics ===
RevmcRuntime: enabled=true, hit_rate=0.0%, cached=1/64, hot=1, pending=0
```

## Integration Notes

In a real reth/Tempo integration, you would:

1. Create an `Arc<RevmcRuntime>` at node startup
2. Register a custom frame execution handler that calls `runtime.execute_or_fallback()`
3. The runtime automatically tracks executions and compiles hot contracts

See the [revmc-reth README](../../crates/revmc-reth/README.md) for full integration examples.

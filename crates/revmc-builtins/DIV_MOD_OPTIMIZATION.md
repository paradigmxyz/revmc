# DIV/MOD: ruint builtins vs LLVM i256

## Problem

LLVM's generic `i256 udiv/urem` lowering produces significantly worse machine code
than [ruint](https://crates.io/crates/ruint)'s hand-optimized division algorithm.
This caused the JIT compiler to be **43% slower** than the interpreter on
division-heavy EVM workloads.

### Root Cause

| Aspect | LLVM i256 udiv | ruint wrapping_div |
|--------|---------------|--------------------|
| Algorithm | Generic binary long-division | Möller-Granlund reciprocal (div_2x1_mg10) |
| Reciprocal precomputation | None | Yes — converts division to multiply |
| Per-operation cost (u256÷u64) | ~15-50 ns | ~2.7 ns (from ruint benchmarks) |
| Branch pattern | Branch-heavy (hard to predict) | Branch-light (straight-line code) |
| I-cache pressure | High (large lowered sequence) | Low (~20 instructions) |

ruint's `div_2x1_mg10` converts division into a reciprocal multiplication, which is
dramatically faster on modern CPUs. LLVM's generic legalization for >128-bit integers
does not apply this optimization.

## Fix

Replace LLVM's native `i256 udiv/urem` IR instructions with `extern "C"` builtin
calls that use ruint's `U256::wrapping_div()` / `U256::wrapping_rem()` for the
EVM `DIV` (0x04) and `MOD` (0x06) opcodes.

## A/B Test Results

### Curve StableSwap (division-heavy: Newton iteration in `get_y()`)

Single-transaction benchmark (`cargo bench --bench bench_curve`):

| Version | plain (interpreter) | jit | JIT speedup |
|---------|-------------------|-----|-------------|
| **Before** (LLVM i256 udiv) | 76.0 µs | 108.7 µs | **0.70x (43% regression)** |
| **After** (ruint builtin) | 76.0 µs | 36.2 µs | **2.10x** |

**Net improvement: 3.0x** (from 0.70x to 2.10x).

### Uniswap V2 Swap (moderate division: `getAmountOut()`)

Single-transaction benchmark (`cargo bench --bench bench_json_test`):

| Version | plain (interpreter) | jit | JIT speedup |
|---------|-------------------|-----|-------------|
| **Before** (LLVM i256 udiv) | 53.2 µs | 29.9 µs | 1.78x |
| **After** (ruint builtin) | 53.2 µs | 22.5 µs | **2.37x** |

### ERC20 Airdrop (no division: pure storage SLOAD/SSTORE)

Single-transaction benchmark (`cargo bench --bench bench_airdrop`):

| Recipients | plain | jit | JIT speedup |
|-----------|-------|-----|-------------|
| 200 | 7.0 µs | 6.8 µs | 1.02x |
| 2,000 | 49.0 µs | 48.6 µs | 1.01x |
| 20,000 | 472.4 µs | 466.9 µs | 1.01x |

No regression on workloads that don't use DIV/MOD.

## What about MUL, SDIV, SMOD?

A/B tested — converting these to builtins is **slower**:

| Change | Curve JIT (µs) | vs DIV/MOD-only |
|--------|---------------|-----------------|
| DIV/MOD only (this PR) | 36.2 | baseline |
| + SDIV/SMOD builtin | 45.8 | 27% slower |
| + MUL builtin | 47.2 | 30% slower |

LLVM's i256 codegen for multiplication (`imul`) and signed division (`sdiv/srem`)
is already competitive with ruint. The function-call overhead of a builtin outweighs
any algorithmic advantage. Only **unsigned division/remainder** has a large enough
gap to benefit.

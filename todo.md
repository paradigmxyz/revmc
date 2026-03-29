# revmc Runtime TODO

Tracked issues and remaining work for the `runtime` module, separate from `plan.md`.

## API / Lifecycle

- [ ] Control APIs (`clear_*`, `prepare_aot`, `compile_jit`) use `try_send()` — silently drop on full channel. Add error feedback or use blocking send for control commands.
- [ ] Add `reconfigure(RuntimeConfigUpdate)` API.
- [ ] Add `RuntimeError` / `error.rs` instead of using `eyre` throughout public API.
- [ ] `InterpretReason::{QueueSaturated, UnsupportedBackend, Invalidated}` are defined but never emitted in normal lookup flow.

## Eviction / Accounting

- [ ] Add `approx_size_bytes` to `CompiledProgram` (plan marks it ✅ but it's absent).
- [ ] Budget eviction uses global LLVM JIT memory (`jit_total_bytes()`), not per-program resident accounting. Can evict AOT entries that don't reduce JIT bytes, and can overshoot.
- [ ] Document or fix: eviction is a global JIT memory pressure heuristic, not the plan's per-entry resident budget.

## AOT Prepare

- [x] `prepare_aot()` does not probe storage before compiling — recompiles even if already persisted.
- [ ] `prepare_aot()` skips if key is already resident (even as JIT) — JIT-resident code never gets persisted as AOT for faster startup.

## Correctness / State Machine

- [x] `on_compilation` callback classifies all failures as `CompilationKind::Jit`, including AOT failures.
- [x] Stats counters `jit_promotions` / `jit_successes` / `jit_failures` are incremented for AOT work too — metric names are misleading. Renamed to `compilations_dispatched` / `compilations_succeeded` / `compilations_failed`.
- [ ] Failed entry retry: `EntryPhase::Failed` is sticky forever. Add `negative_jit_ttl` / `next_retry_at`.
- [x] `set_enabled()` mutates `AtomicBool` directly, bypassing coordinator. Accepted: atomic toggle is sufficient; coordinator routing adds complexity without benefit since the flag is only checked on the lookup hot path.
- [x] Tx-local lookup cache in integration layers means repeated calls within one tx don't emit tracking events. Accepted: this is a deliberate performance tradeoff — caching avoids redundant backend calls on the hot path. Hotness accuracy is "good enough" since distinct txs still emit events.

## Artifact Versioning

- [ ] `ArtifactKey` lacks `abi_version`, `revmc_semver`, `compiler_fingerprint`. Persisted AOT artifacts may be incompatible across revmc upgrades.

## Integration Layers

- `alloy_evm.rs` bypasses JIT when `inspect == true` — intentional, inspect is not jittable.
- [x] `alloy_evm.rs` system calls now go through `JitHandler`, matching `revm_evm.rs` behavior.

## Shutdown Safety

Not an issue. JIT code lifetime is managed by `Arc<JitCodeBacking>` (holds ORCv2
`ResourceTracker` + `JitDylibGuard`). The `GlobalOrcJit` is `'static`. Dropping the
last `JitBackend` clone stops the backend thread and workers but does not free any
code — outstanding `Arc<CompiledProgram>` holders remain valid. No public `shutdown()`
is exposed; cleanup is purely drop-based.

## Test Coverage

- [x] Concurrent multi-thread lookup on same key.
- [x] Channel saturation behavior (full command channel).
- [x] Single JIT admission per key under contention.
- [x] Multiple `SpecId`s in same backend.
- [x] `prepare_aot()` with already-persisted-but-not-resident artifact.
- [x] Resident JIT + `prepare_aot()` should still persist.
- [x] Clear / shutdown during active compilation work.
- [x] CREATE2 / repeated factory deployments.
- [x] Nested `CALL*` / `CREATE*` suspend-resume (revm_evm integration).
- [x] Integration tests in non-blocking mode.
- [x] `set_enabled` toggle while compilations are in-flight.
- [x] Stats accuracy under concurrent load.
- [x] `compile_jit_sync` blocks and deduplicates.
- [x] `on_compilation` callback correctness.
- [x] CREATE + CALL integration (revm_evm).

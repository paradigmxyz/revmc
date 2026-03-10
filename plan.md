# revmc Runtime JIT Coordinator Plan

## Status

Proposed.

## Summary

Add a new `revmc::runtime` module that provides a production-grade runtime JIT coordinator service for reth and other users.

The service should:

1. Decide compiled vs interpreter execution at runtime.
1. Keep the execution hot path lock-light and non-blocking.
1. Perform background JIT compilation off-path.
1. Support direct compile requests for explicit prewarming or blocking compilation.
1. Support AOT persistence and reload through an abstract storage backend.
1. Centralize policy and lifecycle changes in one coordinator thread.
1. Run compilation work on a custom bounded rayon worker pool.

The first production version should intentionally use a simple 2-tier model:

1. Tier 0: interpreter.
1. Tier 1: compiled code (JIT-loaded in memory, or AOT-loaded from storage).

No speculative optimization, no mid-frame tier switching, and no deoptimization machinery.

## Goals

### Functional goals

1. Provide a reusable runtime service usable by reth execution, benchmarks, statetest-like runners, and other revm/revmc embedders.
1. Decide on each bytecode entry whether to run compiled code immediately or run the interpreter and queue background compilation.
1. Support explicit/direct compile requests for direct JIT compile, direct AOT compile and persist, and direct AOT load if artifact already exists.
1. Support runtime-created code from `CREATE` and `CREATE2`.
1. Support runtime enable/disable, clear/reset, tuning updates, worker count changes, backend changes, and storage backend changes.
1. Preserve correctness under `SpecId`-sensitive semantics, dynamic gas behavior, warm/cold state, suspend/resume for `CALL*` and `CREATE*`, and interpreter fallback on all miss/error paths.

### Operational goals

1. No blocking waits on the normal execution hot path.
1. Bounded background work and bounded memory growth.
1. Strong observability for rollout and tuning.
1. Safe invalidation/clear semantics.

## Non-Goals

The initial module will not attempt to provide OSR, multi-stage optimizing recompilation beyond interpreter to compiled, speculative guards and deopt, cross-contract inlining, whole-block or whole-transaction compilation, network/distributed artifact caches, guaranteed cross-machine artifact portability, automatic compilation of every seen contract, or hot-path waiting for compile completion.

## Why This Shape Fits Ethereum Execution

Ethereum node execution is different from browser JS or JVM workloads:

1. Many contracts are cold and executed only once.
1. Correctness is more important than peak per-contract throughput.
1. Semantics are fork-sensitive (`SpecId`).
1. Runtime state affects gas and behavior.
1. Code can appear dynamically via `CREATE/CREATE2`.
1. Execution is mostly synchronous and latency-sensitive.

That makes a simple design the right default:

1. Keep interpreter as the always-correct baseline.
1. Compile only after repeated use or an explicit request.
1. Publish compiled code asynchronously.
1. Never stall transaction/frame execution waiting for background compilation.

## Proposed Public API

```rust
pub mod runtime {
    pub struct JitCoordinator;
    pub struct JitCoordinatorHandle;

    pub struct RuntimeConfig;
    pub struct RuntimeTuning;
    pub struct RuntimeStatsSnapshot;

    pub struct LookupRequest<'a> {
        pub code_hash: B256,
        pub code: &'a [u8],
        pub spec_id: SpecId,
    }

    pub enum LookupDecision {
        Compiled(Arc<CompiledProgram>),
        Interpret(InterpretReason),
    }

    pub enum InterpretReason {
        Disabled,
        NotReady,
        QueueSaturated,
        CompileFailed,
        LoadFailed,
        UnsupportedBackend,
        Invalidated,
    }

    pub struct CompileRequest<'a> {
        pub code_hash: B256,
        pub code: Cow<'a, [u8]>,
        pub spec_id: SpecId,
        pub flavor: CompileFlavor,
        pub priority: CompilePriority,
    }

    pub enum CompileFlavor {
        Jit,
        AotPersist,
        AotLoadOrCompilePersist,
    }

    pub enum CompilePriority {
        Direct,
        Warm,
        ColdLoad,
    }

    pub struct CompileTicket;
    impl CompileTicket {
        pub fn wait(self) -> Result<Arc<CompiledProgram>, RuntimeError>;
        pub fn wait_timeout(self, timeout: Duration) -> Result<Option<Arc<CompiledProgram>>, RuntimeError>;
    }

    pub struct CompiledProgram {
        pub key: RuntimeCacheKey,
        pub kind: ProgramKind,
        pub func: EvmCompilerFn,
    }

    pub enum ProgramKind {
        Jit,
        AotLoaded,
    }

    impl JitCoordinatorHandle {
        pub fn lookup(&self, req: LookupRequest<'_>) -> LookupDecision;
        pub fn compile_now(&self, req: CompileRequest<'_>) -> Result<CompileTicket, RuntimeError>;

        pub fn set_enabled(&self, enabled: bool) -> Result<(), RuntimeError>;
        pub fn reconfigure(&self, update: RuntimeConfigUpdate) -> Result<(), RuntimeError>;
        pub fn clear(&self, scope: ClearScope) -> Result<(), RuntimeError>;
        pub fn stats(&self) -> RuntimeStatsSnapshot;
        pub fn shutdown(self) -> Result<(), RuntimeError>;
    }
}
```

### API behavior

1. `lookup()` is the hot-path API.
1. `lookup()` must never block, avoid coarse locks, return compiled code immediately if ready, otherwise queue work opportunistically and return `Interpret(...)`.
1. `compile_now()` is the explicit/blocking or prewarm path.
1. `clear()` and `reconfigure()` are control-plane operations and may synchronize with the coordinator thread.

## Module Layout

Create a new module under `crates/revmc/src/runtime/`:

```text
runtime/
  mod.rs
  api.rs
  config.rs
  coordinator.rs
  cache.rs
  entry.rs
  worker.rs
  artifact.rs
  storage.rs
  stats.rs
  error.rs
```

### Responsibilities

1. `api.rs`: public handle/types and request/result enums.
1. `config.rs`: runtime config, tuning, defaults.
1. `coordinator.rs`: single-threaded control loop, commands, invalidation, scheduling, worker-pool rebuild.
1. `cache.rs`: runtime entry map and lock-light hot-path lookup helpers.
1. `entry.rs`: entry state machine, atomics, metadata.
1. `worker.rs`: compile task execution, custom rayon pool, batch construction.
1. `artifact.rs`: compiled backing, artifact keys/manifests, library loading handles.
1. `storage.rs`: abstract storage trait, filesystem backend, no-op backend.
1. `stats.rs`: counters, gauges, snapshots.
1. `error.rs`: typed runtime errors.

Keep the core module generic. Any reth-specific adapter should stay downstream or in a thin optional adapter layer.

## Core Architecture

### 1. Hot path: lock-light lookup

The hot path should do only:

1. Check global enabled/config state.
1. Compute `RuntimeCacheKey = (code_hash, spec_id)`.
1. Fetch/create entry from a concurrent map.
1. Atomically check if compiled program is ready.
1. If ready, return it.
1. Otherwise update hotness/first-seen metadata and try to enqueue background work.
1. Return interpreter.

#### Hot-path invariants

1. No waiting.
1. No worker-pool interaction directly.
1. No filesystem access.
1. No linker invocation.
1. No storage backend calls.
1. No blocking lock on a shared global mutex.

A `DashMap<RuntimeCacheKey, Arc<Entry>>` is acceptable in v1. Inside each `Entry`, compiled publication should use atomic-publish semantics (`ArcSwapOption` or equivalent).

### 2. Single coordinator thread

A dedicated coordinator thread owns control-plane logic:

1. Queue admission.
1. Prioritization.
1. Deduplication.
1. Background scheduling.
1. Artifact load/store bookkeeping.
1. Cache eviction.
1. Runtime reconfiguration.
1. Generation-based invalidation.
1. Worker-pool rebuild.
1. Clear/shutdown sequencing.

#### Why single coordinator

1. Simpler reasoning.
1. Deterministic control flow.
1. Easier invalidation and lifecycle management.
1. Avoid subtle multi-writer races between queueing, clearing, and reconfiguration.

### 3. Compile workers: custom rayon thread pool

Compilation happens on a dedicated rayon pool, not execution threads.

#### Requirements

1. Custom thread count.
1. Custom thread names (`revmc-compile-{i}`).
1. Bounded work admission.
1. No use of global rayon pool.

#### Default worker count

`min(max(1, available_parallelism / 2), 4)`

Examples:

1. 2 cores: 1 worker.
1. 4 cores: 2 workers.
1. 8 cores: 4 workers.
1. 32 cores: 4 workers.

#### Rationale

Compilation is CPU-heavy and should not starve block/tx execution, networking, DB activity, or trie/state work.

## Runtime Configuration And Defaults

### Top-level config

```rust
pub struct RuntimeConfig {
    pub enabled: bool,
    pub backend: BackendSelection,
    pub storage: StorageConfig,
    pub tuning: RuntimeTuning,
}
```

### Backend selection

```rust
pub enum BackendSelection {
    Auto,
    Llvm,
    Cranelift,
}
```

Default: `Auto`.

Behavior:

1. Prefer LLVM when available.
1. Return `UnsupportedBackend` if requested backend is not built in.
1. Keep Cranelift feature-gated/experimental until ready for production revmc requirements.

### Storage selection

```rust
pub enum StorageConfig {
    None,
    Filesystem { root: PathBuf },
    Custom(Arc<dyn ArtifactStore>),
}
```

Default: `None`.

### Tuning defaults

| Knob | Default | Why |
|---|---:|---|
| `enabled` | `false` | Safe rollout default for node operators |
| `warm_threshold` | `8` | Filters one-off contracts but promotes hot code quickly |
| `load_persisted_on_first_seen` | `true` | Cheap path to reclaim prewarmed artifacts |
| `max_pending_jobs` | `2048` | Bounded memory/backpressure under bursts |
| `reserved_direct_slots` | `64` | Keep direct compile requests responsive |
| `batch_max_items` | `8` | Good amortization without large per-batch latency |
| `batch_max_total_bytecode_bytes` | `256 KiB` | Avoid giant batches dominating workers |
| `batch_max_wait` | `2 ms` | Micro-batching without visible delay |
| `jit_opt_level` | `Default` | Better compile-latency/runtime tradeoff for background JIT |
| `aot_opt_level` | `Aggressive` | Better persisted artifact quality |
| `negative_compile_ttl` | `300 s` | Avoid repeated compile storms on persistent failures |
| `negative_load_ttl` | `60 s` | Avoid repeated storage miss/load storms |
| `enqueue_cooldown` | `1 s` | Avoid queue hammering on hot misses/full queues |
| `resident_code_cache_bytes` | `128 MiB` | Predictable memory bound for node workloads |
| `persisted_artifact_budget_bytes` | `512 MiB` | Sensible default disk budget |
| `worker_count` | `min(max(1, cpus/2), 4)` | Leave CPU headroom for node work |

## Data Model

### Runtime cache key

```rust
pub struct RuntimeCacheKey {
    pub code_hash: B256,
    pub spec_id: SpecId,
}
```

### Persisted artifact key

```rust
pub struct ArtifactKey {
    pub runtime: RuntimeCacheKey,
    pub backend: BackendSelection,
    pub target: Target,
    pub opt_level: OptimizationLevel,
    pub revmc_semver: String,
    pub compiler_fingerprint: String,
    pub abi_version: u32,
}
```

Persisted machine code must be invalidated by backend/target/version/ABI fingerprint, not just hash.

## Entry Model

Each runtime entry should contain:

```rust
struct Entry {
    ready: ArcSwapOption<CompiledProgram>,
    state: AtomicU8,
    hotness: AtomicU32,
    generation: AtomicU64,
    last_used_at: AtomicU64,
    next_retry_at: AtomicU64,
    fail_count: AtomicU32,
    load_attempted: AtomicBool,
}
```

### Entry states

```rust
enum EntryState {
    Cold,
    LoadQueued,
    Queued,
    Compiling,
    Ready,
    Failed,
}
```

### State transitions

```text
Absent -> Cold

Cold --first seen + load enabled--> LoadQueued
LoadQueued -> Compiling(load)
Compiling(load) -> Ready
Compiling(load) -> Cold
Compiling(load) -> Failed

Cold --hotness >= threshold--> Queued
Queued -> Compiling(compile)
Compiling(compile) -> Ready
Compiling(compile) -> Failed

Failed --retry ttl elapsed--> Cold
Ready --evicted--> Cold
Any --clear/reconfigure--> Cold(next generation)
```

Only `Ready` changes execution choice. All other states mean interpret now.

## Compiled Program Lifetime Model

A raw `EvmCompilerFn` is not sufficient on its own. Backing allocation must outlive active calls.

```rust
pub struct CompiledProgram {
    pub key: RuntimeCacheKey,
    pub kind: ProgramKind,
    pub func: EvmCompilerFn,
    pub approx_size_bytes: usize,
    backing: ProgramBacking,
}

enum ProgramBacking {
    JitModule(Arc<JitModuleOwner>),
    LoadedLibrary(Arc<LoadedLibraryOwner>),
}
```

### JIT lifetime rule

1. Compile a batch in a dedicated compiler/module owner.
1. Publish `CompiledProgram` per symbol.
1. Keep module owner alive behind `Arc`.
1. Eviction removes cache strong ref.
1. Memory reclaimed only when no active execution still holds `Arc<CompiledProgram>`.

### AOT lifetime rule

1. Load shared library.
1. Resolve symbol.
1. Keep loaded library and temp file backing alive in an `Arc`.
1. Use same eviction semantics.

## Coordinator Commands

```rust
enum Command {
    EnqueueWarm(RuntimeCacheKey, Bytes, SpecId),
    EnqueueLoad(RuntimeCacheKey),
    DirectCompile(CompileRequestOwned, ReplySender),
    SetEnabled(bool, ReplySender<()>),
    Reconfigure(RuntimeConfigUpdate, ReplySender<()>),
    Clear(ClearScope, ReplySender<()>),
    WorkerFinished(WorkerResult),
    Shutdown(ReplySender<()>),
}
```

### Priority order

1. Direct compile requests.
1. Persisted artifact loads.
1. Warm-threshold background compile requests.

## Queueing And Backpressure

Use a bounded coordinator-owned priority queue with dedupe by key + generation + flavor.

Rules:

1. If request arrives for already `Queued`, `Compiling`, or `Ready`, ignore.
1. If queue full, drop low-priority background work, set `next_retry_at = now + enqueue_cooldown`, continue interpreter fallback.
1. Direct compile requests consume reserved capacity and are not silently dropped.

## Compilation Pipeline

### Background JIT compile path

1. Hot path sees miss and warm threshold reached.
1. Coordinator admits warm compile request.
1. Coordinator forms micro-batch.
1. Worker compiles batch to JIT.
1. Worker publishes `CompiledProgram`s.
1. Coordinator stores results and updates resident usage.

### Batch grouping

Group by backend, compile flavor, target, and optimization level. `SpecId` remains per item.

## Direct Compile Path

`compile_now()` supports:

1. `Jit`: high-priority compile, publish, return ticket.
1. `AotPersist`: compile AOT, persist artifact, optionally load+publish, return ticket.
1. `AotLoadOrCompilePersist`: try load first, otherwise AOT compile/persist/load/publish.

## AOT Persistence Design

### Storage trait

```rust
pub trait ArtifactStore: Send + Sync + 'static {
    fn load(&self, key: &ArtifactKey) -> Result<Option<StoredArtifact>, StorageError>;
    fn store(&self, key: &ArtifactKey, artifact: &StoredArtifact) -> Result<(), StorageError>;
    fn delete(&self, key: &ArtifactKey) -> Result<(), StorageError>;
    fn clear(&self) -> Result<(), StorageError>;
}
```

### Stored artifact

```rust
pub struct StoredArtifact {
    pub manifest: ArtifactManifest,
    pub dylib_bytes: Bytes,
}
```

### Manifest

```rust
pub struct ArtifactManifest {
    pub artifact_key: ArtifactKey,
    pub symbol_name: String,
    pub bytecode_len: usize,
    pub artifact_len: usize,
    pub created_at_unix_secs: u64,
    pub sha256: [u8; 32],
}
```

Store final dylib bytes in v1 for simple reload path.

## Execution Lifecycle Hooks

### Generic embedder hooks

1. `on_enter_bytecode`: lookup, hotness increment, queueing, compiled hit return.
1. `on_created_runtime_code`: register runtime code from `CREATE/CREATE2`, optional low-priority load/compile queueing.
1. `on_admin_event`: enable/disable, clear, reconfigure, stats, shutdown.

### reth integration behavior

1. Call `lookup()` on frame entry.
1. If `Compiled`, run `EvmCompilerFn::call_with_interpreter(...)`.
1. Otherwise run interpreter.
1. If compiled execution suspends via existing `CALL*`/`CREATE*` semantics, continue existing suspend/resume machinery.
1. If `CREATE/CREATE2` produces runtime code, notify coordinator with new bytecode.

No frame should wait for background compilation.

## Correctness Invariants

1. Interpreter is always safe fallback for disabled runtime, not-ready entries, queue full, compile/load errors, unsupported backend, stale generation, and clear in progress.
1. Cache key includes `SpecId`.
1. No speculative state assumptions about warm/cold, gas progression, storage/account state, or non-suspending flow.
1. Runtime-created code is first-class and uses same keying/policy.
1. Clear/reconfigure uses generations; stale worker results are discarded.
1. Backing object lifetime is explicit; no function pointer outlives module/library owner.

## Cache Eviction And Invalidation

### Initial eviction policy

Coordinator-managed LRU-ish policy by `last_used_at` and `approx_size_bytes` against resident budget.

When over budget:

1. Remove least-recently-used `Ready` entries.
1. Drop only cache strong reference.
1. Allow memory to free naturally when active users release remaining `Arc`s.

### Invalidations

Support:

1. Clear resident only.
1. Clear persisted only.
1. Clear all.
1. Reconfigure backend implies generation bump plus resident clear.
1. Reconfigure worker count rebuilds worker pool.
1. Disable stops new queue admission but existing resident programs remain callable until evicted/cleared.

## Observability

Expose metrics for lookup outcomes, queue depth/enqueue/drop, compile totals/duration/batch sizing, storage ops, resident bytes/evictions/generation, and direct compile latency/outcomes.

Add tracing spans around lookup enqueue decisions, batch formation, compile jobs, AOT link/store/load, and clear/reconfigure/shutdown.

## Testing Strategy

### 1. Unit tests

Cover entry transitions, generation invalidation, queue dedupe, backpressure, negative TTLs, resident budget eviction, and direct compile wait semantics.

### 2. Concurrency tests

Use claim/wait ideas from statetest `CompileCache`: many threads same `(code_hash, spec_id)`, only one compile admitted, others interpret on hot path or wait only via direct compile.

### 3. Correctness tests vs interpreter

State tests/block replays asserting same result, gas behavior, halts, suspend/resume across `CALL*` and `CREATE*`, and behavior across multiple `SpecId`s.

### 4. Runtime-created code tests

Exercise `CREATE`, `CREATE2`, repeated factory deployments of identical runtime code, constructor vs runtime code distinction, first-seen vs re-seen behavior.

### 5. Failure injection

Compile errors, backend creation errors, link errors, storage failures, queue saturation, clear during compile, shutdown during compile. All must preserve correctness by interpreter fallback.

### 6. Persistence tests

Artifact round-trip, fingerprint mismatch rejection, reload after restart, budget cleanup, symbol resolution correctness.

### 7. Performance tests

Cold-heavy, hot-heavy, mixed replay, sync-like bursts. Track fallback rate, queue saturation, compile latency, warmup hit-rate, resident memory.

## Rollout Plan

### Phase 0: core API + direct compile only

Implement `compile_now()` and direct JIT compile. No automatic background queueing. Feature-gated and off by default.

### Phase 1: passive runtime tracking

Implement `lookup()` and track hotness/stats, but still interpret always.

### Phase 2: background JIT

Enable enqueue-on-threshold and compiled publication. No persistence required yet.

### Phase 3: AOT persistence

Add storage trait, filesystem backend, AOT load/store flow, and direct `AotLoadOrCompilePersist`.

### Phase 4: operational hardening

Add eviction, reconfigure/clear/shutdown polish, block replay tuning. Keep default disabled until field data validates settings.

## Explicit Runtime Knobs

Provide runtime controls for:

1. Enable/disable JIT.
1. Clear resident state.
1. Clear persisted state.
1. Clear all state.
1. Update warm threshold.
1. Update queue/backpressure parameters.
1. Update worker count.
1. Rebuild worker pool.
1. Switch backend.
1. Switch storage backend.
1. Update optimization levels.
1. Update resident code cache limit.
1. Fetch stats snapshot.

Control-plane changes should flow through coordinator thread. Destructive changes should bump generation.

## Borrowed JIT Ideas To revmc Choices

| Borrowed idea | Source inspiration | revmc design choice |
|---|---|---|
| Warmup before compile | V8, HotSpot, SpiderMonkey, YJIT | `warm_threshold = 8` before background JIT |
| Keep miss path cheap | YJIT, JSC | hot path returns interpreter immediately, no waits |
| Bounded compile workers | RPCS3, browser engines, JVMs | dedicated rayon pool with conservative defaults |
| Bounded compile queue | HotSpot, V8, RPCS3 | `max_pending_jobs = 2048`, reserved direct slots |
| Small compile batches | statetest `CompileCache`, JVM practice | `batch_max_items = 8`, `batch_max_wait = 2ms` |
| Code cache limits | V8, JSC, HotSpot | resident cap + LRU-ish eviction |
| Negative caching/backoff | production JIT patterns | compile/load negative TTLs |
| Runtime knobs | V8, HotSpot, RPCS3 | enable/disable/clear/tune workers/backend/storage |
| Persistent code cache | JSC/V8-style code cache ideas | AOT artifacts via `ArtifactStore` |
| Invalidations via generations | browser/JVM runtime control | generation bump on clear/reconfigure |
| Thread-affine compiler concerns | revmc/LLVM, RPCS3 | compile only on dedicated workers |
| Avoid speculative deopt complexity | PyPy/HotSpot lessons + EVM determinism | no speculative guards/OSR in v1 |

## Why Some JIT Ideas Are Deferred

OSR, speculative guards/deopt, and deep multi-tiering are deliberately deferred. Interpreter to compiled covers most initial value while keeping complexity and correctness risk low.

## When To Consider Advanced Path

Revisit design if queue saturation is sustained, hot contracts remain interpreter-bound too long, startup repeatedly misses persisted artifacts, cache churn is high, or compile latency dominates hot code.

Potential upgrades:

1. Adaptive thresholds by code size.
1. Startup artifact prefetch.
1. Separate low-opt quick-JIT and high-opt AOT policies.
1. Operator prewarm CLI from block traces or snapshots.

## Implementation Notes Tied To Current revmc

1. `EvmCompiler` is single-threaded per instance; do not share one compiler across threads.
1. Reuse statetest cache pattern: dedupe by `(code_hash, spec_id)` with claim/wait semantics.
1. Keep compile batches intentionally small.
1. Use existing JIT/AOT paths and `Linker` for final AOT dylib creation.
1. Treat runtime-created bytecode as first-class cache candidates.
1. Keep interpreter fallback as the universal correctness escape hatch.

## Final Recommendation

Ship the smallest production-safe version first: single coordinator thread, bounded rayon pool, lock-light lookup, direct compile API, background JIT on warm threshold, optional AOT persistence, and interpreter fallback in every error/miss path.

# revmc Runtime JIT Coordinator Plan

## Status

Proposed.

## Summary

Add a new `revmc::runtime` module with a simple coordinator-owned model.

There are two consumers:

1. EVM execution lookup: use compiled code if present, prefer AOT artifact load when available, otherwise follow normal interpreter + JIT warmup path.
1. Explicit AOT preparation: user provides contracts to ensure AOT artifacts exist (compile and persist if missing).

Core constraints:

1. Hot path must be non-blocking.
1. State transitions must be coordinator-only.
1. Runtime JIT output is process-local and never persisted.
1. AOT artifacts are persisted through `ArtifactStore`.

The runtime remains a strict 2-tier system:

1. Tier 0: interpreter.
1. Tier 1: compiled (`Aot` or `Jit`).

No OSR, no deopt, no speculative guards, no mid-frame switching.

## Scope

### In Scope

1. Fast lookup API for execution.
1. Unified in-memory compiled map containing both loaded AOT and JIT programs.
1. On-miss AOT load attempt (when storage configured), with fallback to normal path.
1. Warm-threshold runtime JIT compilation.
1. Explicit AOT prepare API (compile + store when absent).
1. Coordinator-owned scheduling, dedupe, retries, and invalidation.

### Out Of Scope

1. Speculative multi-tier optimization.
1. Distributed artifact caches.
1. Guaranteed cross-machine artifact portability.
1. Hot-path waiting for compilation or load completion.

## Goals

### Functional

1. Reusable service for reth and other revm/revmc embedders.
1. Correct execution under `SpecId`-sensitive semantics.
1. Correct suspend/resume behavior for `CALL*` and `CREATE*` through existing interpreter fallback machinery.
1. Support `CREATE/CREATE2` runtime code as normal JIT candidates.

### Operational

1. No waits on frame execution path.
1. Bounded queue and resident memory.
1. Predictable clear/reconfigure/shutdown behavior.
1. Rollout-friendly observability.

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
        AotMissing,
        AotLoadFailed,
        JitFailed,
        UnsupportedBackend,
        Invalidated,
    }

    pub struct AotRequest<'a> {
        pub code_hash: B256,
        pub code: Cow<'a, [u8]>,
        pub spec_id: SpecId,
    }

    pub struct CompiledProgram {
        pub key: RuntimeCacheKey,
        pub kind: ProgramKind,
        pub func: EvmCompilerFn,
        pub approx_size_bytes: usize,
    }

    pub enum ProgramKind {
        Aot,
        Jit,
    }

    impl JitCoordinatorHandle {
        pub fn lookup(&self, req: LookupRequest<'_>) -> LookupDecision;

        // Explicit APIs. These are enqueue-only and do not return wait tickets.
        pub fn compile_jit(&self, req: LookupRequest<'_>) -> Result<(), RuntimeError>;
        pub fn prepare_aot(&self, req: AotRequest<'_>) -> Result<(), RuntimeError>;
        pub fn prepare_aot_batch(&self, reqs: Vec<AotRequest<'_>>) -> Result<(), RuntimeError>;

        pub fn set_enabled(&self, enabled: bool) -> Result<(), RuntimeError>;
        pub fn reconfigure(&self, update: RuntimeConfigUpdate) -> Result<(), RuntimeError>;
        pub fn clear_resident(&self) -> Result<(), RuntimeError>;
        pub fn clear_persisted(&self) -> Result<(), RuntimeError>;
        pub fn clear_all(&self) -> Result<(), RuntimeError>;
        pub fn stats(&self) -> RuntimeStatsSnapshot;
        pub fn shutdown(self) -> Result<(), RuntimeError>;
    }
}
```

### API Behavior

1. `lookup()` is hot-path only and never blocks.
1. `lookup()` reads the in-memory compiled map and returns immediately.
1. On miss, `lookup()` sends a best-effort event to coordinator and returns interpreter fallback.
1. `compile_jit()`, `prepare_aot()`, and `prepare_aot_batch()` enqueue work and return immediately.
1. `code_hash` is treated as trusted input.

## Module Layout

Create module under `crates/revmc/src/runtime/`:

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

1. `api.rs`: public types and handle methods.
1. `config.rs`: defaults and tuning.
1. `coordinator.rs`: single-threaded state machine, queues, generation control.
1. `cache.rs`: concurrent map for published programs.
1. `entry.rs`: hot-path entry (`ready`, `last_used_at`) only.
1. `worker.rs`: compile/load tasks on custom rayon pool.
1. `artifact.rs`: artifact identity and loaded backing lifetime.
1. `storage.rs`: `ArtifactStore` trait and implementations.
1. `stats.rs`: counters/gauges/snapshots.
1. `error.rs`: typed runtime errors.

## Core Architecture

### Hot Path

Hot path does only:

1. Check runtime enabled flag.
1. Build `RuntimeCacheKey = (code_hash, spec_id)`.
1. Probe compiled map.
1. If ready, return compiled.
1. If missing, submit `LookupMiss` command (best effort).
1. Return interpreter fallback.

Hot-path invariants:

1. No waiting.
1. No filesystem or storage access.
1. No linker interaction.
1. No worker pool calls.
1. No blocking global lock.

### Coordinator-Only State

All mutable state transitions are coordinator-owned.

```rust
struct EntryState {
    phase: Phase,
    hotness: u32,
    generation: u64,
    next_retry_at: u64,
    aot_known: bool,
}

enum Phase {
    Cold,
    AotLoadQueued,
    JitQueued,
    AotCompileQueued,
    Working,
    Ready,
    Failed,
}
```

State transitions:

```text
Absent -> Cold

Cold + lookup miss -> maybe AotLoadQueued (if storage enabled and retry window allows)
AotLoadQueued -> Working(load) -> Ready | Cold(AotMissing) | Failed

Cold + lookup miss increments hotness
Cold + hotness >= threshold -> JitQueued -> Working(jit) -> Ready | Failed

Cold + prepare_aot -> AotCompileQueued -> Working(aot_compile) -> Cold(aot_known=true) | Failed

Failed + retry ttl elapsed -> Cold
Ready + eviction -> Cold
Any + clear/reconfigure -> Cold(next generation)
```

Interpretation rule:

1. Only `Ready` returns compiled.
1. Every other phase returns interpreter.

### Unified Fast Lookup Map

Maintain a single in-memory compiled map keyed by `(code_hash, spec_id)` containing both:

1. Loaded AOT programs.
1. JIT-compiled programs.

This map is the only data read on hot lookup.

### Worker Pool

Use dedicated bounded rayon pool for compile/load work.

1. Custom thread count.
1. Custom names (`revmc-compile-{i}`).
1. No use of global rayon pool.
1. Work admission bounded by coordinator queue.

Default: `min(max(1, available_parallelism / 2), 4)`.

## Runtime Configuration

### Top-Level

```rust
pub struct RuntimeConfig {
    pub enabled: bool,
    pub backend: BackendSelection,
    pub storage: StorageConfig,
    pub tuning: RuntimeTuning,
}
```

### Backend

```rust
pub enum BackendSelection {
    Auto,
    Llvm,
    Cranelift,
}
```

Behavior:

1. Prefer LLVM when available.
1. Return `UnsupportedBackend` if requested backend is not built in.
1. Keep Cranelift feature-gated/experimental.

### Storage

```rust
pub enum StorageConfig {
    None,
    Filesystem { root: PathBuf },
    Custom(Arc<dyn ArtifactStore>),
}
```

### Tuning Defaults

| Knob | Default | Why |
|---|---:|---|
| `enabled` | `false` | Safe rollout default |
| `warm_threshold` | `8` | Promote repeated contracts to JIT |
| `max_pending_jobs` | `2048` | Bound memory and background pressure |
| `reserved_direct_slots` | `64` | Keep explicit requests responsive |
| `batch_max_items` | `8` | Keep batches small |
| `batch_max_total_bytecode_bytes` | `256 KiB` | Avoid giant batches |
| `batch_max_wait` | `2 ms` | Micro-batching without noticeable delay |
| `jit_opt_level` | `Default` | Compile-latency/runtime balance |
| `aot_opt_level` | `Aggressive` | Better persisted artifact quality |
| `negative_jit_ttl` | `300 s` | Avoid failing-JIT retry storms |
| `negative_aot_load_ttl` | `60 s` | Avoid repeated missing-artifact probes |
| `enqueue_cooldown` | `1 s` | Avoid queue hammering |
| `resident_code_cache_bytes` | `128 MiB` | Predictable memory bound |
| `worker_count` | `min(max(1, cpus/2), 4)` | Leave CPU headroom |

## Data Model

### Runtime Key

```rust
pub struct RuntimeCacheKey {
    pub code_hash: B256,
    pub spec_id: SpecId,
}
```

### Artifact Key

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

Persisted artifacts are validated by full key, not hash alone.

## Artifact Store

`ArtifactStore` remains full read/write lifecycle API:

```rust
pub trait ArtifactStore: Send + Sync + 'static {
    fn load(&self, key: &ArtifactKey) -> Result<Option<StoredArtifact>, StorageError>;
    fn store(&self, key: &ArtifactKey, artifact: &StoredArtifact) -> Result<(), StorageError>;
    fn delete(&self, key: &ArtifactKey) -> Result<(), StorageError>;
    fn clear(&self) -> Result<(), StorageError>;
}
```

```rust
pub struct StoredArtifact {
    pub manifest: ArtifactManifest,
    pub dylib_bytes: Bytes,
}

pub struct ArtifactManifest {
    pub artifact_key: ArtifactKey,
    pub symbol_name: String,
    pub bytecode_len: usize,
    pub artifact_len: usize,
    pub created_at_unix_secs: u64,
    pub sha256: [u8; 32],
}
```

## Compiled Program Lifetime

`EvmCompilerFn` must not outlive backing allocation.

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

Rules:

1. JIT output is process-local and never persisted.
1. AOT load keeps library handle and temp backing alive behind `Arc`.
1. Eviction drops runtime strong refs; memory frees after active users release `Arc`.

## Coordinator Commands

```rust
enum Command {
    LookupMiss(LookupRequestOwned),
    CompileJit(LookupRequestOwned),
    PrepareAot(Vec<AotRequestOwned>),
    SetEnabled(bool, ReplySender<()>),
    Reconfigure(RuntimeConfigUpdate, ReplySender<()>),
    ClearResident(ReplySender<()>),
    ClearPersisted(ReplySender<()>),
    ClearAll(ReplySender<()>),
    WorkerFinished(WorkerResult),
    Shutdown(ReplySender<()>),
}
```

Priority order:

1. Explicit `PrepareAot` and `CompileJit` requests.
1. AOT load attempts from lookup misses.
1. JIT compile requests from warm lookup misses.

## Pipeline Behavior

### Lookup-Miss Handling

For each miss:

1. Attempt AOT load path first (if storage configured and not in AOT negative TTL).
1. If no usable AOT artifact, continue normal path.
1. Increment hotness and queue JIT once threshold is reached.
1. Return interpreter immediately from hot path.

### AOT Prepare Flow

For each requested contract:

1. Compute `ArtifactKey`.
1. Probe storage.
1. If present and valid, mark `aot_known = true`.
1. If absent, compile AOT and persist through `ArtifactStore::store`.
1. Do not require immediate resident load; lookup can load on demand.

### JIT Flow

1. Queue compile when hotness threshold reached.
1. Compile on worker pool.
1. Publish to unified compiled map.
1. Mark entry `Ready`.

## Clear, Reconfigure, Shutdown

1. `clear_resident()`: clear in-memory compiled map and bump generation.
1. `clear_persisted()`: call `ArtifactStore::clear()` and clear AOT-known markers.
1. `clear_all()`: both operations above.
1. `reconfigure()`: apply config updates, rebuild worker pool if needed, bump generation for destructive changes.
1. `shutdown()`: stop accepting new work, stop coordinator loop, and discard stale/in-flight results by generation.

No compile wait API is exposed.

## Correctness Invariants

1. Interpreter is always the fallback on miss/error paths.
1. Cache key always includes `SpecId`.
1. `code_hash` is treated as trusted input.
1. Coordinator is the only mutator of entry state.
1. Stale worker results from older generations are discarded.
1. Function pointer lifetime is tied to backing owner.

## Eviction And Invalidation

Eviction is coordinator-managed LRU-ish by `last_used_at` and `approx_size_bytes` against `resident_code_cache_bytes`.

When over budget:

1. Evict least-recently-used `Ready` entries.
1. Drop runtime strong references only.
1. Allow memory to free naturally when outstanding `Arc` holders finish.

## Observability

Expose metrics for:

1. Lookup outcomes (`compiled`, `aot_missing`, `interpreted_not_ready`, `interpreted_failed`).
1. Queue depth, enqueue rate, dedupe count, drop count.
1. AOT load attempts/success/failure/latency.
1. AOT prepare compile/store attempts/success/failure/latency.
1. JIT compile attempts/success/failure/latency.
1. Resident bytes and evictions.

Tracing spans around lookup-miss decisions, aot-load jobs, aot-prepare jobs, jit jobs, publish, clear/reconfigure/shutdown.

## Testing Strategy

### 1. Unit

1. Coordinator-only transitions.
1. Dedupe and queue priority.
1. Negative TTL behavior for AOT missing and JIT failures.
1. Generation invalidation.

### 2. Concurrency

1. Many threads same `(code_hash, spec_id)` lookup misses.
1. Ensure single queued work per generation/key.
1. Verify hot path remains non-blocking under saturation.

### 3. Correctness vs Interpreter

1. State tests and block replays compare result/gas/halts.
1. Include suspend/resume across `CALL*` and `CREATE*`.
1. Include multiple `SpecId` coverage.

### 4. Runtime-Created Code

1. `CREATE/CREATE2` constructor/runtime distinction.
1. Repeated factory deployments.
1. JIT warmup behavior for newly created runtime bytecode.

### 5. Failure

1. AOT load miss/corruption.
1. AOT compile/store errors.
1. JIT compile errors.
1. Queue saturation.
1. Clear/reconfigure/shutdown during active work.

### 6. Performance

1. Cold-heavy replay.
1. Hot-heavy replay.
1. Mixed replay.
1. Track fallback rate, queue saturation, compile/load latencies, resident memory.

## Rollout Plan

### Phase 0

1. Coordinator lifecycle.
1. Hot-path lookup + miss event.
1. Unified compiled map.

### Phase 1

1. AOT load-on-miss path.
1. JIT warm-threshold compile path.

### Phase 2

1. Explicit `prepare_aot` and batch flow.
1. AOT compile + persist integration.

### Phase 3

1. Eviction and operational hardening.
1. Reconfigure/clear/shutdown polish.
1. Rollout tuning from metrics.

## Final Recommendation

Implement the smallest coordinator-only design that keeps lookup fast and deterministic: always try resident compiled first, opportunistically use AOT from storage, and otherwise rely on interpreter with background JIT warmup, while supporting explicit AOT preparation for known contracts.

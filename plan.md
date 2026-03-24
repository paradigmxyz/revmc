# revmc Runtime JIT Coordinator Plan

## Status

Proposed.

## Summary

Add a new `revmc::runtime` module with a coordinator-owned design and a single O(1) lookup path.

Startup behavior:

1. Call `ArtifactStore::load_all()`.
1. Load all returned `(ArtifactKey, StoredArtifact)` entries into in-memory compiled map.
1. Start serving lookups.

Runtime behavior:

1. `lookup()` checks only in-memory compiled map (`AOT` or `JIT`).
1. `lookup()` never touches storage and never waits.
1. `lookup()` sends fire-and-forget tracking event to coordinator.
1. Coordinator tracks hotness and may enqueue background JIT compile.

JIT outputs are process-local and never persisted.

## Scope

### In Scope

1. Startup AOT preload from `ArtifactStore::load_all()` into runtime compiled map.
1. O(1) lookup path using only in-memory map.
1. Fire-and-forget lookup tracking to coordinator.
1. Warm-threshold runtime JIT compilation.
1. Explicit AOT preparation API for known contract lists (compile+store if missing).
1. Coordinator-owned state transitions, scheduling, invalidation, and eviction.

### Out Of Scope

1. Storage access on lookup miss.
1. Hot-path waiting for compile/load completion.
1. Multi-tier speculative optimization and deopt.
1. Distributed artifact caches and portability guarantees.

## Goals

### Functional

1. Reusable runtime for reth and other revm/revmc embedders.
1. Correct execution under `SpecId`-sensitive semantics.
1. Correct fallback and suspend/resume compatibility with existing interpreter path.
1. Support runtime-created code (`CREATE`/`CREATE2`) as normal JIT candidates.

### Operational

1. Non-blocking execution hot path.
1. Bounded background work and resident memory.
1. Deterministic coordinator-only control flow.
1. Strong observability for rollout/tuning.

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

        // Explicit enqueue APIs (non-waiting).
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

1. `lookup()` never blocks.
1. `lookup()` only reads in-memory compiled map.
1. Every `lookup()` emits best-effort tracking event to coordinator (hit or miss).
1. `compile_jit()` and `prepare_aot*()` are enqueue-only and return immediately.
1. `code_hash` is trusted input.

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
1. `cache.rs`: concurrent in-memory compiled map.
1. `entry.rs`: hot-path data (`ready`, `last_used_at`) only.
1. `worker.rs`: background JIT and AOT-prepare compile tasks.
1. `artifact.rs`: artifact identity and loaded lifetime owners.
1. `storage.rs`: `ArtifactStore` and backends.
1. `stats.rs`: counters/gauges/snapshots.
1. `error.rs`: runtime error types.

## Core Architecture

### Startup Bootstrap

Before serving lookups:

1. Build coordinator.
1. Call `ArtifactStore::load_all()`.
1. Validate and publish each returned artifact into compiled map as `ProgramKind::Aot`.
1. Start coordinator event loop and return handle.

Startup may block; hot path must not.

### Hot Path

For each `lookup(req)`:

1. Check enabled flag.
1. Compute key `(code_hash, spec_id)`.
1. Probe compiled map.
1. Return `Compiled` on hit or `Interpret` on miss.
1. Emit non-blocking `LookupObserved` command to coordinator.

Hot-path invariants:

1. No waiting.
1. No storage or filesystem calls.
1. No linker calls.
1. No worker pool calls.
1. No blocking global mutex.

### Coordinator-Only State

All mutable state is written only by coordinator thread.

```rust
struct EntryState {
    phase: Phase,
    hotness: u32,
    generation: u64,
    next_retry_at: u64,
    source: Source,
}

enum Source {
    Aot,
    Jit,
    Unknown,
}

enum Phase {
    Cold,
    JitQueued,
    AotPrepareQueued,
    Working,
    Ready,
    Failed,
}
```

State transitions:

```text
Startup-loaded AOT -> Ready(source=Aot)
Absent -> Cold(source=Unknown)

Cold + observed lookups + hotness<threshold -> Cold
Cold + observed lookups + hotness>=threshold -> JitQueued -> Working(jit) -> Ready(source=Jit) | Failed

Cold + prepare_aot -> AotPrepareQueued -> Working(aot_compile) -> Cold(source=Aot) | Failed

Failed + retry ttl elapsed -> Cold
Ready + eviction -> Cold
Any + clear/reconfigure -> Cold(next generation)
```

Only `Ready` returns compiled.

### Unified In-Memory Compiled Map

Single map keyed by `(code_hash, spec_id)` containing both AOT and JIT compiled programs.

At startup, map contains only AOT entries. JIT entries are added later by coordinator.

### Worker Pool

Dedicated bounded rayon pool:

1. Custom thread count.
1. Thread names `revmc-compile-{i}`.
1. No global rayon pool usage.
1. Admission bounded by coordinator queue.

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
| `max_pending_jobs` | `2048` | Bound memory/background pressure |
| `reserved_direct_slots` | `64` | Keep explicit requests responsive |
| `batch_max_items` | `8` | Keep batches small |
| `batch_max_total_bytecode_bytes` | `256 KiB` | Avoid giant batches |
| `batch_max_wait` | `2 ms` | Micro-batching without visible delay |
| `jit_opt_level` | `Default` | Compile-latency/runtime balance |
| `aot_opt_level` | `Aggressive` | Better persisted artifact quality |
| `negative_jit_ttl` | `300 s` | Avoid failing-JIT retry storms |
| `enqueue_cooldown` | `1 s` | Avoid queue hammering |
| `jit_max_bytecode_len` | `0` (no limit) | Skip oversized contracts |
| `resident_code_cache_bytes` | `0` (no limit) | Predictable memory bound |
| `idle_evict_duration` | `None` (disabled) | Downgrade stale entries after hardforks |
| `eviction_sweep_interval` | `60 s` | Periodic eviction sweep cadence |
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

Persisted artifacts must match full key.

## Artifact Store

`ArtifactStore` API remains:

```rust
pub trait ArtifactStore: Send + Sync + 'static {
    fn load_all(&self) -> Result<Vec<(ArtifactKey, StoredArtifact)>, StorageError>;
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

1. Startup AOT and loaded libraries stay valid via shared backing owners.
1. JIT output is process-local and never persisted.
1. Eviction drops runtime strong refs only; memory frees after active `Arc` holders release.

## Coordinator Commands

```rust
enum Command {
    LookupObserved(LookupObservedEvent),
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
1. `LookupObserved` processing for hotness/JIT admission.

## Pipeline Behavior

### Lookup Tracking

Every lookup emits `LookupObserved { key, was_hit }`.

Coordinator behavior:

1. Update counters/hotness.
1. If key already `Ready`, do nothing further.
1. If key is cold and hotness reaches threshold, enqueue JIT compile.

### AOT Prepare

For each requested contract:

1. Build `ArtifactKey`.
1. Probe storage.
1. If missing, compile AOT and persist via `store()`.
1. Optionally load into resident map immediately (policy knob), otherwise available after next restart/bootstrap.

### JIT Compile

1. Enqueue when hotness threshold crossed.
1. Compile on worker pool.
1. Publish into in-memory map as `ProgramKind::Jit`.

## Clear, Reconfigure, Shutdown

1. `clear_resident()`: clear in-memory compiled map and bump generation.
1. `clear_persisted()`: call `ArtifactStore::clear()` and clear any in-memory AOT catalog metadata.
1. `clear_all()`: both operations above.
1. `reconfigure()`: apply updates, rebuild worker pool if needed, bump generation for destructive changes.
1. `shutdown()`: stop accepting events and ignore stale/in-flight worker results.

No compile wait API is exposed.

## Correctness Invariants

1. Interpreter is always fallback on non-ready/error path.
1. Key includes `SpecId`.
1. `code_hash` is trusted input.
1. Coordinator is sole state mutator.
1. Stale generation results are discarded.
1. Function pointer lifetime is tied to backing owner.

## Eviction And Downgrading

Coordinator-managed eviction via periodic sweeps (every `eviction_sweep_interval`).

Two eviction modes (both optional, composable):

1. **Idle eviction** (`idle_evict_duration`): entries with no lookup hits for longer than this duration are evicted. This naturally handles hardfork transitions — when a new `SpecId` activates, contracts compiled for old spec IDs stop being looked up and are cleaned up after the idle timeout.
1. **Memory budget** (`resident_code_cache_bytes`): when total `approx_size_bytes` exceeds the budget, least-recently-used entries are evicted until under budget.

When evicting:

1. Remove from resident map and metadata.
1. Reset coordinator entry state (key can be re-promoted if it becomes hot again).
1. Drop runtime strong refs.
1. Let memory free naturally as outstanding `Arc` refs drop.

## Observability

Expose metrics for:

1. Lookup outcomes (`compiled`, `interpreted_not_ready`, `interpreted_failed`).
1. Lookup-observed enqueue success/drop counts.
1. Hotness promotions to JIT queue.
1. JIT compile attempts/success/failure/latency.
1. AOT prepare probe/compile/store success/failure/latency.
1. Startup `load_all` count/failure/latency.
1. Resident entries, bytes, and evictions.

Add tracing spans around startup preload, lookup-observed handling, JIT jobs, AOT prepare jobs, and lifecycle operations.

## Testing Strategy

### Unit

1. Coordinator-only state transitions.
1. Hotness tracking from `LookupObserved`.
1. JIT admission threshold behavior.
1. Generation invalidation.

### Concurrency

1. Many threads calling `lookup()` on same key.
1. Ensure lookup remains non-blocking under saturated coordinator channel.
1. Ensure single JIT work admission per key/generation.

### Correctness

1. Compare against interpreter on state tests/block replays.
1. Cover `CALL*` and `CREATE*` suspend/resume behavior.
1. Cover multiple `SpecId`s.

### Runtime-Created Code

1. `CREATE/CREATE2` runtime bytecode warmup and JIT admission.
1. Repeated factory deployments.

### Failure

1. Startup preload partial failures.
1. AOT prepare compile/store errors.
1. JIT compile errors.
1. Queue saturation/drop behavior.
1. Clear/reconfigure/shutdown during active work.

### Performance

1. Cold-heavy replay.
1. Hot-heavy replay.
1. Mixed replay.
1. Measure lookup cost, queue pressure, JIT latency, and resident memory.

## Rollout Plan

### Phase 0

1. Startup AOT preload + in-memory map.
1. O(1) lookup + fire-and-forget observed event.

### Phase 1

1. Coordinator hotness tracking.
1. Threshold-based JIT background compile.

### Phase 2

1. Explicit `prepare_aot` APIs.
1. Compile+persist flow through `ArtifactStore`.

### Phase 3

1. Eviction, lifecycle hardening, and rollout tuning.

## Final Recommendation

Keep lookup minimal: probe resident map and return immediately. Move all policy into coordinator by consuming fire-and-forget lookup events, preload AOT at startup via `load_all`, and add JIT only as background promotion for observed hot keys.

# Out-of-process JIT exploration

`RuntimeConfig::jit_mode` now has an `OutOfProcess` variant, disabled by default. This document records the current prototype and the remaining work needed to make it production-ready.

## Recommended architecture

Keep execution and ORC linking in the parent process. Move only translation, optimization, and object emission to a helper process that owns the background worker pool.

The helper process receives `CompileJob`s over IPC and returns relocatable object bytes plus timings/errors. The parent process then adds the object to its existing ORC `LLJIT`, resolves the symbol to a local `EvmCompilerFn`, and owns the `ResourceTracker`/`JitDylibGuard` exactly as it does today. This preserves the current runtime API and keeps compiled function pointers valid in the caller process.

Using LLVM ORC's remote executor APIs (`ExecutorProcessControl`, `SimpleRemoteEPC`, etc.) would put the executable code in the child process. That is not transparent for the current `EvmCompilerFn` API, because callers need a local function pointer and direct calls into `revm` state.

## LLVM ORC support needed

- Add C/Rust bindings to add an already-compiled object buffer to a specific `JITDylib` with a `ResourceTracker`:
  - `LLJIT::add_object_file_with_rt` via `LLVMOrcLLJITAddObjectFileWithRT` if available, or a small C++ wrapper around `ObjectLayer::add` / `object::ObjectFile` materialization.
  - Continue to perform `lookup_in(jd, symbol)` in the parent after linking.
- Keep builtin absolute symbols, process symbol generators, perf/debug plugins, and memory accounting in the parent ORC instance. The child only emits relocatable objects with unresolved external symbols.
- Preserve per-entry eviction by creating the `ResourceTracker` in the parent before adding the object and returning it through `JitCodeBacking`.
- Replace the current object capture TLS path for out-of-process jobs: object bytes are already returned by the helper, so disassembly/debug dumping should consume those bytes directly.

## Runtime/IPC work

Current prototype:

- `RuntimeConfig::jit_mode = JitMode::OutOfProcess` makes the runtime keep a global persistent helper process spawned via `RuntimeConfig::jit_helper_path`, or `std::env::current_exe()` when unset.
- `RuntimeConfig::default()` uses plain in-process defaults. `JitBackend::new` applies `RuntimeConfig::with_env_overrides()`, which recognizes `REVMC_JIT_MODE=out-of-process` and `REVMC_JIT_MODE=in-process`; other spellings are rejected. `REVMC_JIT_HELPER_PATH` overrides the helper executable path. Test harnesses should point this at a binary that calls `revmc::runtime::maybe_run_jit_helper()` at startup, such as `target/debug/revmc`.
- `REVMC_JIT_HELPER_MEMORY_LIMIT_BYTES` and `REVMC_JIT_HELPER_CPU_SECONDS` apply Unix `RLIMIT_AS` and `RLIMIT_CPU` limits to helper processes before `exec`.
- Helper binaries must call `revmc::runtime::maybe_run_jit_helper()` at process startup. `revmc-cli` does this already.
- Workers send length-prefixed wincode-serialized JIT object requests to the helper over stdin and receive length-prefixed wincode-serialized responses from stdout.
- The parent links returned object bytes into its local ORC instance, resolves the symbol, and constructs `JitCodeBacking` with a parent-owned `ResourceTracker`.
- `RuntimeTuning::jit_timeout` bounds each helper compilation; timed-out helpers are killed and replaced on the next job.
- Clearing resident code or shutting down the runtime kills the helper process so in-flight out-of-process compiles can be interrupted instead of waiting for LLVM to finish.

## Fork-only helper startup

Using `fork()` without `exec` is not safe with the current lazy helper model. The helper is spawned from a runtime worker after the backend has started threads, and a child forked from a multithreaded process can only safely run async-signal-safe operations until `exec`. The helper would immediately need to run normal Rust code, allocate, use locks, deserialize IPC messages, and initialize/run LLVM, so it does not fit that rule.

Avoiding LLVM translation in the parent is not sufficient. The parent still uses LLVM ORC to link returned object files, and the Rust runtime, allocator, tracing, channels, and other libraries can have process-global locks or thread-local state before the helper is spawned.

A fork-only helper may be viable only as an early fork server: fork during single-threaded startup before any LLVM initialization or backend worker creation, then let the child own all helper-side Rust/LLVM state. That would require an explicit startup path instead of the current lazy spawn.

Still needed:

- Move the worker pool into a single helper process; the parent should only enqueue IPC requests.
- Carry the remaining data in the IPC payloads: gas params, dump settings, generation, timings, and richer errors.
- Keep AOT jobs either in the helper too or explicitly route them through the existing in-process AOT path; the first option gives consistent isolation.
- Define graceful shutdown semantics for queued helper jobs; the current implementation kills the helper for cancellation/shutdown.
- Treat helper crash as worker-pool failure: fail pending synchronous jobs, drop pending async jobs, and optionally respawn.

## Open questions

- Stability requirements for the private IPC protocol.
- Whether debug dumps should be written by the child, the parent, or both.
- Whether compiler recycling is still needed per helper worker once the whole helper can be restarted.
- How to expose helper process configuration such as executable path, environment, and restart policy without making the default API noisy.

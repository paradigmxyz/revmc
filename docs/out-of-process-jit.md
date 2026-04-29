# Out-of-process JIT exploration

`RuntimeConfig::jit_process_mode` now has an `OutOfProcess` variant, disabled by default. It currently returns an error if selected; this document records the implementation work needed to make it transparent.

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

- Add a helper-process entrypoint, preferably the same binary invoked with a hidden argument or environment variable.
- Spawn one helper process from the backend thread when `jit_process_mode == OutOfProcess`.
- The helper owns the existing Rayon worker pool and thread-local `EvmCompiler` instances.
- Define a framed IPC protocol for `CompileJob` and `WorkerResult` data: key, bytecode, symbol name, spec id, optimization level, gas params, debug flags, dedup/DSE flags, dump settings, generation, timings, object bytes, and errors.
- In the parent, turn a successful JIT worker result into a resident program by linking object bytes into ORC, looking up the symbol, and constructing `JitCodeBacking`.
- Keep AOT jobs either in the helper too or explicitly route them through the existing in-process AOT path; the first option gives consistent isolation.
- Define shutdown semantics: close IPC, let the helper drain or cancel queued jobs, then kill on timeout.
- Treat helper crash as worker-pool failure: fail pending synchronous jobs, drop pending async jobs, and optionally respawn.

## Open questions

- Serialization crate and stability requirements for the private IPC protocol.
- Whether debug dumps should be written by the child, the parent, or both.
- Whether compiler recycling is still needed per helper worker once the whole helper can be restarted.
- How to expose helper process configuration such as executable path, environment, and restart policy without making the default API noisy.

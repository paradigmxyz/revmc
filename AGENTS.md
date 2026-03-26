# revmc - EVM JIT/AOT compiler

## Commands
- Lint: `cargo +nightly cl`
- Format: `cargo +nightly fmt --all`
- Check docs: `cargo +nightly docs`
- Test all: `cargo nextest run --workspace`
- Test single: `cargo nextest run -p revmc 'test_name'`

## Architecture
- `revmc` — main crate: EVM compiler, bytecode analysis, linker, and test infrastructure.
- `revmc-backend` — abstract compiler backend trait. `revmc-cranelift` and `revmc-llvm` are implementations.
- `revmc-builtins` — runtime builtins called by JIT-compiled code (host calls, gas accounting).
- `revmc-context` — EVM execution context types bridging revm and compiled code.
- `revmc-build` — build-script helpers for AOT compilation.
- `revmc-cli` / `revmc-cli-tests` — CLI frontend and its integration tests.
- `revmc-statetest` — Ethereum state test runner.

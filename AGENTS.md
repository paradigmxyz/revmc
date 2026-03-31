# revmc - EVM JIT/AOT compiler

## Commands

```bash
cargo cl                                   # lint
cargo fmt --all                            # format
cargo docs                                 # check docs

cargo nextest run --workspace              # test all
cargo nextest run -p revmc "test_name"     # test single
cargo nextest run -p revmc "statetest"     # test statetests
SUBDIR=stRevertTest cargo nextest run -p revmc "statetest" # test single statetest

```

## Architecture
- `revmc` — main crate: EVM compiler, bytecode analysis, linker, and test infrastructure.
- `revmc-backend` — abstract compiler backend trait. `revmc-llvm` is the main implementation.
- `revmc-builtins` — runtime builtins called by JIT-compiled code (host calls, gas accounting).
- `revmc-context` — EVM execution context types bridging revm and compiled code.
- `revmc-build` — build-script helpers for AOT compilation.
- `revmc-cli` / `revmc-cli-tests` — CLI frontend and its integration tests.
- `revmc-statetest` — Ethereum state test runner.

## CLI

Do NOT use `--release` — dev profile already uses `opt-level = 3`, and release
strips debug info and uses LTO which makes builds much slower for no benefit
during development.

```bash
cargo r -- run --list                      # list available benchmarks
cargo r -- run usdc_proxy                  # compile and run a benchmark
cargo r -- run usdc_proxy --parse-only     # parse and analyze only (no codegen)
cargo r -- run usdc_proxy --display        # print parsed bytecode IR
cargo r -- run usdc_proxy --dot            # render CFG as DOT/SVG
cargo r -- run usdc_proxy --aot            # compile to shared library
cargo r -- run custom --code 6001600201    # run custom bytecode
```

Use `RUST_LOG` to control log output:

```bash
RUST_LOG=debug cargo r -- run usdc_proxy   # all debug logs
RUST_LOG=revmc=debug cargo r -- ...        # only revmc crate logs
RUST_LOG=revmc::bytecode=trace cargo r --  # trace a specific module
```

## Checking dynamic jump resolution

To see how many jumps are resolved vs unresolved on a real contract:

```bash
RUST_LOG=debug cargo r -- run usdc_proxy --display |& rg 'jump|JUMP'
```

- `resolved jumps newly_resolved=N` — jumps resolved by block analysis.
- `unresolved dynamic jumps remain n=N` — jumps that couldn't be resolved.
- `JUMP bb<N>` / `JUMP bb<N>, bb<M>` — resolved (single/multi-target).
- `JUMP               ; pc=<N>` (no `bb` target) — unresolved dynamic jump.

To get a summary across all benchmarks:

```bash
./scripts/jump-resolution.sh              # all benchmarks
./scripts/jump-resolution.sh usdc_proxy weth  # specific benchmarks
```

Use `cargo r -- run --list` to see available benchmark names.

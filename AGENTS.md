# revmc - EVM JIT/AOT compiler

## Commands

```bash
cargo cl                                   # lint
cargo fmt --all                            # format
cargo docs                                 # check docs

cargo nextest run --workspace              # test all
cargo nextest run "test_name"              # test single
cargo nextest run "statetest"              # test statetests
SUBDIR=stRevertTest cargo nextest run "statetest" # test single statetest
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
cargo r -- run usdc_proxy -o tmp/dump      # compile and run a benchmark; dump files like opt.ll, remarks.txt to tmp/dump
cargo r -- run usdc_proxy --parse-only     # parse and analyze only (no codegen)
cargo r -- run usdc_proxy --display        # print parsed bytecode IR
cargo r -- run usdc_proxy --dot            # render CFG as DOT/SVG
cargo r -- run usdc_proxy --aot            # compile to shared library
cargo r -- run custom --code 6001600201    # run custom bytecode (hex)
cargo r -- run custom --code 'PUSH1 1 PUSH1 2 ADD' # run custom bytecode (asm string)
```

Use `RUST_LOG` to control log output:

```bash
RUST_LOG=debug cargo r -- run usdc_proxy   # all debug logs
RUST_LOG=revmc=debug cargo r -- ...        # only revmc crate logs
RUST_LOG=revmc::bytecode=trace cargo r --  # trace a specific module
```

## Injecting LLVM args

Extra LLVM command-line arguments can be passed via the `REVMC_LLVM_ARGS`
environment variable (space-separated):

```bash
REVMC_LLVM_ARGS="-debug-only=isel" cargo r -- run usdc_proxy
REVMC_LLVM_ARGS="-print-after-all" cargo r -- run usdc_proxy
```

LLVM args are a one-shot global (`LLVMParseCommandLineOptions`); only the first
call takes effect.

## Checking dynamic jump resolution

To get jump resolution stats across benchmarks:

```bash
./scripts/bench.py /tmp/bench --jump-resolution                    # all benchmarks
./scripts/bench.py /tmp/bench --jump-resolution usdc_proxy weth    # specific benchmarks
```

To inspect a single contract in detail:

```bash
RUST_LOG=debug cargo r -- run usdc_proxy --display |& rg 'jump|JUMP'
```

- `resolved jumps newly_resolved=N` — jumps resolved by block analysis.
- `unresolved dynamic jumps remain n=N` — jumps that couldn't be resolved.
- `JUMP bb<N>` / `JUMP bb<N>, bb<M>` — resolved (single/multi-target).
- `JUMP               ; pc=<N>` (no `bb` target) — unresolved dynamic jump.

Use `cargo r -- run --list` to see available benchmark names.

## Benchmarking against another revision

`./scripts/bench.py` is the unified benchmarking tool. It collects codegen line
counts, compile times, jump resolution stats, and constant-input statistics.

```bash
./scripts/bench.py /tmp/bench --diff main                          # codegen + compile time vs main
./scripts/bench.py /tmp/bench --diff main usdc_proxy seaport       # specific benchmarks
./scripts/bench.py /tmp/bench --diff main --extra-dir tmp/mainnet  # include mainnet .bin files
./scripts/bench.py /tmp/bench                                      # current branch only (no diff)
./scripts/bench.py /tmp/bench --diff main --compile-times          # compile times only
./scripts/bench.py /tmp/bench --diff main --codegen-lines          # codegen lines only
./scripts/bench.py /tmp/bench --jump-resolution                    # jump resolution stats
./scripts/bench.py /tmp/bench --input-stats                        # constant-input stats
./scripts/bench.py /tmp/bench --codegen-lines --jump-resolution    # combine multiple analyses
```

## Important

- NEVER delete or modify `./tmp/` — it contains manually generated IR/asm dumps used for comparison.
- `tmp/dump/` contains dumps from `main`, `tmp/dump2/` contains dumps from the current branch.
  Use these for manual `diff` comparison of LLVM IR and assembly.

## Code style

- Never call `.index()` on an index type just to reconstruct the same type.
  Use arithmetic on the index directly:
  ```rust
  // BAD
  let prev = &self.insts[Inst::from_usize(term_inst.index() - 1)];
  // GOOD
  let prev = &self.insts[term_inst - 1];
  ```
- Don't prefix log messages with the pass/function name — `#[instrument]` spans
  already provide that context. Just describe what happened:
  ```rust
  // BAD
  trace!("local: resolved jump");
  // GOOD
  trace!("resolved jump");
  ```

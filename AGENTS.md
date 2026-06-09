# revmc - EVM JIT/AOT compiler

## Commands

`--all-features` does not work. Just use default features and the following commands:

```bash
cargo cl                                                      # lint
cargo fmt --all                                               # format
cargo docs                                                    # check docs

cargo nextest run --workspace                                 # test all
cargo nextest run --workspace "test_name"                     # test single
cargo nextest run --workspace "statetest"                     # test statetests
SUBDIR=stRevertTest cargo nextest run --workspace "statetest" # test single statetest
```

## Architecture

- `revmc` — thin umbrella crate that re-exports codegen and runtime APIs.
- `revmc-codegen` — EVM compiler, bytecode analysis, linker, and compiler test infrastructure.
- `revmc-runtime` — runtime JIT/AOT backend, worker pool, artifact store, and revm integration.
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
cargo r -- run --list                  # list available benchmarks
cargo r -- run usdc_proxy              # compile and run a benchmark
cargo r -- run usdc_proxy -o tmp/dump  # compile and run a benchmark; dump files like opt.ll, remarks.txt to tmp/dump
cargo r -- run usdc_proxy --parse-only # parse and analyze only (no codegen)
cargo r -- run usdc_proxy --display    # print parsed bytecode IR
cargo r -- run usdc_proxy --dot        # render CFG as DOT/SVG
cargo r -- run usdc_proxy --aot        # compile to shared library
cargo r -- run 0x6001600201            # run custom bytecode (hex)
cargo r -- run 'PUSH1 1 PUSH1 2 ADD'   # run custom bytecode (asm string)
```

`-o <dir>` writes dumps under `<dir>/<benchmark>/`. Common files:

- `bytecode.bin` — raw input bytecode.
- `bytecode.txt` — parsed bytecode IR with blocks, gas, stack info, and comments.
- `bytecode.dbg.txt` — verbose debug dump of the parsed bytecode structure.
- `bytecode.dot` / `bytecode.svg` — rendered CFG.
- `unopt.ll` — LLVM IR before optimization.
- `opt.ll` — optimized LLVM IR.
- `opt.s` — final optimized assembly.
- `remarks.txt` — compile timings, JIT size, and generated-file sizes.

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

The script writes its full markdown output to `<dump_dir>/results.md` in
addition to printing it to stdout. Summary tables hide changes within a noise
threshold (1% for codegen, 5% for compile times); the `<details>` tables still
show every change.

```bash
./scripts/bench.py /tmp/bench --diff main                          # codegen + compile time vs main
./scripts/bench.py /tmp/bench --diff main usdc_proxy seaport       # specific benchmarks
./scripts/bench.py /tmp/bench --diff main --extra-dir tmp/mainnet  # include mainnet .bin files
./scripts/bench.py /tmp/bench                                      # current branch only (no diff)
./scripts/bench.py /tmp/bench --diff main --compile-times          # compile times only
./scripts/bench.py /tmp/bench --diff main --codegen-lines          # codegen lines only
./scripts/bench.py /tmp/bench --jump-resolution                    # jump resolution stats
./scripts/bench.py /tmp/bench --input-stats                        # constant-input stats
./scripts/bench.py /tmp/bench --block-stats                        # block stats (min/max/avg/median, suspends)
./scripts/bench.py /tmp/bench --codegen-lines --jump-resolution    # combine multiple analyses
```

## Bench-and-PR workflow

When the user asks to "bench and open pr", "post results to pr", or whenever
making a perf change that needs benchmark numbers in the PR description:

1. Run `./scripts/bench.py <dump_dir> --diff <base>` (typically `--diff main`).
2. Build the PR body **in a single bash command** that inlines
   `<dump_dir>/results.md` VERBATIM. Do NOT reformat, summarize, drop
   columns, or rewrite the numbers in the tables — `cat` the file as-is.
3. Add prose explaining what the PR does ABOVE the inlined results, under a
   `## Benchmarks` (or similar) heading.
4. Under `## Benchmarks`, ABOVE the inlined `results.md`, write a short
   textual summary of the headline numbers (e.g. the `**TOTAL**` row diffs
   from the codegen + compile-time tables, plus any notable per-bench wins
   or regressions worth calling out). Keep it to a few sentences or a tight
   bullet list — this is the at-a-glance summary readers see before the
   tables. The tables themselves stay verbatim.
5. Update the PR with `gh pr edit <number> --body-file <body.md>`.

Example — write the body file with prose, summary, and verbatim results in
one shot:

```bash
{
  cat <<'EOF'
Short description of what this PR does and why.

More prose: motivation, design notes, caveats, anything reviewers need.

## Benchmarks

Headline numbers vs `main`: jit size -7.5%, opt.s +2.9%, total compile time
+0.2%. `counter` regresses on opt.s (+26%); `seaport` is roughly flat.

EOF
  cat /tmp/bench/results.md
} > /tmp/pr-body.md

gh pr edit 123 --body-file /tmp/pr-body.md
```

The heredoc holds whatever prose + summary belongs in the PR; `cat results.md`
appends the benchmark tables exactly as the script produced them.

## Important

- NEVER alter or summarize the benchmark tables themselves — always post them
  verbatim. A short textual summary of the headline numbers ABOVE the tables
  (under `## Benchmarks`) is required.
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

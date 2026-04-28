#![allow(missing_docs, unexpected_cfgs)]

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use revm_handler::ExecuteEvm;
use revmc::{EvmCompiler, EvmLlvmBackend, OptimizationLevel, primitives::hardfork::SpecId};
use revmc_cli::PreparedBench;
use std::time::Duration;

const SPEC_ID: SpecId = SpecId::OSAKA;

/// Benchmarks whose LLVM JIT compilation is too slow for CI under valgrind.
/// The `translate` (frontend) benchmark is still included.
const SKIP_JIT: &[&str] = &[
    "snailtracer",
    "seaport",
    "fiat_token",
    "uniswap_v2_pair",
    "univ2_router",
    "airdrop",
    "usdc_proxy",
];
/// Benchmarks that are too slow for CI entirely (runtime is also very slow under valgrind).
const SKIP_ALL: &[&str] = &["seaport", "snailtracer"];

fn bench(c: &mut Criterion) {
    // Single compiler shared across all benchmarks. JIT'd code lives in its
    // memory, so it must outlive every `PreparedBench`. `clear_ir()` is called
    // between benchmarks to free IR while keeping machine code alive.
    let mut compiler = EvmCompiler::new_llvm(false).unwrap();
    for bench in &revmc_cli::get_benches() {
        if SKIP_ALL.contains(&bench.name) {
            continue;
        }
        run_bench(c, bench, &mut compiler);
    }
}

fn run_bench(
    c: &mut Criterion,
    def: &revmc_cli::Bench,
    compiler: &mut EvmCompiler<EvmLlvmBackend>,
) {
    let name = def.name;
    let is_fixture = def.is_fixture();

    let prepared = PreparedBench::load_with(def, SPEC_ID, compiler);
    if cfg!(any(debug_assertions, not(codspeed))) {
        prepared.sanity_check();
    }

    let mut g = c.benchmark_group(name);
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(1));
    g.measurement_time(Duration::from_secs(5));

    // ── Bytecode-only benchmarks ────────────────────────────────────────

    if !is_fixture {
        // Compile-time.
        g.bench_function(format!("{name}/compile/translate"), |b| {
            b.iter_batched_ref(
                || new_compiler(OptimizationLevel::Default),
                |compiler| {
                    compiler.translate(name, &def.bytecode, SPEC_ID).unwrap();
                },
                BatchSize::PerIteration,
            )
        });

        if !SKIP_JIT.contains(&name) {
            g.bench_function(format!("{name}/compile/jit"), |b| {
                b.iter_batched_ref(
                    || {
                        let mut compiler = new_compiler(OptimizationLevel::default());
                        let id =
                            compiler.translate(name, &def.bytecode, SPEC_ID).expect("translate");
                        (compiler, id)
                    },
                    |(compiler, id)| unsafe {
                        compiler.jit_function(*id).unwrap();
                    },
                    BatchSize::PerIteration,
                )
            });
        }
    }

    // ── Unified runtime benchmarks ──────────────────────────────────────

    let tx = prepared.tx().clone();

    let mut interp_evm = prepared.new_interpreter_evm();
    g.bench_function(format!("{name}/rt/interpreter"), |b| {
        b.iter_batched(
            || tx.clone(),
            |tx| interp_evm.transact_one(tx).unwrap(),
            BatchSize::SmallInput,
        );
    });

    let mut jit_evm = prepared.new_jit_evm();
    g.bench_function(format!("{name}/rt/jit"), |b| {
        b.iter_batched(
            || tx.clone(),
            |tx| jit_evm.transact_one(tx).unwrap(),
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

fn new_compiler(opt_level: OptimizationLevel) -> EvmCompiler<EvmLlvmBackend> {
    let mut compiler = EvmCompiler::new_llvm(false).unwrap();
    compiler.set_opt_level(opt_level);
    compiler
}

criterion_group!(benches, bench);
criterion_main!(benches);

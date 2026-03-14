#![allow(missing_docs)]
//! Compile-time benchmarks: measures the cost of parsing, translating, and JIT-compiling
//! EVM bytecode.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use revmc::{EvmCompiler, EvmLlvmBackend, OptimizationLevel, primitives::hardfork::SpecId};
use revmc_cli::Bench;
use std::time::Duration;

const SPEC_ID: SpecId = SpecId::OSAKA;

fn bench_compile(c: &mut Criterion) {
    for bench in &revmc_cli::get_benches() {
        run_bench(c, bench);
    }
}

fn run_bench(c: &mut Criterion, bench: &Bench) {
    let name = bench.name;
    let bytecode = &bench.bytecode;

    let mut g = c.benchmark_group(format!("compile/{name}"));
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(1));
    g.measurement_time(Duration::from_secs(5));

    g.bench_function("translate", |b| {
        b.iter_batched(
            || new_compiler(OptimizationLevel::None),
            |mut compiler| {
                compiler.translate(name, bytecode.as_slice(), SPEC_ID).unwrap();
            },
            BatchSize::PerIteration,
        )
    });

    g.bench_function("jit", |b| {
        b.iter_batched(
            || new_compiler(OptimizationLevel::Aggressive),
            |mut compiler| unsafe {
                compiler.jit(name, bytecode.as_slice(), SPEC_ID).unwrap();
            },
            BatchSize::PerIteration,
        )
    });

    g.finish();
}

fn new_compiler(opt_level: OptimizationLevel) -> EvmCompiler<EvmLlvmBackend> {
    let backend = EvmLlvmBackend::new(false, opt_level).unwrap();
    EvmCompiler::new(backend)
}

criterion_group!(benches, bench_compile);
criterion_main!(benches);

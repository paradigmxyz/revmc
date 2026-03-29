#![allow(missing_docs)]

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use revm_bytecode::Bytecode;
use revm_interpreter::{
    InputsImpl, SharedMemory, instruction_table,
    interpreter::{EthInterpreter, ExtBytecode},
};
use revmc::{
    EvmCompiler, EvmContext, EvmLlvmBackend, EvmStack, OptimizationLevel,
    primitives::hardfork::SpecId,
};
use revmc_cli::{BenchHost, PreparedFixtureBench};
use std::time::Duration;

const SPEC_ID: SpecId = SpecId::OSAKA;

/// Benchmarks that are too slow for CI due to large bytecode (LLVM compilation time under
/// valgrind).
const SKIP_COMPILE: &[&str] =
    &["snailtracer", "seaport", "fiat_token", "uniswap_v2_pair", "airdrop", "usdc_proxy"];
/// Benchmarks that are too slow for CI entirely (runtime is also very slow under valgrind).
const SKIP_ALL: &[&str] = &["seaport", "snailtracer"];

fn bench(c: &mut Criterion) {
    for bench in &revmc_cli::get_benches() {
        if SKIP_ALL.contains(&bench.name) {
            continue;
        }
        if bench.is_fixture() {
            run_fixture_bench(c, bench);
        } else {
            run_bytecode_bench(c, bench);
        }
    }
}

fn run_fixture_bench(c: &mut Criterion, bench: &revmc_cli::Bench) {
    let name = bench.name;
    let prepared = PreparedFixtureBench::load(bench);

    // Sanity check.
    assert!(prepared.run_interpreter().result.is_success(), "interpreter execution reverted");
    assert!(prepared.run_jit().result.is_success(), "JIT execution reverted");

    let mut g = c.benchmark_group(name);
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(1));
    g.measurement_time(Duration::from_secs(5));

    g.bench_function(format!("{name}/rt/interpreter"), |b| {
        b.iter(|| prepared.run_interpreter());
    });

    g.bench_function(format!("{name}/rt/jit"), |b| {
        b.iter(|| prepared.run_jit());
    });

    g.finish();
}

fn run_bytecode_bench(c: &mut Criterion, bench: &revmc_cli::Bench) {
    let def = bench;
    let name = bench.name;

    let mut g = c.benchmark_group(name);
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(1));
    g.measurement_time(Duration::from_secs(5));

    let gas_limit = u64::MAX / 2;
    let calldata: revmc::primitives::Bytes = def.calldata.clone().into();
    let bytecode_raw = Bytecode::new_raw(revmc::primitives::Bytes::copy_from_slice(&def.bytecode));

    // ── Compile-time ────────────────────────────────────────────────────

    if !SKIP_COMPILE.contains(&name) {
        g.bench_function(format!("{name}/compile/translate"), |b| {
            b.iter_batched_ref(
                || new_compiler(OptimizationLevel::None),
                |compiler| {
                    compiler.translate(name, &def.bytecode, SPEC_ID).unwrap();
                },
                BatchSize::PerIteration,
            )
        });

        g.bench_function(format!("{name}/compile/jit"), |b| {
            b.iter_batched_ref(
                || {
                    let mut compiler = new_compiler(OptimizationLevel::default());
                    let id = compiler.translate(name, &def.bytecode, SPEC_ID).expect("translate");
                    (compiler, id)
                },
                |(compiler, id)| unsafe {
                    compiler.jit_function(*id).unwrap();
                },
                BatchSize::PerIteration,
            )
        });
    }

    // ── Runtime ─────────────────────────────────────────────────────────

    let mut host = BenchHost::new(SPEC_ID);
    host.apply_bench(def);
    let table = instruction_table::<EthInterpreter, BenchHost>();

    let opt_level = revmc::OptimizationLevel::default();
    let backend = EvmLlvmBackend::new(false, opt_level).unwrap();
    let mut compiler = EvmCompiler::new(backend);
    compiler.inspect_stack_length(!def.stack_input.is_empty());
    compiler.gas_metering(true);

    if let Some(native) = def.native {
        g.bench_function(format!("{name}/rt/native"), |b| {
            b.iter_batched(|| (), |()| native(), BatchSize::SmallInput)
        });
    }

    let new_interpreter = || {
        let ext_bytecode = ExtBytecode::new(bytecode_raw.clone());
        let input = InputsImpl {
            input: revm_interpreter::CallInput::Bytes(calldata.clone()),
            ..Default::default()
        };
        revm_interpreter::Interpreter::new(
            SharedMemory::new(),
            ext_bytecode,
            input,
            false,
            SPEC_ID,
            gas_limit,
        )
    };

    const NO_GAS_BENCHES: &[&str] = &["fibonacci", "fibonacci-calldata", "factorial"];
    let mut jit_variants: Vec<(&str, (bool, bool))> = vec![("default", (true, true))];
    if NO_GAS_BENCHES.contains(&name) {
        jit_variants.push(("no_gas", (false, true)));
    }
    let jit_ids: Vec<_> = jit_variants
        .iter()
        .map(|&(kind, (gas, stack))| {
            compiler.gas_metering(gas);
            unsafe { compiler.stack_bound_checks(stack) };
            (
                kind,
                compiler.translate(kind, bytecode_raw.original_byte_slice(), SPEC_ID).expect(kind),
            )
        })
        .collect();
    for &(kind, fn_id) in &jit_ids {
        let jit = unsafe { compiler.jit_function(fn_id) }.expect(kind);
        g.bench_function(format!("{name}/rt/jit/{kind}"), |b| {
            b.iter_batched_ref(
                || {
                    let mut stack = EvmStack::new();
                    for (i, input) in def.stack_input.iter().enumerate() {
                        stack.as_mut_slice()[i] = (*input).into();
                    }
                    (new_interpreter(), stack)
                },
                |(interpreter, stack)| {
                    let mut stack_len = def.stack_input.len();
                    let mut ecx = EvmContext::from_interpreter(interpreter, &mut host);
                    unsafe { jit.call(Some(stack), Some(&mut stack_len), &mut ecx) }
                },
                BatchSize::SmallInput,
            )
        });
    }

    g.bench_function(format!("{name}/rt/interpreter"), |b| {
        b.iter_batched_ref(
            || {
                let mut interpreter = new_interpreter();
                interpreter.stack.data_mut().extend_from_slice(&def.stack_input);
                interpreter
            },
            |interpreter| interpreter.run_plain(&table, &mut host),
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

fn new_compiler(opt_level: OptimizationLevel) -> EvmCompiler<EvmLlvmBackend> {
    let backend = EvmLlvmBackend::new(false, opt_level).unwrap();
    EvmCompiler::new(backend)
}

criterion_group!(benches, bench);
criterion_main!(benches);

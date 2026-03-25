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
use revmc_cli::{Bench, BenchHost, BenchKind, PreparedFixtureBench};
use std::time::Duration;

const SPEC_ID: SpecId = SpecId::OSAKA;

/// Benchmarks that are too slow for CI due to large bytecode (LLVM compilation time under
/// valgrind).
const SKIP_COMPILE: &[&str] = &[
    "snailtracer",
    "seaport",
    "fiat_token",
    "uniswap_v2_pair",
    "airdrop",
    "erc20_transfer",
    "weth",
    "usdc_proxy",
];
/// Benchmarks that are too slow for CI entirely (runtime is also very slow under valgrind).
const SKIP_ALL: &[&str] = &["seaport", "snailtracer"];

fn bench(c: &mut Criterion) {
    for bench in &revmc_cli::get_benches() {
        if SKIP_ALL.contains(&bench.name) {
            continue;
        }
        match &bench.kind {
            BenchKind::Bytecode { .. } => run_bytecode_bench(c, bench),
            BenchKind::TxFixture(def) => run_fixture_bench(c, bench.name, def),
        }
    }
}

fn run_fixture_bench(c: &mut Criterion, name: &str, def: &revmc_cli::FixtureBenchDef) {
    let prepared = PreparedFixtureBench::load(def);

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

fn run_bytecode_bench(c: &mut Criterion, bench: &Bench) {
    let (bytecode, calldata, stack_input, native, host_config) =
        bench.as_bytecode().expect("expected bytecode bench");
    let name = bench.name;

    let mut g = c.benchmark_group(name);
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(1));
    g.measurement_time(Duration::from_secs(5));

    let gas_limit = u64::MAX / 2;
    let calldata: revmc::primitives::Bytes = calldata.to_vec().into();
    let bytecode_raw = Bytecode::new_raw(revmc::primitives::Bytes::copy_from_slice(bytecode));

    // ── Compile-time ────────────────────────────────────────────────────

    if !SKIP_COMPILE.contains(&name) {
        g.bench_function(format!("{name}/compile/translate"), |b| {
            b.iter_batched(
                || new_compiler(OptimizationLevel::None),
                |mut compiler| {
                    compiler.translate(name, bytecode, SPEC_ID).unwrap();
                },
                BatchSize::PerIteration,
            )
        });

        g.bench_function(format!("{name}/compile/jit"), |b| {
            b.iter_batched(
                || {
                    let mut compiler = new_compiler(OptimizationLevel::Aggressive);
                    let id = compiler.translate(name, bytecode, SPEC_ID).expect("translate");
                    (compiler, id)
                },
                |(mut compiler, id)| unsafe {
                    compiler.jit_function(id).unwrap();
                },
                BatchSize::PerIteration,
            )
        });
    }

    // ── Runtime ─────────────────────────────────────────────────────────

    let mut host = BenchHost::new(SPEC_ID);
    host.apply_config(host_config);
    let table = instruction_table::<EthInterpreter, BenchHost>();

    let opt_level = revmc::OptimizationLevel::Aggressive;
    let backend = EvmLlvmBackend::new(false, opt_level).unwrap();
    let mut compiler = EvmCompiler::new(backend);
    compiler.inspect_stack_length(!stack_input.is_empty());
    compiler.gas_metering(true);

    if let Some(native) = native {
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

    let jit_matrix = [("default", (true, true)), ("no_gas", (false, true))];
    let jit_ids = jit_matrix.map(|(kind, (gas, stack))| {
        compiler.gas_metering(gas);
        unsafe { compiler.stack_bound_checks(stack) };
        (kind, compiler.translate(kind, bytecode_raw.original_byte_slice(), SPEC_ID).expect(kind))
    });
    for &(kind, fn_id) in &jit_ids {
        let jit = unsafe { compiler.jit_function(fn_id) }.expect(kind);
        g.bench_function(format!("{name}/rt/jit/{kind}"), |b| {
            b.iter_batched_ref(
                || {
                    let mut stack = EvmStack::new();
                    for (i, input) in stack_input.iter().enumerate() {
                        stack.as_mut_slice()[i] = (*input).into();
                    }
                    (new_interpreter(), stack)
                },
                |(interpreter, stack)| {
                    let mut stack_len = stack_input.len();
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
                interpreter.stack.data_mut().extend_from_slice(stack_input);
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

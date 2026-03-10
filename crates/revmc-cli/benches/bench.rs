#![allow(missing_docs)]

use criterion::{
    BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use revm_bytecode::Bytecode;
use revm_interpreter::{
    InputsImpl, SharedMemory,
    host::DummyHost,
    instruction_table,
    interpreter::{EthInterpreter, ExtBytecode},
};
use revmc::{
    EvmCompiler, EvmCompilerFn, EvmContext, EvmLlvmBackend, EvmStack, primitives::hardfork::SpecId,
};
use revmc_cli::Bench;
use std::time::Duration;

const SPEC_ID: SpecId = SpecId::OSAKA;

fn bench(c: &mut Criterion) {
    for bench in &revmc_cli::get_benches() {
        run_bench(c, bench);
    }
}

fn run_bench(c: &mut Criterion, bench: &Bench) {
    let Bench { name, bytecode, calldata, stack_input, native, requires_storage } = bench;

    let mut g = mk_group(c, name);

    let gas_limit = 1_000_000_000;

    let calldata: revmc::primitives::Bytes = calldata.clone().into();

    let bytecode_raw = Bytecode::new_raw(revmc::primitives::Bytes::copy_from_slice(bytecode));

    let mut host = DummyHost::new(SPEC_ID);

    let table = instruction_table::<EthInterpreter, DummyHost>();

    // Set up the compiler.
    let opt_level = revmc::OptimizationLevel::Aggressive;
    let backend = EvmLlvmBackend::new(false, opt_level).unwrap();
    let mut compiler = EvmCompiler::new(backend);
    compiler.inspect_stack_length(!stack_input.is_empty());
    compiler.gas_metering(true);

    if let Some(native) = *native {
        g.bench_function("native", |b| b.iter(native));
    }

    let mut stack = EvmStack::new();
    let mut call_jit = |f: EvmCompilerFn| {
        for (i, input) in stack_input.iter().enumerate() {
            stack.as_mut_slice()[i] = (*input).into();
        }
        let mut stack_len = stack_input.len();

        let ext_bytecode = ExtBytecode::new(bytecode_raw.clone());
        let input = InputsImpl {
            input: revm_interpreter::CallInput::Bytes(calldata.clone()),
            ..Default::default()
        };
        let mut interpreter = revm_interpreter::Interpreter::new(
            SharedMemory::new(),
            ext_bytecode,
            input,
            false,
            SPEC_ID,
            gas_limit,
        );
        let mut ecx = EvmContext::from_interpreter(&mut interpreter, &mut host);

        unsafe { f.call(Some(&mut stack), Some(&mut stack_len), &mut ecx) }
    };

    let jit_matrix = [
        ("default", (true, true)),
        ("no_gas", (false, true)),
        // ("no_stack", (true, false)),
        // ("no_gas_no_stack", (false, false)),
    ];
    let jit_ids = jit_matrix.map(|(name, (gas, stack))| {
        compiler.gas_metering(gas);
        unsafe { compiler.stack_bound_checks(stack) };
        (name, compiler.translate(name, bytecode_raw.original_byte_slice(), SPEC_ID).expect(name))
    });
    for &(name, fn_id) in &jit_ids {
        let jit = unsafe { compiler.jit_function(fn_id) }.expect(name);
        g.bench_function(format!("revmc/{name}"), |b| b.iter(|| call_jit(jit)));
    }

    // Skip interpreter benchmark for contracts that require storage (SLOAD/SSTORE)
    // since DummyHost doesn't support storage operations.
    if !requires_storage {
        g.bench_function("revm-interpreter", |b| {
            b.iter(|| {
                let ext_bytecode = ExtBytecode::new(bytecode_raw.clone());
                let input = InputsImpl {
                    input: revm_interpreter::CallInput::Bytes(calldata.clone()),
                    ..Default::default()
                };
                let mut interpreter = revm_interpreter::Interpreter::new(
                    SharedMemory::new(),
                    ext_bytecode,
                    input,
                    false,
                    SPEC_ID,
                    gas_limit,
                );

                interpreter.stack.data_mut().extend_from_slice(stack_input);

                let action = interpreter.run_plain(&table, &mut host);
                let result = action
                    .instruction_result()
                    .unwrap_or(revm_interpreter::InstructionResult::Stop);
                assert!(result.is_ok(), "Interpreter failed with {result:?}");
                assert!(action.is_return(), "Interpreter bad action: {action:?}");
                action
            })
        });
    }

    g.finish();
}

fn mk_group<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut g = c.benchmark_group(name);
    g.sample_size(20);
    g.warm_up_time(Duration::from_secs(2));
    g.measurement_time(Duration::from_secs(5));
    g
}

criterion_group!(benches, bench);
criterion_main!(benches);

#![allow(missing_docs)]

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use revm_interpreter::SharedMemory;
use revm_jit::{llvm, new_llvm_backend, EvmCompiler, EvmCompilerFn, EvmContext, EvmStack};
use revm_jit_cli::Bench;
use revm_primitives::{Env, SpecId};
use std::time::Duration;

const SPEC_ID: SpecId = SpecId::CANCUN;

fn bench(c: &mut Criterion) {
    for bench in &revm_jit_cli::get_benches() {
        if matches!(bench.name, "push0_proxy" | "weth") {
            continue;
        }
        run_bench(c, bench);
    }
}

fn run_bench(c: &mut Criterion, bench: &Bench) {
    let Bench { name, bytecode, calldata, stack_input, native } = bench;

    let mut g = mk_group(c, name);

    let gas_limit = 1_000_000_000;

    let mut env = Env::default();
    env.tx.data = calldata.clone().into();
    env.tx.gas_limit = gas_limit;

    let bytecode = revm_interpreter::analysis::to_analysed(revm_primitives::Bytecode::new_raw(
        revm_primitives::Bytes::copy_from_slice(bytecode),
    ));
    let contract = revm_interpreter::Contract::new_env(&env, bytecode, None);
    let mut host = revm_interpreter::DummyHost::new(env);

    let bytecode = contract.bytecode.original_byte_slice();

    let table = &revm_interpreter::opcode::make_instruction_table::<
        revm_interpreter::DummyHost,
        revm_primitives::CancunSpec,
    >();

    // Set up the compiler.
    let opt_level = revm_jit::OptimizationLevel::Aggressive;
    let context = llvm::inkwell::context::Context::create();
    let backend = new_llvm_backend(&context, false, opt_level).unwrap();
    let mut compiler = EvmCompiler::new(backend);
    compiler.inspect_stack_length(!stack_input.is_empty());
    compiler.gas_metering(true);

    if let Some(native) = *native {
        g.bench_function("native", |b| b.iter(native));
    }

    let mut stack = EvmStack::new();
    let mut call_jit = |f: EvmCompilerFn| {
        for (i, input) in stack_input.iter().enumerate() {
            stack.as_mut_slice()[i] = input.into();
        }
        let mut stack_len = stack_input.len();

        let mut interpreter =
            revm_interpreter::Interpreter::new(contract.clone(), gas_limit, false);
        host.clear();
        let mut ecx = EvmContext::from_interpreter(&mut interpreter, &mut host);

        let r = unsafe { f.call(Some(&mut stack), Some(&mut stack_len), &mut ecx) };
        assert!(r.is_ok(), "JIT failed with {r:?}");
        r
    };

    let jit_matrix = [
        ("default", (true, true)),
        ("no_gas", (false, true)),
        ("no_stack", (true, false)),
        ("no_gas_no_stack", (false, false)),
    ];
    let jit_ids = jit_matrix.map(|(name, (gas, stack))| {
        compiler.gas_metering(gas);
        unsafe { compiler.stack_bound_checks(stack) };
        (name, compiler.translate(None, bytecode, SPEC_ID).expect(name))
    });
    for &(name, fn_id) in &jit_ids {
        let jit = compiler.jit_function(fn_id).expect(name);
        g.bench_function(&format!("revm-jit/{name}"), |b| b.iter(|| call_jit(jit)));
    }

    g.bench_function("revm-interpreter", |b| {
        b.iter(|| {
            let mut int = revm_interpreter::Interpreter::new(contract.clone(), gas_limit, false);
            let mut host = host.clone();

            int.stack.data_mut().extend_from_slice(stack_input);

            let action = int.run(SharedMemory::new(), table, &mut host);
            assert!(
                int.instruction_result.is_ok(),
                "Interpreter failed with {:?}",
                int.instruction_result
            );
            assert!(action.is_return(), "Interpreter bad action: {action:?}");
            action
        })
    });

    g.finish();
}

fn mk_group<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut g = c.benchmark_group(name);
    g.sample_size(20);
    g.warm_up_time(Duration::from_secs(5));
    g.measurement_time(Duration::from_secs(15));
    g
}

criterion_group!(benches, bench);
criterion_main!(benches);

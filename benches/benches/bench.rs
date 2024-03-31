use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use revm_interpreter::SharedMemory;
use revm_jit::{llvm, EvmContext, EvmStack, JitEvm, JitEvmFn};
use revm_jit_benches::Bench;
use revm_primitives::{Env, SpecId};
use std::time::Duration;

const SPEC_ID: SpecId = SpecId::CANCUN;

fn bench(c: &mut Criterion) {
    for bench in &revm_jit_benches::get_benches() {
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
    let bytecode_hash = bytecode.hash_slow();
    let contract = revm_interpreter::Contract::new_env(&env, bytecode, bytecode_hash);
    let mut host = revm_interpreter::DummyHost::new(env);

    let bytecode = contract.bytecode.original_bytecode_slice();

    let table = &revm_interpreter::opcode::make_instruction_table::<
        revm_interpreter::DummyHost,
        revm_primitives::CancunSpec,
    >();

    // Set up JIT.
    let opt_level = revm_jit::OptimizationLevel::Aggressive;
    let context = llvm::inkwell::context::Context::create();
    let backend = llvm::JitEvmLlvmBackend::new(&context, opt_level).unwrap();
    let mut jit = JitEvm::new(backend);
    if !stack_input.is_empty() {
        jit.set_inspect_stack_length(true);
    }
    jit.set_disable_gas(true);

    if let Some(native) = *native {
        g.bench_function("native", |b| b.iter(native));
    }

    let mut stack = EvmStack::new();
    let mut call_jit = |f: JitEvmFn| {
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

    let jit_no_gas = jit.compile(Some(name), bytecode, SPEC_ID).unwrap();
    g.bench_function("revm-jit/no_gas", |b| b.iter(|| call_jit(jit_no_gas)));

    unsafe { jit.free_all_functions() }.unwrap();
    jit.set_disable_gas(false);
    let jit_gas = jit.compile(Some(name), bytecode, SPEC_ID).unwrap();
    g.bench_function("revm-jit/gas", |b| b.iter(|| call_jit(jit_gas)));

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

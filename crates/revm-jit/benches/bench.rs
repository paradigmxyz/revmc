#![allow(missing_docs)]

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use revm_interpreter::{opcode as op, EMPTY_SHARED_MEMORY};
use revm_jit::{llvm, EvmContext, EvmStack, EvmWord, JitEvm, JitEvmFn};
use revm_primitives::{hex, SpecId, U256};
use std::{hint::black_box, time::Duration};

const SPEC_ID: SpecId = SpecId::CANCUN;

#[derive(Clone, Debug, Default)]
struct Bench {
    name: &'static str,
    bytecode: Vec<u8>,
    calldata: Vec<u8>,
    stack_input: Vec<U256>,
    native: Option<fn()>,
}

fn bench(c: &mut Criterion) {
    let benches = [
        //
        Bench {
            name: "fibonacci",
            bytecode: FIBONACCI.to_vec(),
            calldata: vec![],
            stack_input: vec![U256::from(69)],
            native: Some(|| {
                black_box(fibonacci_rust(70));
            }),
        },
        Bench {
            name: "counter",
            bytecode: hex::decode(include_str!("../../../data/counter.rt.hex")).unwrap(),
            // `increment()`
            calldata: hex!("d09de08a").to_vec(),
            stack_input: vec![],
            native: None,
        },
        // Bench {
        //     name: "snailtracer",
        //     bytecode: hex::decode(include_str!("../../../data/snailtracer.rt.hex")).unwrap(),
        //     calldata: hex!("30627b7c").to_vec(),
        //     stack_input: vec![],
        //     native: None,
        // },
    ];
    for bench in &benches {
        run_bench(c, bench);
    }
}

fn run_bench(c: &mut Criterion, bench: &Bench) {
    let Bench { name, bytecode, calldata, stack_input, native } = bench;

    let mut g = mk_group(c, name);

    let contract = revm_interpreter::Contract {
        bytecode: revm_interpreter::analysis::to_analysed(revm_primitives::Bytecode::new_raw(
            revm_primitives::Bytes::copy_from_slice(bytecode),
        ))
        .try_into()
        .unwrap(),
        input: calldata.clone().into(),
        ..Default::default()
    };
    let table = &revm_interpreter::opcode::make_instruction_table::<
        revm_interpreter::DummyHost,
        revm_primitives::CancunSpec,
    >();
    let host = revm_interpreter::DummyHost::new(Default::default());

    // Set up JIT.
    let opt_level = revm_jit::OptimizationLevel::Aggressive;
    let context = llvm::inkwell::context::Context::create();
    let backend = llvm::JitEvmLlvmBackend::new(&context, opt_level).unwrap();
    let mut jit = JitEvm::new(backend);
    jit.set_disable_gas(true);

    let gas_limit = 100_000;

    if let Some(native) = *native {
        g.bench_function("native", |b| b.iter(native));
    }

    let mut stack_buf = EvmStack::new_heap();
    let mut call_jit = |f: JitEvmFn| {
        stack_buf.clear();
        stack_buf.extend(stack_input.iter().map(|w| EvmWord::from(*w)));
        let stack = EvmStack::from_mut_vec(&mut stack_buf);
        let mut stack_len = stack_input.len();

        let mut interpreter =
            revm_interpreter::Interpreter::new(contract.clone(), gas_limit, false);
        let mut host = host.clone();
        let mut ecx = EvmContext::from_interpreter(&mut interpreter, &mut host);

        unsafe { f.call(Some(stack), Some(&mut stack_len), &mut ecx) }
    };

    let jit_no_gas = jit.compile(bytecode, SPEC_ID).unwrap();
    g.bench_function("revm-jit/no_gas", |b| b.iter(|| call_jit(jit_no_gas)));

    unsafe { jit.free_all_functions() }.unwrap();
    jit.set_disable_gas(false);
    let jit_gas = jit.compile(bytecode, SPEC_ID).unwrap();
    g.bench_function("revm-jit/gas", |b| b.iter(|| call_jit(jit_gas)));

    g.bench_function("revm-interpreter", |b| {
        b.iter(|| {
            let mut int = revm_interpreter::Interpreter::new(contract.clone(), gas_limit, false);
            let mut host = host.clone();

            int.stack.data_mut().extend_from_slice(stack_input);

            int.run(EMPTY_SHARED_MEMORY, table, &mut host)
        })
    });

    g.finish();
}

fn fibonacci_rust(n: u16) -> U256 {
    let mut a = U256::from(0);
    let mut b = U256::from(1);
    for _ in 0..n {
        let tmp = a;
        a = b;
        b = b.wrapping_add(tmp);
    }
    a
}

fn mk_group<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut g = c.benchmark_group(name);
    g.sample_size(100);
    g.warm_up_time(Duration::from_secs(5));
    g.measurement_time(Duration::from_secs(10));
    g
}

criterion_group!(benches, bench);
criterion_main!(benches);

#[rustfmt::skip]
const FIBONACCI: &[u8] = &[
    // input to the program (which fib number we want)
    // op::PUSH2, input[0], input[1],
    op::JUMPDEST, op::JUMPDEST, op::JUMPDEST,

    // 1st/2nd fib number
    op::PUSH1, 0,
    op::PUSH1, 1,
    // 7

    // MAINLOOP:
    op::JUMPDEST,
    op::DUP3,
    op::ISZERO,
    op::PUSH1, 28, // cleanup
    op::JUMPI,

    // fib step
    op::DUP2,
    op::DUP2,
    op::ADD,
    op::SWAP2,
    op::POP,
    op::SWAP1,
    // 19

    // decrement fib step counter
    op::SWAP2,
    op::PUSH1, 1,
    op::SWAP1,
    op::SUB,
    op::SWAP2,
    op::PUSH1, 7, // goto MAINLOOP
    op::JUMP,
    // 28

    // CLEANUP:
    op::JUMPDEST,
    op::SWAP2,
    op::POP,
    op::POP,
    // done: requested fib number is the only element on the stack!
    op::STOP,
];

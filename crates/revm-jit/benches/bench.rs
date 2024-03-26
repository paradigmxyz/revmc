#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion};
use revm_interpreter::{opcode as op, Gas, EMPTY_SHARED_MEMORY};
use revm_jit::{llvm, EvmContext, EvmStack, JitEvm, JitEvmFn};
use revm_primitives::{SpecId, U256};
use std::{hint::black_box, time::Duration};

const SPEC_ID: SpecId = SpecId::CANCUN;

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("fibonacci");
    g.sample_size(100);
    g.warm_up_time(Duration::from_secs(5));
    g.measurement_time(Duration::from_secs(10));

    // Compile.
    let bytecode = FIBONACCI;
    let opt_level = revm_jit::OptimizationLevel::Aggressive;
    let context = llvm::inkwell::context::Context::create();
    let backend = llvm::JitEvmLlvmBackend::new(&context, opt_level).unwrap();
    let mut jit = JitEvm::new(backend);
    jit.set_pass_stack_through_args(true);
    jit.set_pass_stack_len_through_args(true);
    jit.set_disable_gas(true);
    let jit_no_gas = jit.compile(bytecode, SPEC_ID).unwrap();

    let gas_limit = 100_000;

    let input: u16 = black_box(69);
    let input_u256 = U256::from(input);
    let input_adj = input + 1;
    g.bench_function("rust", |b| b.iter(|| fibonacci_rust(input_adj)));

    let mut gas = Gas::new(gas_limit);
    let mut stack_buf = EvmStack::new_heap();
    stack_buf.push(input_u256.into());
    let stack = EvmStack::from_mut_vec(&mut stack_buf);
    let mut stack_len = 1;
    let mut cx = EvmContext::dummy_do_not_use();
    let mut call_jit = |f: JitEvmFn| {
        stack.as_mut_slice()[0] = input_u256.into();
        gas = Gas::new(gas_limit);
        stack_len = 1;
        unsafe { f.call(Some(&mut gas), Some(stack), Some(&mut stack_len), &mut cx) }
    };

    g.bench_function("revm-jit/no_gas", |b| b.iter(|| call_jit(jit_no_gas)));

    unsafe { jit.free_all_functions() }.unwrap();
    jit.set_disable_gas(false);
    let jit_gas = jit.compile(bytecode, SPEC_ID).unwrap();

    g.bench_function("revm-jit/gas", |b| b.iter(|| call_jit(jit_gas)));

    let contract = Box::new(revm_interpreter::Contract {
        bytecode: revm_interpreter::analysis::to_analysed(revm_primitives::Bytecode::new_raw(
            revm_primitives::Bytes::from_static(bytecode),
        ))
        .try_into()
        .unwrap(),
        ..Default::default()
    });
    let table = revm_interpreter::opcode::make_instruction_table::<
        revm_interpreter::DummyHost,
        revm_primitives::LatestSpec,
    >();
    let mut host = revm_interpreter::DummyHost::new(Default::default());

    g.bench_function("revm-interpreter", |b| {
        b.iter(|| {
            let mut int = revm_interpreter::Interpreter::new(contract.clone(), gas_limit, false);
            int.stack.push(input_u256).unwrap();
            int.run(EMPTY_SHARED_MEMORY, &table, &mut host)
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

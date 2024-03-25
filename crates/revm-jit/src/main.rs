#![allow(missing_docs)]

use revm_interpreter::{opcode as op, Gas};
use revm_jit::{Backend, EvmStack, InstructionResult, JitEvm, OptimizationLevel};
use revm_primitives::{SpecId, U256};
use std::{hint::black_box, path::PathBuf};

fn main() -> color_eyre::Result<()> {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let _ = color_eyre::install();
    let _ = tracing_subscriber::fmt::try_init();

    // Build the compiler.
    // let opt_level = OptimizationLevel::None;
    let opt_level = OptimizationLevel::Aggressive;
    let context = revm_jit::llvm::inkwell::context::Context::create();
    let backend = revm_jit::llvm::JitEvmLlvmBackend::new(&context, opt_level).unwrap();
    let mut jit = JitEvm::new(backend);
    jit.set_dump_to(Some(PathBuf::from("./target/")));
    jit.set_debug_assertions(false);
    jit.set_pass_stack_through_args(true);
    jit.set_pass_stack_len_through_args(true);
    jit.set_disable_gas(true);

    #[rustfmt::skip]
    let code: &[u8] = &[
        op::PUSH1, 0x20, op::SIGNEXTEND
    ];

    let mut stack_buf = EvmStack::new_heap();
    let stack = EvmStack::from_mut_vec(&mut stack_buf);
    stack.as_mut_slice()[0] = U256::from(0xab).into();
    // let mut gas = Gas::new(100_000);
    let f = jit.compile(code, SpecId::LATEST)?;
    let ret = unsafe { f.call(None, Some(stack), Some(&mut 1)) };
    assert_eq!(ret, InstructionResult::Stop);
    // assert_eq!(stack.as_slice()[0], U256::from(0xab).into());
    // fibonacci(jit)?;

    Ok(())
}

#[allow(dead_code)]
fn fibonacci<B: Backend>(mut jit: JitEvm<B>) -> color_eyre::Result<()> {
    let gas_limit = 100_000;

    let input_u16: u16 = 69;
    let input = U256::from(input_u16);

    // Compile the bytecode.
    #[rustfmt::skip]
    let bytecode: &[u8] = &[
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
    let f = jit.compile(bytecode, SpecId::LATEST)?;

    let n_iters: u64 = std::env::args().nth(1).map(|s| s.parse().expect("not u64")).unwrap_or(1);

    bench(n_iters, "RUST", || fibonacci_rust(input_u16 + 1));

    let mut gas = Gas::new(gas_limit);
    let mut stack_buf = EvmStack::new_heap();
    stack_buf.push(input.into());
    let stack = EvmStack::from_mut_vec(&mut stack_buf);
    let mut stack_len = 1;

    bench(n_iters, " JIT", || {
        stack.as_mut_slice()[0] = input.into();
        gas = Gas::new(gas_limit);
        stack_len = 1;
        unsafe { f.call(Some(&mut gas), Some(stack), Some(&mut stack_len)) }
    });

    let contract = Box::new(revm_interpreter::Contract {
        bytecode: revm_interpreter::analysis::to_analysed(revm_primitives::Bytecode::new_raw(
            revm_primitives::Bytes::copy_from_slice(bytecode),
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

    bench(n_iters, "REVM", || {
        let mut int = revm_interpreter::Interpreter::new(contract.clone(), gas_limit, false);
        int.stack.push(input).unwrap();
        int.run(Default::default(), &table, &mut host)
    });

    Ok(())
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

fn bench<T>(n_iters: u64, name: &str, mut f: impl FnMut() -> T) {
    let t = std::time::Instant::now();
    for _ in 0..n_iters {
        black_box(f());
    }
    let d = t.elapsed();
    eprintln!("{name}: {:>9?} ({d:>12?} / {n_iters})", d / n_iters as u32);
}

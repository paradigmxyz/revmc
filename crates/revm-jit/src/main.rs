#![allow(missing_docs)]

use revm_interpreter::{opcode as op, Gas};
use revm_jit::{Backend, EvmStack, JitEvm, OptimizationLevel};
use revm_primitives::SpecId;
use std::path::PathBuf;

fn main() -> color_eyre::Result<()> {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let _ = color_eyre::install();
    let _ = tracing_subscriber::fmt::try_init();

    // Build the compiler.
    let context = revm_jit::llvm::inkwell::context::Context::create();
    let backend =
        revm_jit::llvm::JitEvmLlvmBackend::new(&context, OptimizationLevel::Aggressive).unwrap();
    let mut jit = JitEvm::new(backend);
    jit.set_dump_to(Some(PathBuf::from("./target/")));
    jit.set_pass_stack_through_args(true);
    // jit.set_disable_gas(true);

    // #[rustfmt::skip]
    // let code: &[u8] = &[
    //     op::PUSH1, 3, op::JUMP, op::JUMPDEST
    // ];

    // let mut stack_buf = EvmStack::new_heap();
    // let stack = EvmStack::from_mut_vec(&mut stack_buf);
    // let mut gas = Gas::new(100_000);
    // let f = jit.compile(code, SpecId::LATEST)?;
    // unsafe { f.call(Some(&mut gas), Some(stack), None) };
    fibonacci(jit)?;

    Ok(())
}

#[allow(dead_code)]
fn fibonacci<B: Backend>(mut jit: JitEvm<B>) -> color_eyre::Result<()> {
    let gas_limit = 100_000;

    // Compile the bytecode.
    let input: u16 = 69;
    let input = input.to_be_bytes();
    #[rustfmt::skip]
    let bytecode: &[u8] = &[
        // input to the program (which fib number we want)
        op::PUSH2, input[0], input[1],

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

    let n_iters: u64 = std::env::args().nth(1).expect("expected n_iters").parse().expect("not u64");

    let mut gas = Gas::new(gas_limit);
    let mut stack_buf = EvmStack::new_heap();
    let stack = EvmStack::from_mut_vec(&mut stack_buf);
    let t = std::time::Instant::now();
    for _ in 0..n_iters {
        std::hint::black_box(unsafe { f.call(Some(&mut gas), Some(stack), None) });
        gas = Gas::new(gas_limit);
    }
    let d = t.elapsed();
    eprintln!(" JIT: {:>8?} ({d:?}/{n_iters})", d / n_iters as u32);

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

    let t = std::time::Instant::now();
    for _ in 0..n_iters {
        let mut int = revm_interpreter::Interpreter::new(contract.clone(), gas_limit, false);
        std::hint::black_box(int.run(Default::default(), &table, &mut host));
    }
    let d = t.elapsed();
    eprintln!("REVM: {:>8?} ({d:?}/{n_iters})", d / n_iters as u32);

    Ok(())
}

#![allow(missing_docs)]

use revm_interpreter::opcode as op;
use revm_jit::{ContextStack, InstructionResult, JitEvm, OptimizationLevel};
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
    jit.dump_to(Some(PathBuf::from("./target/")));

    /*
    vec![
        // input to the program (which fib number we want)
        Push(2, U256::zero() + 6000 - 2), // 5 (needs to be >= 3)
        // 1st/2nd fib number
        Push(1, U256::zero()),
        Push(1, U256::one()),
        // 7

        // MAINLOOP:
        Jumpdest,
        Dup3,
        Iszero,
        Push(1, U256::zero() + 28), // CLEANUP
        Jumpi,
        // 13

        // fib step
        Dup2,
        Dup2,
        Add,
        Swap2,
        Pop,
        Swap1,
        // 19

        // decrement fib step counter
        Swap2,
        Push(1, U256::one()),
        Swap1,
        Sub,
        Swap2,
        // 25
        Push(1, U256::zero() + 7), // goto MAINLOOP
        Jump,
        // 28

        // CLEANUP:
        Jumpdest,
        Swap2,
        Pop,
        Pop,
        // done: requested fib number is only element on the stack!
        Stop,
    ]
    */

    // Compile the bytecode.
    let input: u16 = 100;
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

    // Run the compiled function.
    let mut stack = ContextStack::new();
    let ret = unsafe { f(&mut stack) };
    assert_eq!(ret, InstructionResult::Stop);
    println!("{}", stack.word(0));

    Ok(())
}

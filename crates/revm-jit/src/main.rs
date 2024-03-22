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
    jit.set_dump_to(Some(PathBuf::from("./target/")));

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
    let ret = unsafe { f(&mut stack, 100000) };
    assert_eq!(ret, InstructionResult::Stop);
    println!("{}", stack.word(0));

    Ok(())
}

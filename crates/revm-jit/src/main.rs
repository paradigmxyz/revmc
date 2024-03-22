#![allow(missing_docs)]

use revm_interpreter::opcode as op;
use revm_jit::{ContextStack, InstructionResult, JitEvm};
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
    let backend = revm_jit::llvm::JitEvmLlvmBackend::new(&context).unwrap();
    let mut jit = JitEvm::new(backend);
    jit.dump_to(Some(PathBuf::from("./target/")));

    // Compile the bytecode.
    #[rustfmt::skip]
    let bytecode = [
        op::PUSH1, 0x42,
        op::PUSH1, 0x69,
        op::ADD,
        op::STOP,
    ];
    let f = jit.compile(&bytecode, SpecId::LATEST)?;

    // Run the compiled function.
    let mut stack = ContextStack::new();
    let ret = unsafe { f(&mut stack) };
    assert_eq!(ret, InstructionResult::Stop);

    Ok(())
}

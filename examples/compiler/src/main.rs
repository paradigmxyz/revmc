//! Simple JIT compiler example.
//!
//! For a more complete example, see the `revmc-cli` crate.

use clap::Parser;
use eyre::Context;
use revmc::{
    interpreter::{Contract, DummyHost, Interpreter},
    primitives::SpecId,
    EvmCompiler, EvmLlvmBackend, OptimizationLevel,
};
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    #[arg(long, required_unless_present = "code_path")]
    code: Option<String>,
    #[arg(long, conflicts_with = "code")]
    code_path: Option<PathBuf>,
}

fn main() -> eyre::Result<()> {
    // Parse CLI arguments.
    let cli = Cli::parse();
    let code = match (cli.code, cli.code_path) {
        (Some(code), None) => code,
        (None, Some(path)) => std::fs::read_to_string(&path)
            .wrap_err_with(|| format!("Failed to read code from file: {path:?}"))?,
        _ => unreachable!(),
    };
    let bytecode = revmc::primitives::hex::decode(code.trim())
        .wrap_err("Failed to decode hex-encoded code")?;

    // Compile the code.
    let context = revmc::llvm::inkwell::context::Context::create();
    let backend = EvmLlvmBackend::new(&context, false, OptimizationLevel::Aggressive)?;
    let mut compiler = EvmCompiler::new(backend);
    let f = unsafe { compiler.jit("test", &bytecode[..], SpecId::CANCUN) }
        .wrap_err("Failed to JIT-compile code")?;

    // Set up runtime context and run the function.
    let mut interpreter = Interpreter::new(Contract::default(), 1_000_000, false);
    let mut host = DummyHost::default();
    let result = unsafe { f.call_with_interpreter(&mut interpreter, &mut host) };
    eprintln!("{result:#?}");

    Ok(())
}

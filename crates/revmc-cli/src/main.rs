#![allow(missing_docs)]

mod cmd;

use clap::{Parser, Subcommand};
use color_eyre::Result;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile and/or run EVM bytecode.
    Run(cmd::RunArgs),
    /// Compare interpreter vs JIT execution of Ethereum state tests.
    StatetestDiff(cmd::StatetestDiffArgs),
}

fn main() -> Result<()> {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let _ = color_eyre::install();
    let _ = init_tracing_subscriber();

    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => args.run(),
        Command::StatetestDiff(args) => args.run(),
    }
}

fn init_tracing_subscriber() -> Result<(), tracing_subscriber::util::TryInitError> {
    use tracing_subscriber::prelude::*;
    let registry = tracing_subscriber::Registry::default()
        .with(tracing_subscriber::EnvFilter::from_default_env());
    #[cfg(feature = "tracy")]
    let registry = registry.with(tracing_tracy::TracyLayer::default());
    registry.with(tracing_subscriber::fmt::layer()).try_init()
}

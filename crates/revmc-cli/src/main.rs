#![allow(missing_docs)]

mod cmd;

use clap::{Parser, Subcommand};
use color_eyre::Result;
use eyre::eyre;

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
    if revmc::runtime::maybe_run_jit_helper()? {
        return Ok(());
    }
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        // SAFETY: This is called at the very beginning of main, before any other threads are
        // spawned.
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
    }
    let _ = color_eyre::install();
    let _ = init_tracing_subscriber();

    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => args.run(),
        Command::StatetestDiff(args) => args.run(),
    }
}

fn init_tracing_subscriber() -> Result<()> {
    use tracing_subscriber::prelude::*;

    let (profile_layer, is_profiling) = match std::env::var("REVMC_PROFILE").as_deref() {
        Ok("tracy") => {
            if !cfg!(feature = "tracy") {
                return Err(eyre!("tracy profiler support is not compiled in"));
            }
            (Some(tracy_layer().boxed()), true)
        }
        Ok(s) => return Err(eyre!("unknown profiler '{s}'; valid values: 'tracy'")),
        Err(_) => (None, false),
    };
    // Disable the fmt layer when profiling.
    let fmt_layer = (!is_profiling).then(tracing_subscriber::fmt::layer);
    tracing_subscriber::Registry::default()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(profile_layer)
        .with(fmt_layer)
        .try_init()
        .map_err(Into::into)
}

#[cfg(feature = "tracy")]
fn tracy_layer() -> tracing_tracy::TracyLayer<impl tracing_tracy::Config> {
    struct Config(tracing_subscriber::fmt::format::DefaultFields);
    impl tracing_tracy::Config for Config {
        type Formatter = tracing_subscriber::fmt::format::DefaultFields;
        fn formatter(&self) -> &Self::Formatter {
            &self.0
        }
        fn format_fields_in_zone_name(&self) -> bool {
            false
        }
    }

    tracing_tracy::client::register_demangler!();
    tracing_tracy::TracyLayer::new(Config(Default::default()))
}

#[cfg(not(feature = "tracy"))]
fn tracy_layer() -> tracing_subscriber::layer::Identity {
    tracing_subscriber::layer::Identity::new()
}

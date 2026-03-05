// Binary that compares interpreter vs JIT execution of Ethereum state tests.
//
// Usage:
//   cargo run -p revmc-statetest --bin statetest-diff -- path/to/test.json
//   cargo run -p revmc-statetest --bin statetest-diff -- path/to/test.json --trace --keep-going

use clap::Parser;
use revm::{
    context::cfg::CfgEnv,
    primitives::{hardfork::SpecId, U256},
    statetest_types::{SpecName, TestSuite},
};
use revmc_statetest::{compiled::CompileCache, diagnostic};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "statetest-diff", about = "Compare interpreter vs JIT execution of state tests")]
struct Cli {
    /// Path to a JSON state test file.
    path: PathBuf,

    /// Continue after mismatches instead of stopping at the first one.
    #[arg(long)]
    keep_going: bool,

    /// Re-run mismatched tests with EIP-3155 tracing (interpreter only).
    #[arg(long)]
    trace: bool,

    /// Only run tests for this spec (e.g. "Cancun", "Prague").
    #[arg(long)]
    spec: Option<String>,

    /// Only run tests whose name contains this substring.
    #[arg(long)]
    name: Option<String>,
}

fn main() {
    let _ = tracing_subscriber::fmt::try_init();
    let cli = Cli::parse();

    if !cli.path.exists() {
        eprintln!("File not found: {}", cli.path.display());
        std::process::exit(1);
    }

    let s = std::fs::read_to_string(&cli.path).unwrap();
    let suite: TestSuite = match serde_json::from_str(&s) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to parse {}: {e}", cli.path.display());
            std::process::exit(1);
        }
    };

    let cache = CompileCache::new();
    let mut n_total = 0u64;
    let mut n_mismatches = 0u64;
    let mut n_compile_errors = 0u64;

    for (name, unit) in &suite.0 {
        if let Some(filter) = &cli.name {
            if !name.contains(filter.as_str()) {
                continue;
            }
        }

        let cache_state = unit.state();

        let mut cfg = CfgEnv::default();
        cfg.chain_id = unit.env.current_chain_id.unwrap_or(U256::ONE).try_into().unwrap_or(1);

        for (spec_name, tests) in &unit.post {
            if *spec_name == SpecName::Constantinople {
                continue;
            }

            if let Some(filter) = &cli.spec {
                let spec_str = format!("{spec_name:?}");
                if !spec_str.eq_ignore_ascii_case(filter) {
                    continue;
                }
            }

            let spec_id = spec_name.to_spec_id();
            cfg.set_spec_and_mainnet_gas_params(spec_id);

            if cfg.spec().is_enabled_in(SpecId::OSAKA) {
                cfg.set_max_blobs_per_tx(6);
            } else if cfg.spec().is_enabled_in(SpecId::PRAGUE) {
                cfg.set_max_blobs_per_tx(9);
            } else {
                cfg.set_max_blobs_per_tx(6);
            }

            let block = unit.block_env(&mut cfg);

            let compiled = match cache.jit_compile(unit, spec_id) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("COMPILE ERROR [{name}] spec={spec_name:?}: {e}");
                    n_compile_errors += 1;
                    continue;
                }
            };

            for (idx, test) in tests.iter().enumerate() {
                let tx = match test.tx_env(unit) {
                    Ok(tx) => tx,
                    Err(_) if test.expect_exception.is_some() => continue,
                    Err(_) => {
                        eprintln!(
                            "SKIP [{name}] spec={spec_name:?} idx={idx}: unknown private key"
                        );
                        continue;
                    }
                };

                n_total += 1;

                let interp = diagnostic::run_interpreter(&cfg, &block, &tx, &cache_state);
                let jit = diagnostic::run_jit(
                    &compiled,
                    &cache,
                    spec_id,
                    &cfg,
                    &block,
                    &tx,
                    &cache_state,
                );

                let mismatches = diagnostic::compare(&interp, &jit);
                if mismatches.is_empty() {
                    continue;
                }

                n_mismatches += 1;
                println!(
                    "\nMISMATCH [{name}] spec={spec_name:?} idx={idx} d={} g={} v={}",
                    test.indexes.data, test.indexes.gas, test.indexes.value,
                );
                for m in &mismatches {
                    println!("{m}");
                }

                println!("\n--- Interpreter post-state ---");
                println!("{}", interp.post_state_dump);
                println!("--- JIT post-state ---");
                println!("{}", jit.post_state_dump);

                if cli.trace {
                    eprintln!("\n=== Interpreter trace ===");
                    diagnostic::trace_interpreter(&cfg, &block, &tx, &cache_state);
                }

                if !cli.keep_going {
                    println!("\nStopping at first mismatch. Use --keep-going to continue.");
                    std::process::exit(1);
                }
            }
        }
    }

    println!("\n--- Summary ---");
    println!("Total test cases: {n_total}");
    println!("Mismatches: {n_mismatches}");
    if n_compile_errors > 0 {
        println!("Compile errors: {n_compile_errors}");
    }
    if n_mismatches > 0 {
        std::process::exit(1);
    }
}

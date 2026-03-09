use clap::Parser;
use color_eyre::{Result, eyre::eyre};
use revm::{
    context::cfg::CfgEnv,
    primitives::{U256, hardfork::SpecId as SI},
    statetest_types::{SpecName, TestSuite},
};
use revmc_statetest::{
    compiled::{CompileCache, CompileMode},
    diagnostic,
    runner::skip_test,
};
use std::path::PathBuf;

#[derive(Parser)]
pub(crate) struct StatetestDiffArgs {
    /// Path to a JSON state test file or directory.
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

impl StatetestDiffArgs {
    pub(crate) fn run(self) -> Result<()> {
        if !self.path.exists() {
            return Err(eyre!("Path not found: {}", self.path.display()));
        }

        let test_files = revmc_statetest::find_all_json_tests(&self.path);
        if test_files.is_empty() {
            return Err(eyre!("No JSON test files found in {}", self.path.display()));
        }

        eprintln!("Found {} test file(s)", test_files.len());

        let cache = CompileCache::new(CompileMode::Jit);
        let mut n_total = 0u64;
        let mut n_mismatches = 0u64;
        let mut n_compile_errors = 0u64;
        let mut n_files = 0u64;
        let mut n_skipped = 0u64;

        for test_path in &test_files {
            if skip_test(test_path) {
                n_skipped += 1;
                continue;
            }

            let s = match std::fs::read_to_string(test_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("SKIP {}: {e}", test_path.display());
                    continue;
                }
            };
            let suite: TestSuite = match serde_json::from_str(&s) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("SKIP {} (parse error): {e}", test_path.display());
                    continue;
                }
            };

            n_files += 1;

            for (name, unit) in &suite.0 {
                if let Some(filter) = &self.name
                    && !name.contains(filter.as_str())
                {
                    continue;
                }

                let cache_state = unit.state();

                let mut cfg = CfgEnv::default();
                cfg.chain_id =
                    unit.env.current_chain_id.unwrap_or(U256::ONE).try_into().unwrap_or(1);

                for (spec_name, tests) in &unit.post {
                    if *spec_name == SpecName::Constantinople {
                        continue;
                    }

                    if let Some(filter) = &self.spec {
                        let spec_str = format!("{spec_name:?}");
                        if !spec_str.eq_ignore_ascii_case(filter) {
                            continue;
                        }
                    }

                    let spec_id = spec_name.to_spec_id();
                    cfg.set_spec_and_mainnet_gas_params(spec_id);

                    if cfg.spec().is_enabled_in(SI::OSAKA) {
                        cfg.set_max_blobs_per_tx(6);
                    } else if cfg.spec().is_enabled_in(SI::PRAGUE) {
                        cfg.set_max_blobs_per_tx(9);
                    } else {
                        cfg.set_max_blobs_per_tx(6);
                    }

                    let block = unit.block_env(&mut cfg);

                    let compiled = match cache.compile(unit, spec_id) {
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
                            "\nMISMATCH [{name}] spec={spec_name:?} idx={idx} d={} g={} v={} file={}",
                            test.indexes.data,
                            test.indexes.gas,
                            test.indexes.value,
                            test_path.display(),
                        );
                        for m in &mismatches {
                            println!("{m}");
                        }

                        println!("\n--- Interpreter post-state ---");
                        println!("{}", interp.post_state_dump);
                        println!("--- JIT post-state ---");
                        println!("{}", jit.post_state_dump);

                        if self.trace {
                            eprintln!("\n=== Interpreter trace ===");
                            diagnostic::trace_interpreter(&cfg, &block, &tx, &cache_state);
                        }

                        if !self.keep_going {
                            println!("\nStopping at first mismatch. Use --keep-going to continue.");
                            std::process::exit(1);
                        }
                    }
                }
            }
        }

        println!("\n--- Summary ---");
        println!("Files processed: {n_files} ({n_skipped} skipped)");
        println!("Total test cases: {n_total}");
        println!("Mismatches: {n_mismatches}");
        if n_compile_errors > 0 {
            println!("Compile errors: {n_compile_errors}");
        }
        if n_mismatches > 0 {
            std::process::exit(1);
        }
        Ok(())
    }
}

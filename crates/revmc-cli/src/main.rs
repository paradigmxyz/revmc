#![allow(missing_docs)]

use clap::{Parser, Subcommand, ValueEnum};
use color_eyre::{eyre::eyre, Result};
use revm_bytecode::Bytecode;
use revm_interpreter::{
    host::DummyHost, instruction_table, interpreter::ExtBytecode, InputsImpl, SharedMemory,
};
use revmc::{
    eyre::ensure, primitives::hardfork::SpecId, EvmCompiler, EvmContext, EvmLlvmBackend,
    OptimizationLevel,
};
use revmc_cli::{get_benches, read_code, Bench};
use std::{
    hint::black_box,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(name = "revmc-cli")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile and/or run EVM bytecode.
    Run(RunArgs),
    /// Compare interpreter vs JIT execution of Ethereum state tests.
    StatetestDiff(StatetestDiffArgs),
}

#[derive(Parser)]
struct RunArgs {
    /// Benchmark name, "custom", path to a file, or a symbol to load from a shared object.
    bench_name: String,
    #[arg(default_value = "1")]
    n_iters: u64,

    #[arg(long)]
    code: Option<String>,
    #[arg(long, conflicts_with = "code")]
    code_path: Option<PathBuf>,
    #[arg(long)]
    calldata: Option<String>,

    /// Load a shared object file instead of JIT compiling.
    ///
    /// Use with `--aot` to also run the compiled library.
    #[arg(long)]
    load: Option<Option<PathBuf>>,

    /// Parse the bytecode only.
    #[arg(long)]
    parse_only: bool,

    /// Compile and link to a shared library.
    #[arg(long)]
    aot: bool,

    /// Interpret the code instead of compiling.
    #[arg(long, conflicts_with = "aot")]
    interpret: bool,

    /// Target triple.
    #[arg(long, default_value = "native")]
    target: String,
    /// Target CPU.
    #[arg(long)]
    target_cpu: Option<String>,
    /// Target features.
    #[arg(long)]
    target_features: Option<String>,

    /// Compile only, do not link.
    #[arg(long, requires = "aot")]
    no_link: bool,

    #[arg(short = 'o', long)]
    out_dir: Option<PathBuf>,
    #[arg(short = 'O', long, default_value = "3")]
    opt_level: OptimizationLevel,
    #[arg(long, value_enum, default_value = "osaka")]
    spec_id: SpecIdValueEnum,
    #[arg(long)]
    debug_assertions: bool,
    #[arg(long)]
    no_gas: bool,
    #[arg(long)]
    no_len_checks: bool,
    #[arg(long, default_value = "1000000000")]
    gas_limit: u64,
}

#[derive(Parser)]
struct StatetestDiffArgs {
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

fn main() -> Result<()> {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let _ = color_eyre::install();
    let _ = init_tracing_subscriber();

    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => cmd_run(args),
        Command::StatetestDiff(args) => cmd_statetest_diff(args),
    }
}

fn cmd_run(cli: RunArgs) -> Result<()> {
    // Build the compiler.
    let context = revmc::llvm::inkwell::context::Context::create();
    let target = revmc::Target::new(cli.target, cli.target_cpu, cli.target_features);
    let backend = EvmLlvmBackend::new_for_target(&context, cli.aot, cli.opt_level, &target)?;
    let mut compiler = EvmCompiler::new(backend);
    compiler.set_dump_to(cli.out_dir);
    compiler.gas_metering(!cli.no_gas);
    unsafe { compiler.stack_bound_checks(!cli.no_len_checks) };
    compiler.frame_pointers(true);
    compiler.debug_assertions(cli.debug_assertions);

    let Bench { name, bytecode, calldata, stack_input, native: _, requires_storage: _ } =
        if cli.bench_name == "custom" {
            Bench {
                name: "custom",
                bytecode: read_code(cli.code.as_deref(), cli.code_path.as_deref())?,
                ..Default::default()
            }
        } else if Path::new(&cli.bench_name).exists() {
            let path = Path::new(&cli.bench_name);
            ensure!(path.is_file(), "argument must be a file");
            ensure!(cli.code.is_none(), "--code is not allowed with a file argument");
            ensure!(cli.code_path.is_none(), "--code-path is not allowed with a file argument");
            Bench {
                name: path.file_stem().unwrap().to_str().unwrap().to_string().leak(),
                bytecode: read_code(None, Some(path))?,
                ..Default::default()
            }
        } else {
            match get_benches().into_iter().find(|b| b.name == cli.bench_name) {
                Some(b) => b,
                None => {
                    if cli.load.is_some() {
                        Bench {
                            name: cli.bench_name.clone().leak(),
                            bytecode: Vec::new(),
                            ..Default::default()
                        }
                    } else {
                        return Err(eyre!("unknown benchmark: {}", cli.bench_name));
                    }
                }
            }
        };
    compiler.set_module_name(name);

    let calldata: revmc::primitives::Bytes = if let Some(calldata) = cli.calldata {
        revmc::primitives::hex::decode(calldata)?.into()
    } else {
        calldata.into()
    };
    let gas_limit = cli.gas_limit;

    let spec_id = cli.spec_id.into();

    let bytecode_raw = Bytecode::new_raw(revmc::primitives::Bytes::copy_from_slice(&bytecode));
    let bytecode_slice = bytecode_raw.original_byte_slice();

    let mut host = DummyHost::new(spec_id);

    if !stack_input.is_empty() {
        compiler.inspect_stack_length(true);
    }

    if cli.parse_only {
        let _ = compiler.parse(bytecode_slice.into(), spec_id)?;
        return Ok(());
    }

    let f_id = compiler.translate(name, bytecode_slice, spec_id)?;

    let mut load = cli.load;
    if cli.aot {
        let out_dir = if let Some(out_dir) = compiler.out_dir() {
            out_dir.join(&cli.bench_name)
        } else {
            let dir = std::env::temp_dir().join("revmc-cli").join(&cli.bench_name);
            std::fs::create_dir_all(&dir)?;
            dir
        };

        // Compile.
        let obj = out_dir.join("a.o");
        compiler.write_object_to_file(&obj)?;
        ensure!(obj.exists(), "Failed to write object file");
        eprintln!("Compiled object file to {}", obj.display());

        // Link.
        if !cli.no_link {
            let so = out_dir.join("a.so");
            let linker = revmc::Linker::new();
            linker.link(&so, [obj.to_str().unwrap()])?;
            ensure!(so.exists(), "Failed to link object file");
            eprintln!("Linked shared object file to {}", so.display());
        }

        // Fall through to loading the library below if requested.
        if let Some(load @ None) = &mut load {
            *load = Some(out_dir.join("a.so"));
        } else {
            return Ok(());
        }
    }

    let lib;
    let f = if let Some(load) = load {
        if let Some(load) = load {
            lib = unsafe { libloading::Library::new(load) }?;
            let f: libloading::Symbol<'_, revmc::EvmCompilerFn> =
                unsafe { lib.get(name.as_bytes())? };
            *f
        } else {
            return Err(eyre!("--load with no argument requires --aot"));
        }
    } else {
        unsafe { compiler.jit_function(f_id)? }
    };

    use revm_interpreter::interpreter::EthInterpreter;
    let table = instruction_table::<EthInterpreter, DummyHost>();
    let mut run = |f: revmc::EvmCompilerFn| {
        let ext_bytecode = ExtBytecode::new(bytecode_raw.clone());
        let input = InputsImpl {
            input: revm_interpreter::CallInput::Bytes(calldata.clone()),
            ..Default::default()
        };
        let mut interpreter = revm_interpreter::Interpreter::new(
            SharedMemory::new(),
            ext_bytecode,
            input,
            false,
            spec_id,
            gas_limit,
        );

        if cli.interpret {
            let action = interpreter.run_plain(&table, &mut host);
            let result =
                action.instruction_result().unwrap_or(revm_interpreter::InstructionResult::Stop);
            (result, action)
        } else {
            let (mut ecx, stack, stack_len) =
                EvmContext::from_interpreter_with_stack(&mut interpreter, &mut host);

            for (i, input) in stack_input.iter().enumerate() {
                stack.as_mut_slice()[i] = (*input).into();
            }
            *stack_len = stack_input.len();

            let r = unsafe { f.call_noinline(Some(stack), Some(stack_len), &mut ecx) };
            // The JIT code may not set an action (e.g., for STOP), so we need to handle that.
            // If action is None, create a default action based on the return result.
            let action = ecx.next_action.take().unwrap_or_else(|| {
                revm_interpreter::InterpreterAction::Return(revm_interpreter::InterpreterResult {
                    result: r,
                    output: revm_primitives::Bytes::new(),
                    gas: *ecx.gas,
                })
            });
            (r, action)
        }
    };

    if cli.n_iters == 0 {
        return Ok(());
    }

    let (ret, action) = run(f);
    println!("InstructionResult::{ret:?}");
    println!("InterpreterAction::{action:#?}");

    if cli.n_iters > 1 {
        bench(cli.n_iters, name, || run(f));
        return Ok(());
    }

    Ok(())
}

fn cmd_statetest_diff(cli: StatetestDiffArgs) -> Result<()> {
    use revm::{
        context::cfg::CfgEnv,
        primitives::{hardfork::SpecId as SI, U256},
        statetest_types::{SpecName, TestSuite},
    };
    use revmc_statetest::{
        compiled::{CompileCache, CompileMode},
        diagnostic,
        runner::skip_test,
    };

    if !cli.path.exists() {
        return Err(eyre!("Path not found: {}", cli.path.display()));
    }

    let test_files = revmc_statetest::find_all_json_tests(&cli.path);
    if test_files.is_empty() {
        return Err(eyre!("No JSON test files found in {}", cli.path.display()));
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

fn bench<T>(n_iters: u64, name: &str, mut f: impl FnMut() -> T) {
    let warmup = (n_iters / 10).max(10);
    for _ in 0..warmup {
        black_box(f());
    }

    let t = std::time::Instant::now();
    for _ in 0..n_iters {
        black_box(f());
    }
    let d = t.elapsed();
    eprintln!("{name}: {:>9?} ({d:>12?} / {n_iters})", d / n_iters as u32);
}

fn init_tracing_subscriber() -> Result<(), tracing_subscriber::util::TryInitError> {
    use tracing_subscriber::prelude::*;
    let registry = tracing_subscriber::Registry::default()
        .with(tracing_subscriber::EnvFilter::from_default_env());
    #[cfg(feature = "tracy")]
    let registry = registry.with(tracing_tracy::TracyLayer::default());
    registry.with(tracing_subscriber::fmt::layer()).try_init()
}

#[derive(Clone, Copy, Debug, ValueEnum)]
#[clap(rename_all = "lowercase")]
#[allow(non_camel_case_types)]
pub enum SpecIdValueEnum {
    FRONTIER,
    FRONTIER_THAWING,
    HOMESTEAD,
    DAO_FORK,
    TANGERINE,
    SPURIOUS_DRAGON,
    BYZANTIUM,
    CONSTANTINOPLE,
    PETERSBURG,
    ISTANBUL,
    MUIR_GLACIER,
    BERLIN,
    LONDON,
    ARROW_GLACIER,
    GRAY_GLACIER,
    MERGE,
    SHANGHAI,
    CANCUN,
    PRAGUE,
    OSAKA,
    LATEST,
}

impl From<SpecIdValueEnum> for SpecId {
    fn from(v: SpecIdValueEnum) -> Self {
        match v {
            SpecIdValueEnum::FRONTIER => Self::FRONTIER,
            SpecIdValueEnum::FRONTIER_THAWING => Self::FRONTIER_THAWING,
            SpecIdValueEnum::HOMESTEAD => Self::HOMESTEAD,
            SpecIdValueEnum::DAO_FORK => Self::DAO_FORK,
            SpecIdValueEnum::TANGERINE => Self::TANGERINE,
            SpecIdValueEnum::SPURIOUS_DRAGON => Self::SPURIOUS_DRAGON,
            SpecIdValueEnum::BYZANTIUM => Self::BYZANTIUM,
            SpecIdValueEnum::CONSTANTINOPLE => Self::CONSTANTINOPLE,
            SpecIdValueEnum::PETERSBURG => Self::PETERSBURG,
            SpecIdValueEnum::ISTANBUL => Self::ISTANBUL,
            SpecIdValueEnum::MUIR_GLACIER => Self::MUIR_GLACIER,
            SpecIdValueEnum::BERLIN => Self::BERLIN,
            SpecIdValueEnum::LONDON => Self::LONDON,
            SpecIdValueEnum::ARROW_GLACIER => Self::ARROW_GLACIER,
            SpecIdValueEnum::GRAY_GLACIER => Self::GRAY_GLACIER,
            SpecIdValueEnum::MERGE => Self::MERGE,
            SpecIdValueEnum::SHANGHAI => Self::SHANGHAI,
            SpecIdValueEnum::CANCUN => Self::CANCUN,
            SpecIdValueEnum::PRAGUE => Self::PRAGUE,
            SpecIdValueEnum::OSAKA => Self::OSAKA,
            SpecIdValueEnum::LATEST => Self::OSAKA,
        }
    }
}

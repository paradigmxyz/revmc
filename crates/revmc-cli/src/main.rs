#![allow(missing_docs)]

use clap::{Parser, ValueEnum};
use color_eyre::{eyre::eyre, Result};
use revm_interpreter::{opcode::make_instruction_table, SharedMemory};
use revm_primitives::{address, spec_to_generic, Env, SpecId, TransactTo};
use revmc::{eyre::ensure, EvmCompiler, EvmContext, EvmLlvmBackend, OptimizationLevel};
use revmc_cli::{get_benches, read_code, Bench};
use std::{
    hint::black_box,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Cli {
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
    #[arg(long, value_enum, default_value = "pragueeof")]
    spec_id: SpecIdValueEnum,
    /// Short-hand for `--spec-id pragueeof`.
    #[arg(long, conflicts_with = "spec_id")]
    eof: bool,
    /// Skip validating EOF code.
    #[arg(long, requires = "eof")]
    no_validate: bool,
    #[arg(long)]
    debug_assertions: bool,
    #[arg(long)]
    no_gas: bool,
    #[arg(long)]
    no_len_checks: bool,
    #[arg(long, default_value = "1000000000")]
    gas_limit: u64,
}

fn main() -> Result<()> {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let _ = color_eyre::install();
    let _ = init_tracing_subscriber();

    let cli = Cli::parse();

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
    compiler.validate_eof(!cli.no_validate);

    let Bench { name, bytecode, calldata, stack_input, native: _ } = if cli.bench_name == "custom" {
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

    let calldata = if let Some(calldata) = cli.calldata {
        revmc::primitives::hex::decode(calldata)?.into()
    } else {
        calldata.into()
    };
    let gas_limit = cli.gas_limit;

    let mut env = Env::default();
    env.tx.caller = address!("0000000000000000000000000000000000000001");
    env.tx.transact_to = TransactTo::Call(address!("0000000000000000000000000000000000000002"));
    env.tx.data = calldata;
    env.tx.gas_limit = gas_limit;

    let bytecode = revm_interpreter::analysis::to_analysed(revm_primitives::Bytecode::new_raw(
        revm_primitives::Bytes::copy_from_slice(&bytecode),
    ));
    let contract = revm_interpreter::Contract::new_env(&env, bytecode, None);
    let mut host = revm_interpreter::DummyHost::new(env);

    let bytecode = contract.bytecode.original_byte_slice();

    let spec_id = if cli.eof { SpecId::PRAGUE_EOF } else { cli.spec_id.into() };
    if !stack_input.is_empty() {
        compiler.inspect_stack_length(true);
    }

    if cli.parse_only {
        let _ = compiler.parse(bytecode.into(), spec_id)?;
        return Ok(());
    }

    let f_id = compiler.translate(name, bytecode, spec_id)?;

    let mut load = cli.load;
    if cli.aot {
        let out_dir = if let Some(out_dir) = compiler.out_dir() {
            out_dir.join(cli.bench_name)
        } else {
            let dir = std::env::temp_dir().join("revmc-cli").join(cli.bench_name);
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

    #[allow(unused_parens)]
    let table = spec_to_generic!(spec_id, (const { &make_instruction_table::<_, SPEC>() }));
    let mut run = |f: revmc::EvmCompilerFn| {
        let mut interpreter =
            revm_interpreter::Interpreter::new(contract.clone(), gas_limit, false);
        host.clear();

        if cli.interpret {
            let action = interpreter.run(SharedMemory::new(), table, &mut host);
            (interpreter.instruction_result, action)
        } else {
            let (mut ecx, stack, stack_len) =
                EvmContext::from_interpreter_with_stack(&mut interpreter, &mut host);

            for (i, input) in stack_input.iter().enumerate() {
                stack.as_mut_slice()[i] = input.into();
            }
            *stack_len = stack_input.len();

            let r = unsafe { f.call_noinline(Some(stack), Some(stack_len), &mut ecx) };
            (r, interpreter.next_action)
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
    PRAGUE_EOF,
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
            SpecIdValueEnum::PRAGUE_EOF => Self::PRAGUE_EOF,
            SpecIdValueEnum::LATEST => Self::LATEST,
        }
    }
}

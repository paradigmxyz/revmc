use clap::Parser;
use color_eyre::{eyre::eyre, Result};
use revm_jit::{
    debug_time, eyre::Context, new_llvm_backend, EvmContext, JitEvm, OptimizationLevel,
};
use revm_jit_benches::Bench;
use revm_primitives::{hex, Env, SpecId};
use std::{
    hint::black_box,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Cli {
    bench_name: String,
    #[arg(default_value = "1")]
    n_iters: u64,

    #[arg(long)]
    code: Option<String>,
    #[arg(long, conflicts_with = "code")]
    code_path: Option<PathBuf>,

    #[arg(short = 'O', long)]
    opt_level: Option<OptimizationLevel>,
    // #[arg(long)]
    #[arg(skip)]
    spec_id: Option<SpecId>,
    #[arg(long)]
    debug_assertions: bool,
    #[arg(long)]
    no_gas: bool,
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
    let opt_level = cli.opt_level.unwrap_or(OptimizationLevel::Aggressive);
    let context = revm_jit::llvm::inkwell::context::Context::create();
    let backend = new_llvm_backend(&context, opt_level)?;
    let mut jit = JitEvm::new(backend);
    jit.set_dump_to(Some(PathBuf::from("./tmp/revm-jit")));
    jit.set_disable_gas(cli.no_gas);
    jit.set_frame_pointers(true);
    jit.set_debug_assertions(cli.debug_assertions);

    let Bench { name, bytecode, calldata, stack_input, native: _ } = if cli.bench_name == "custom" {
        Bench {
            bytecode: read_code(cli.code.as_deref(), cli.code_path.as_deref())?,
            name: "custom",
            ..Default::default()
        }
    } else {
        revm_jit_benches::get_benches()
            .into_iter()
            .find(|b| b.name == cli.bench_name)
            .ok_or_else(|| eyre!("unknown benchmark: {}", cli.bench_name))?
    };

    let gas_limit = cli.gas_limit;

    let mut env = Env::default();
    env.tx.data = calldata.clone().into();
    env.tx.gas_limit = gas_limit;

    let bytecode = revm_interpreter::analysis::to_analysed(revm_primitives::Bytecode::new_raw(
        revm_primitives::Bytes::copy_from_slice(&bytecode),
    ));
    let bytecode_hash = bytecode.hash_slow();
    let contract = revm_interpreter::Contract::new_env(&env, bytecode, bytecode_hash);
    let mut host = revm_interpreter::DummyHost::new(env);

    let bytecode = contract.bytecode.original_bytecode_slice();

    // let table = &revm_interpreter::opcode::make_instruction_table::<
    //     revm_interpreter::DummyHost,
    //     revm_primitives::CancunSpec,
    // >();

    let spec_id = cli.spec_id.unwrap_or(SpecId::LATEST);
    if !stack_input.is_empty() {
        jit.set_inspect_stack_length(true);
    }
    let f = jit.compile(Some(name), bytecode, spec_id)?;

    let mut run = |f: revm_jit::JitEvmFn| {
        let mut interpreter =
            revm_interpreter::Interpreter::new(contract.clone(), gas_limit, false);
        host.clear();
        let (mut ecx, stack, stack_len) =
            EvmContext::from_interpreter_with_stack(&mut interpreter, &mut host);

        for (i, input) in stack_input.iter().enumerate() {
            stack.as_mut_slice()[i] = input.into();
        }
        *stack_len = stack_input.len();

        unsafe { f.call(Some(stack), Some(stack_len), &mut ecx) }
    };

    let ret = debug_time!("run", || run(f));
    if cli.n_iters <= 1 {
        eprintln!("{ret:?}");
    } else {
        assert!(ret.is_ok(), "{ret:?}");
        bench(cli.n_iters, name, || run(f));
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
    tracing_subscriber::Registry::default()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_error::ErrorLayer::default())
        .with(tracing_subscriber::fmt::layer())
        .try_init()
}

fn read_code(code: Option<&str>, code_path: Option<&Path>) -> Result<Vec<u8>> {
    if let Some(code) = code {
        return hex::decode(code).wrap_err("--code is not valid hex");
    }

    if let Some(code_path) = code_path {
        let contents = std::fs::read(code_path)?;
        let ext = code_path.extension().map(|s| s.to_str().unwrap_or(""));
        let has_prefix = contents.starts_with(b"0x") || contents.starts_with(b"0X");
        let is_hex =
            ext != Some("bin") && (ext == Some("hex") || has_prefix || contents.is_ascii());
        return if is_hex {
            let code_s =
                std::str::from_utf8(&contents).wrap_err("--code-path is not valid UTF-8")?;
            hex::decode(code_s).wrap_err("--code-path is not valid hex")
        } else {
            Ok(contents)
        };
    }

    Err(eyre!("--code is required when bench_name is 'custom'"))
}

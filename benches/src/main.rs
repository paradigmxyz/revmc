use clap::Parser;
use color_eyre::Result;
use revm_interpreter::InstructionResult;
use revm_jit::{debug_time, eyre::eyre, EvmContext, EvmStack, JitEvm, OptimizationLevel};
use revm_jit_benches::Bench;
use revm_primitives::{Env, SpecId};
use std::{hint::black_box, path::PathBuf};

#[derive(Parser)]
struct Cli {
    bench_name: String,
    #[arg(default_value = "1")]
    n_iters: u64,
    #[arg(short = 'O', long)]
    opt_level: Option<OptimizationLevel>,
    // #[arg(short, long)]
    #[arg(skip)]
    spec_id: Option<SpecId>,
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
    let backend = revm_jit::llvm::JitEvmLlvmBackend::new(&context, opt_level).unwrap();
    let mut jit = JitEvm::new(backend);
    jit.set_dump_to(Some(PathBuf::from("./tmp/revm-jit")));
    // jit.set_debug_assertions(true);

    let all_benches = revm_jit_benches::get_benches();
    let Bench { name, bytecode, calldata, stack_input, native: _ } = all_benches
        .iter()
        .find(|b| b.name == cli.bench_name)
        .ok_or_else(|| eyre!("unknown benchmark: {}", cli.bench_name))?;

    let mut env = Env::default();
    env.tx.data = calldata.clone().into();

    let bytecode = revm_interpreter::analysis::to_analysed(revm_primitives::Bytecode::new_raw(
        revm_primitives::Bytes::copy_from_slice(bytecode),
    ));
    let bytecode_hash = bytecode.hash_slow();
    let contract = revm_interpreter::Contract::new_env(&env, bytecode, bytecode_hash);
    let mut host = revm_interpreter::DummyHost::new(env);

    let gas_limit = 100_000;
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

    let mut stack = EvmStack::new();
    let mut run = |f: revm_jit::JitEvmFn| {
        for (i, input) in stack_input.iter().enumerate() {
            stack.as_mut_slice()[i] = input.into();
        }
        let mut stack_len = stack_input.len();

        let mut interpreter =
            revm_interpreter::Interpreter::new(contract.clone(), gas_limit, false);
        host.clear();
        let mut cx = EvmContext::from_interpreter(&mut interpreter, &mut host);

        unsafe { f.call(Some(&mut stack), Some(&mut stack_len), &mut cx) }
    };

    let ret = debug_time!("run", || run(f));
    assert!(matches!(ret, InstructionResult::Return | InstructionResult::Stop), "{ret:?}");

    bench(cli.n_iters, name, || run(f));

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

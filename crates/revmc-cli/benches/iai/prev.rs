#![allow(missing_docs)]

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use revm_primitives::{Env, SpecId};
use revmc::{
    llvm::with_llvm_context, Backend, EvmCompiler, EvmContext, EvmLlvmBackend, OptimizationLevel,
};
use revmc_cli::{get_bench, Bench};
use std::{hint::black_box, path::PathBuf};

const SPEC_ID: SpecId = SpecId::CANCUN;
const GAS_LIMIT: u64 = 100_000_000;

fn out_dir() -> PathBuf {
    std::env::temp_dir().join("revmc-cli")
}

fn compile_time_impl(name: &str) {
    with_llvm_context(|cx| {
        let backend = EvmLlvmBackend::new(cx, true, OptimizationLevel::Aggressive).unwrap();
        compile_time_impl_inner(&mut EvmCompiler::new(backend), &get_bench(name).expect(name));
    });
}

fn compile_time_impl_inner<B: Backend>(compiler: &mut EvmCompiler<B>, bench: &Bench) {
    if !bench.stack_input.is_empty() {
        compiler.inspect_stack_length(true);
    }
    let _ = compiler.translate(Some(bench.name), &bench.bytecode, SPEC_ID).unwrap();

    let out_dir = out_dir();
    let obj = out_dir.join(bench.name).with_extension(".o");
    compiler.write_object_to_file(&obj).unwrap();

    // TODO: Clear envs.
    /*
    let so = out_dir.join(bench.name).with_extension(".so");
    Linker::new().link(&so, &[&obj]).unwrap();
    assert!(so.exists());
    */
}

fn run_time_setup(name: &str) -> Box<dyn FnOnce()> {
    with_llvm_context(|cx| {
        let backend = EvmLlvmBackend::new(cx, false, OptimizationLevel::Aggressive).unwrap();
        run_time_setup_inner(&mut EvmCompiler::new(backend), &get_bench(name).expect(name))
    })
}

fn run_time_setup_inner<B: Backend>(
    compiler: &mut EvmCompiler<B>,
    bench: &Bench,
) -> Box<dyn FnOnce()> {
    let Bench { name, bytecode, calldata, stack_input, .. } = bench;

    let stack_input = stack_input.clone();
    if !stack_input.is_empty() {
        compiler.inspect_stack_length(true);
    }
    let f = compiler.jit(Some(name), bytecode, SPEC_ID).unwrap();

    let mut env = Env::default();
    env.tx.data = calldata.clone().into();

    let bytecode = revm_interpreter::analysis::to_analysed(revm_primitives::Bytecode::new_raw(
        revm_primitives::Bytes::copy_from_slice(&bytecode),
    ));
    let contract = revm_interpreter::Contract::new_env(&env, bytecode, None);
    let mut host = revm_interpreter::DummyHost::new(env);

    let mut interpreter = revm_interpreter::Interpreter::new(contract, GAS_LIMIT, false);

    Box::new(move || {
        let (mut ecx, stack, stack_len) =
            EvmContext::from_interpreter_with_stack(&mut interpreter, &mut host);

        for (i, input) in stack_input.iter().enumerate() {
            stack.as_mut_slice()[i] = input.into();
        }
        *stack_len = stack_input.len();

        let r = unsafe { f.call_noinline(Some(stack), Some(stack_len), &mut ecx) };
        black_box(r);
    })
}

macro_rules! make_benchmarks {
    ($($name:ident),*) => {
        #[library_benchmark]
        $(
            #[bench::$name(stringify!($name))]
        )*
        fn compile_time(name: &str) {
            crate::compile_time_impl(name);
        }

        #[library_benchmark]
        $(
            #[bench::$name(run_time_setup(stringify!($name)))]
        )*
        fn run_time(f: Box<dyn FnOnce()>) {
            f();
        }

        library_benchmark_group!(
            name = all;
            benchmarks =
                // compile_time,
                run_time
        );

        main!(library_benchmark_groups = all);
    };
}

make_benchmarks!(fibonacci, counter, snailtracer, push0_proxy, weth, hash_10k);

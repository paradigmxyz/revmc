#![allow(missing_docs)]

use iai_callgrind::{
    binary_benchmark_group, main, Arg, BinaryBenchmarkGroup, EventKind, FlamegraphConfig,
    RegressionConfig, Run,
};

const CMD: &str = env!("CARGO_BIN_EXE_revm-jit-cli");

binary_benchmark_group!(
    name = compile_time;
    benchmark = |group: &mut BinaryBenchmarkGroup| setup_group(group, true)
);

binary_benchmark_group!(
    name = run_time;
    benchmark = |group: &mut BinaryBenchmarkGroup| setup_group(group, false)
);

fn setup_group(group: &mut BinaryBenchmarkGroup, is_ct: bool) {
    let make_run = |name: &str| {
        let mut args = Vec::with_capacity(3);
        args.push(name);
        if is_ct {
            args.extend(["--aot", "--no-link"]);
        } else {
            args.push("1");
        }
        let arg = Arg::new(name, args);
        let mut run = Run::with_cmd(CMD, arg);

        if !is_ct {
            run.entry_point("*EvmCompilerFn::call*");
        }

        let mut regression = RegressionConfig::default();
        if is_ct {
            regression.limits([(EventKind::EstimatedCycles, 10.0)]);
        } else {
            regression.limits([(EventKind::EstimatedCycles, 5.0)]);
        }
        run.regression(regression);

        if !is_ci() {
            let flamegraph = FlamegraphConfig::default();
            run.flamegraph(flamegraph);
        }

        run
    };
    let benches = ["fibonacci", "counter", "push0_proxy", "weth", "hash_20k", "usdc_proxy"];
    for bench in &benches {
        group.bench(make_run(bench));
    }
}

fn is_ci() -> bool {
    std::env::var_os("CI").is_some()
}

main!(binary_benchmark_groups = compile_time, run_time);

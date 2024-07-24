#![allow(missing_docs)]

use iai_callgrind::{
    binary_benchmark_group, main, Arg, BinaryBenchmarkGroup, EventKind, FlamegraphConfig,
    RegressionConfig, Run,
};

const CMD: &str = env!("CARGO_BIN_EXE_revmc-cli");

binary_benchmark_group!(
    name = compile_time;
    benchmark = |group: &mut BinaryBenchmarkGroup| setup_group(group, true)
);

binary_benchmark_group!(
    name = run_time;
    benchmark = |group: &mut BinaryBenchmarkGroup| setup_group(group, false)
);

fn setup_group(group: &mut BinaryBenchmarkGroup, is_ct: bool) {
    let make_run = |name: &str, small: bool| {
        let mut args = Vec::with_capacity(3);
        args.push(name);
        // let out_dir = std::env::temp_dir().join("revmc-cli-iai");
        // let so = out_dir.join(name).join("a.so");
        if is_ct {
            args.extend(["--aot", "--no-link"]);
            // args.extend(["--aot", "--no-link", "-o", out_dir.to_str().unwrap()]);
        } else {
            args.push("1");
            // args.extend(["1", "--shared-library", so.to_str().unwrap()]);
        }
        let arg = Arg::new(name, args);
        let mut run = Run::with_cmd(CMD, arg);

        if !is_ct {
            run.entry_point("*EvmCompilerFn::call*");
        }

        let mut regression = RegressionConfig::default();
        if is_ct {
            let cycles = if small { 50.0 } else { 20.0 };
            regression.limits([(EventKind::EstimatedCycles, cycles)]);
        } else {
            regression.limits([(EventKind::EstimatedCycles, 5.0)]);
        }
        run.regression(regression);

        // Uses an insane amount of memory (???)
        if cfg!(any()) && small && !is_ci() {
            let flamegraph = FlamegraphConfig::default();
            run.flamegraph(flamegraph);
        }

        run
    };
    let benches = [
        // ("fibonacci", true),
        // ("counter", true),
        // ("hash_10k", true),
        // ("bswap64", true),
        // ("usdc_proxy", false),
        // ("weth", false),
        ("hash_10k", true),
        ("hash_10k-eof", true),
        ("snailtracer", true),
        ("snailtracer-eof", true),
    ];
    for (bench, small) in benches {
        if !is_ct && !small {
            continue;
        }
        group.bench(make_run(bench, small));
    }
}

fn is_ci() -> bool {
    std::env::var_os("CI").is_some()
}

main!(binary_benchmark_groups = compile_time, run_time);

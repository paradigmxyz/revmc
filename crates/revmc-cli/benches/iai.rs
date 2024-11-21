#![allow(missing_docs)]

use iai_callgrind::{
    binary_benchmark_group, main, Bench, BinaryBenchmark, BinaryBenchmarkConfig,
    BinaryBenchmarkGroup, Command, EventKind, FlamegraphConfig, RegressionConfig,
};

const CMD: &str = env!("CARGO_BIN_EXE_revmc-cli");

binary_benchmark_group!(
    name = revmc;
    benchmarks = |group: &mut BinaryBenchmarkGroup| setup_group(group)
);

fn setup_group(group: &mut BinaryBenchmarkGroup) {
    let make_bench = |name: &str, small: bool, is_ct: bool| {
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
        let mut bench = Bench::new(name);
        bench.command(Command::new(CMD).args(&args)).config(BinaryBenchmarkConfig::default());

        if !is_ct {
            bench.config.as_mut().unwrap().entry_point = Some("*EvmCompilerFn::call*".into());
        }

        let mut regression = RegressionConfig::default();
        if is_ct {
            let cycles = if small { 50.0 } else { 20.0 };
            regression.limits([(EventKind::EstimatedCycles, cycles)]);
        } else {
            regression.limits([(EventKind::EstimatedCycles, 5.0)]);
        }
        bench.config.as_mut().unwrap().regression_config = Some(regression.into());

        // Uses an insane amount of memory (???)
        if cfg!(any()) && small && !is_ci() {
            let flamegraph = FlamegraphConfig::default();
            bench.config.as_mut().unwrap().flamegraph_config = Some(flamegraph.into());
        }

        bench
    };
    let benches = [
        ("fibonacci", true),
        ("counter", true),
        ("hash_10k", true),
        ("hash_10k-eof", true),
        ("bswap64", true),
        ("usdc_proxy", false),
        ("weth", false),
        // ("snailtracer", false),
        // ("snailtracer-eof", false),
    ];
    for is_ct in [false, true] {
        let mut bench = BinaryBenchmark::new(if is_ct { "compile_time" } else { "run_time" });
        for (bench_name, small) in benches {
            if !is_ct && !small {
                continue;
            }
            bench.bench(make_bench(bench_name, small, is_ct));
        }
        group.binary_benchmark(bench);
    }
}

fn is_ci() -> bool {
    std::env::var_os("CI").is_some()
}

main!(binary_benchmark_groups = revmc);

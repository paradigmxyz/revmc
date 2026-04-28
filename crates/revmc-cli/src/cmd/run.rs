use clap::{Parser, ValueEnum};
use color_eyre::{Result, eyre::eyre};
use revmc::{
    EvmCompiler, OptimizationLevel, eyre::ensure, primitives::hardfork::SpecId, shared_library_path,
};
use revmc_cli::{Bench, PreparedBench, get_benches, read_code};
use std::{
    hint::black_box,
    path::{Path, PathBuf},
};

#[derive(Parser)]
pub(crate) struct RunArgs {
    /// Benchmark name, "custom", path to a file, or a symbol to load from a shared object.
    ///
    /// Use `--list` to see all available benchmark names.
    bench_name: Option<String>,
    #[arg(default_value = "1")]
    n_iters: u64,

    /// List available benchmark names and exit.
    #[arg(long)]
    list: bool,

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

    /// Print the parsed bytecode IR.
    #[arg(long)]
    display: bool,

    /// Parse the bytecode and render the CFG as a DOT graph.
    #[arg(long, default_missing_value = "svg", num_args = 0..=1)]
    dot: Option<DotFormat>,

    /// Don't open URLs in the browser.
    #[arg(long)]
    no_open: bool,

    /// Compile and link to a shared library.
    #[arg(long)]
    aot: bool,

    /// Interpret the code instead of compiling.
    #[arg(long, conflicts_with = "aot")]
    interpret: bool,

    /// Run JIT only (skip interpreter comparison in benchmarks).
    #[arg(long, conflicts_with = "interpret")]
    jit_only: bool,

    /// Compile only, do not link.
    #[arg(long, requires = "aot")]
    no_link: bool,

    #[arg(short = 'o', long)]
    out_dir: Option<PathBuf>,
    #[arg(short = 'O', long, default_value = "2")]
    opt_level: OptimizationLevel,
    #[arg(long, value_enum, default_value = "osaka")]
    spec_id: SpecIdValueEnum,
    #[arg(long)]
    debug_assertions: bool,
    #[arg(long)]
    no_gas: bool,
    #[arg(long)]
    no_len_checks: bool,
    /// Force all failure paths to yield a single `OutOfGas`.
    #[arg(long)]
    single_error: bool,
    /// Inspect the stack after the function has been executed.
    #[arg(long)]
    inspect_stack: bool,
    #[arg(long, default_value = "1000000000")]
    gas_limit: u64,
}

impl RunArgs {
    pub(crate) fn run(self) -> Result<()> {
        if self.list {
            for b in get_benches() {
                println!("{}", b.name);
            }
            return Ok(());
        }

        let Some(bench_name) = self.bench_name.clone() else {
            return Err(eyre!("missing <BENCH_NAME>; use `--list` to see available benchmarks"));
        };

        // Resolve bench entry first (before any partial moves of self).
        let bench_entry = if bench_name == "custom" {
            Bench {
                name: "custom",
                bytecode: read_code(self.code.as_deref(), self.code_path.as_deref())?,
                ..Default::default()
            }
        } else if Path::new(&bench_name).exists() {
            let path = Path::new(&bench_name);
            ensure!(path.is_file(), "argument must be a file");
            ensure!(self.code.is_none(), "--code is not allowed with a file argument");
            ensure!(self.code_path.is_none(), "--code-path is not allowed with a file argument");
            Bench {
                name: path.file_stem().unwrap().to_str().unwrap().to_string().leak(),
                bytecode: read_code(None, Some(path))?,
                ..Default::default()
            }
        } else {
            match get_benches().into_iter().find(|b| b.name == bench_name) {
                Some(b) => b,
                None => return Err(eyre!("unknown benchmark: {bench_name}")),
            }
        };

        let name = bench_entry.name;
        let is_fixture = bench_entry.is_fixture();
        let default_aot_dir = || std::env::temp_dir().join("revmc-cli").join(&bench_name);
        let mut aot_dir = None;

        // Compile the entry-point contract: parse, display, dot, dump, aot.
        // For fixture benchmarks, extract the tx-target bytecode.
        {
            let bytecode;
            let compile_spec_id: SpecId;
            if is_fixture {
                bytecode = fixture_entry_bytecode(bench_entry.fixture_json.unwrap());
                compile_spec_id = bench_entry.spec_id.unwrap();
            } else {
                bytecode = bench_entry.bytecode.clone();
                compile_spec_id = self.spec_id.into();
            };

            let mut compiler = EvmCompiler::new_llvm(self.aot)?;
            compiler.set_opt_level(self.opt_level);
            let out_dir = if self.out_dir.is_some() {
                self.out_dir.clone()
            } else if self.dot.is_some() || self.display || self.parse_only {
                Some(std::env::temp_dir().join("revmc-cli"))
            } else {
                None
            };
            compiler.set_dump_to(out_dir);
            compiler.gas_metering(!self.no_gas);
            unsafe { compiler.stack_bound_checks(!self.no_len_checks) };
            compiler.debug_assertions(self.debug_assertions);
            if self.single_error {
                compiler.single_error(true);
            }

            compiler.set_module_name(name);
            if let Some(dump_dir) = compiler.dump_dir() {
                eprintln!("Dump directory: {}", dump_dir.display());
            }

            compiler.inspect_stack(self.inspect_stack);

            let parsed = compiler.parse(bytecode.as_slice().into(), compile_spec_id)?;
            if self.display || self.parse_only {
                println!("{name}()\n{parsed:#}");
            }
            if let Some(fmt) = self.dot {
                let dump_dir =
                    compiler.dump_dir().expect("dump_dir should be set when --dot is used");
                open_dot(&dump_dir.join("bytecode.dot"), fmt, !self.no_open)?;
            }
            if self.parse_only {
                return Ok(());
            }

            let f_id = compiler.translate_inner(name, &parsed)?;

            // Finalize the module (verify + optimize + dump IR/asm).
            if !self.aot {
                let _ = unsafe { compiler.jit_function(f_id)? };
            }

            if self.aot {
                let out_dir = if let Some(out_dir) = compiler.out_dir() {
                    out_dir.join(&bench_name)
                } else {
                    let dir = default_aot_dir();
                    std::fs::create_dir_all(&dir)?;
                    dir
                };

                let obj = out_dir.join("a.o");
                compiler.write_object_to_file(&obj)?;
                ensure!(obj.exists(), "Failed to write object file");
                eprintln!("Compiled object file to {}", obj.display());

                if !self.no_link {
                    let shared_lib = shared_library_path(&out_dir, "a");
                    let linker = revmc::Linker::new();
                    linker.link(&shared_lib, [obj.to_str().unwrap()])?;
                    ensure!(shared_lib.exists(), "Failed to link object file");
                    eprintln!("Linked shared object file to {}", shared_lib.display());
                }

                aot_dir = Some(out_dir);
                if self.load.is_none() {
                    return Ok(());
                }
            }
        }

        // Unified runtime path for both bytecode, fixture, and loaded benchmarks.
        let spec_id: SpecId = self.spec_id.into();
        let load_path = self.load.as_ref().map(|load| match load {
            Some(p) => p.clone(),
            None => {
                let out_dir = aot_dir.unwrap_or_else(&default_aot_dir);
                shared_library_path(&out_dir, "a")
            }
        });
        let (prepared, _compiler, _lib);
        if let Some(ref load_path) = load_path {
            let lib;
            (prepared, lib) =
                PreparedBench::load_from_library(&bench_entry, spec_id, load_path, name);
            _compiler = None;
            _lib = Some(lib);
        } else {
            let compiler;
            (prepared, compiler) = PreparedBench::load(&bench_entry, spec_id);
            _compiler = Some(compiler);
            _lib = None;
        };

        prepared.sanity_check();

        if self.n_iters == 0 {
            return Ok(());
        }

        if self.interpret {
            prepared.run_interpreter();
        } else {
            prepared.run_jit();
        }

        if self.n_iters > 1 {
            if self.interpret || !self.jit_only {
                bench(self.n_iters, &format!("{name}/interpreter"), || prepared.run_interpreter());
            }
            if !self.interpret {
                bench(self.n_iters, &format!("{name}/jit"), || prepared.run_jit());
            }
        }

        Ok(())
    }
}

/// Extract the entry-point contract's bytecode from a fixture JSON.
fn fixture_entry_bytecode(json: &str) -> Vec<u8> {
    let v: serde_json::Value = serde_json::from_str(json).expect("invalid fixture JSON");
    let case = v.as_object().unwrap().values().next().unwrap();
    let to = case["transaction"][0]["to"].as_str().expect("fixture missing transaction.to");
    let code = case["pre"][to]["code"].as_str().expect("fixture missing entry-point code");
    let hex = code.strip_prefix("0x").unwrap_or(code);
    revmc::primitives::hex::decode(hex).expect("invalid hex in fixture code")
}

fn open_dot(dot_path: &Path, fmt: DotFormat, open: bool) -> Result<()> {
    let ext = fmt.extension();
    let out_path = dot_path.with_extension(ext);
    match std::process::Command::new("dot")
        .arg(format!("-T{ext}"))
        .arg("-o")
        .arg(&out_path)
        .arg(dot_path)
        .status()
    {
        Ok(status) if status.success() => {
            eprintln!("DOT graph: {}", out_path.display());
            if open {
                let _ = open::that(out_path.as_os_str());
            }
            return Ok(());
        }
        Ok(status) => eprintln!("warning: dot command failed with {status}, falling back to HTML"),
        Err(e) => eprintln!("warning: dot command not found ({e}), falling back to HTML"),
    }

    // Fallback: write an HTML file that renders the DOT graph client-side.
    let dot_source = std::fs::read_to_string(dot_path)?;
    let dot_escaped = dot_source.replace('\\', "\\\\").replace('`', "\\`").replace("${", "\\${");
    let html_path = dot_path.with_extension("html");
    std::fs::write(
        &html_path,
        format!(
            r#"<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>revmc CFG</title>
<style>
html,body{{margin:0;height:100%;overflow:hidden;background:#1a1a2e}}
#graph{{width:100%;height:100%}}
</style>
</head><body>
<div id="graph"></div>
<script type="module">
import {{ instance }} from "https://cdn.jsdelivr.net/npm/@viz-js/viz@3/+esm";
import svgPanZoom from "https://cdn.jsdelivr.net/npm/svg-pan-zoom@3/+esm";
const viz = await instance();
const svg = viz.renderSVGElement(`{dot_escaped}`);
svg.setAttribute("width", "100%");
svg.setAttribute("height", "100%");
document.getElementById("graph").appendChild(svg);
svgPanZoom(svg, {{zoomScaleSensitivity:0.3, minZoom:0.1, maxZoom:50, controlIconsEnabled:true}});
</script>
</body></html>"#
        ),
    )?;
    eprintln!("DOT graph: {}", html_path.display());
    if open {
        let _ = open::that(html_path.as_os_str());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DotFormat {
    Svg,
    Png,
}

impl DotFormat {
    fn extension(self) -> &'static str {
        match self {
            Self::Svg => "svg",
            Self::Png => "png",
        }
    }
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

#[derive(Clone, Copy, Debug, ValueEnum)]
#[clap(rename_all = "lowercase")]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
enum SpecIdValueEnum {
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
    AMSTERDAM,
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
            SpecIdValueEnum::AMSTERDAM => Self::AMSTERDAM,
            SpecIdValueEnum::LATEST => Self::OSAKA,
        }
    }
}

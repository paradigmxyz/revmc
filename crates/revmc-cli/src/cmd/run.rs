use clap::{Parser, ValueEnum};
use color_eyre::{Result, eyre::eyre};
use revm_bytecode::Bytecode;
use revm_interpreter::{
    InputsImpl, SharedMemory,
    host::DummyHost,
    instruction_table,
    interpreter::{EthInterpreter, ExtBytecode},
};
use revmc::{
    EvmCompiler, EvmContext, EvmLlvmBackend, OptimizationLevel, eyre::ensure,
    primitives::hardfork::SpecId,
};
use revmc_cli::{Bench, get_benches, read_code};
use std::{
    hint::black_box,
    path::{Path, PathBuf},
};

#[derive(Parser)]
pub(crate) struct RunArgs {
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

impl RunArgs {
    pub(crate) fn run(self) -> Result<()> {
        // Build the compiler.
        let target = revmc::Target::new(self.target, self.target_cpu, self.target_features);
        let backend = EvmLlvmBackend::new_for_target(self.aot, self.opt_level, &target)?;
        let mut compiler = EvmCompiler::new(backend);
        compiler.set_dump_to(self.out_dir);
        compiler.gas_metering(!self.no_gas);
        unsafe { compiler.stack_bound_checks(!self.no_len_checks) };
        compiler.frame_pointers(true);
        compiler.debug_assertions(self.debug_assertions);

        let Bench { name, bytecode, calldata, stack_input, native: _, requires_storage: _ } =
            if self.bench_name == "custom" {
                Bench {
                    name: "custom",
                    bytecode: read_code(self.code.as_deref(), self.code_path.as_deref())?,
                    ..Default::default()
                }
            } else if Path::new(&self.bench_name).exists() {
                let path = Path::new(&self.bench_name);
                ensure!(path.is_file(), "argument must be a file");
                ensure!(self.code.is_none(), "--code is not allowed with a file argument");
                ensure!(
                    self.code_path.is_none(),
                    "--code-path is not allowed with a file argument"
                );
                Bench {
                    name: path.file_stem().unwrap().to_str().unwrap().to_string().leak(),
                    bytecode: read_code(None, Some(path))?,
                    ..Default::default()
                }
            } else {
                match get_benches().into_iter().find(|b| b.name == self.bench_name) {
                    Some(b) => b,
                    None => {
                        if self.load.is_some() {
                            Bench {
                                name: self.bench_name.clone().leak(),
                                bytecode: Vec::new(),
                                ..Default::default()
                            }
                        } else {
                            return Err(eyre!("unknown benchmark: {}", self.bench_name));
                        }
                    }
                }
            };
        compiler.set_module_name(name);

        let calldata: revmc::primitives::Bytes = if let Some(calldata) = self.calldata {
            revmc::primitives::hex::decode(calldata)?.into()
        } else {
            calldata.into()
        };
        let gas_limit = self.gas_limit;

        let spec_id = self.spec_id.into();

        let bytecode_raw = Bytecode::new_raw(revmc::primitives::Bytes::copy_from_slice(&bytecode));
        let bytecode_slice = bytecode_raw.original_byte_slice();

        let mut host = DummyHost::new(spec_id);

        if !stack_input.is_empty() {
            compiler.inspect_stack_length(true);
        }

        if self.parse_only {
            let _ = compiler.parse(bytecode_slice.into(), spec_id)?;
            return Ok(());
        }

        let f_id = compiler.translate(name, bytecode_slice, spec_id)?;

        let mut load = self.load;
        if self.aot {
            let out_dir = if let Some(out_dir) = compiler.out_dir() {
                out_dir.join(&self.bench_name)
            } else {
                let dir = std::env::temp_dir().join("revmc-cli").join(&self.bench_name);
                std::fs::create_dir_all(&dir)?;
                dir
            };

            // Compile.
            let obj = out_dir.join("a.o");
            compiler.write_object_to_file(&obj)?;
            ensure!(obj.exists(), "Failed to write object file");
            eprintln!("Compiled object file to {}", obj.display());

            // Link.
            if !self.no_link {
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

            if self.interpret {
                let action = interpreter.run_plain(&table, &mut host);
                let result = action
                    .instruction_result()
                    .unwrap_or(revm_interpreter::InstructionResult::Stop);
                (result, action)
            } else {
                let (mut ecx, stack, stack_len) =
                    EvmContext::from_interpreter_with_stack(&mut interpreter, &mut host);

                for (i, input) in stack_input.iter().enumerate() {
                    stack.as_mut_slice()[i] = (*input).into();
                }
                *stack_len = stack_input.len();

                let r = unsafe { f.call_noinline(Some(stack), Some(stack_len), &mut ecx) };
                let action = ecx.next_action.take().unwrap_or_else(|| {
                    revm_interpreter::InterpreterAction::Return(
                        revm_interpreter::InterpreterResult {
                            result: r,
                            output: revm_primitives::Bytes::new(),
                            gas: *ecx.gas,
                        },
                    )
                });
                (r, action)
            }
        };

        if self.n_iters == 0 {
            return Ok(());
        }

        let (ret, action) = run(f);
        println!("InstructionResult::{ret:?}");
        println!("InterpreterAction::{action:#?}");

        if self.n_iters > 1 {
            bench(self.n_iters, name, || run(f));
            return Ok(());
        }

        Ok(())
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

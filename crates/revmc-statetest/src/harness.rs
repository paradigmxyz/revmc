use crate::{
    compiled::{self, CompileMode},
    discover::find_json_tests,
    fixtures::state_test_roots,
};
use libtest_mimic::{Arguments, Failed, Trial};
use std::{
    env,
    path::{Path, PathBuf},
    process::ExitCode,
};

const NEXTEST_ENV: &str = "NEXTEST";

/// A named fixture root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestRoot {
    /// Stable root name used by the nextest harness.
    pub name: &'static str,
    /// Human readable root label.
    pub label: &'static str,
    /// Directory containing JSON test files.
    pub path: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct Mode {
    name: &'static str,
    compile_mode: CompileMode,
}

const MODES: &[Mode] = &[
    Mode { name: "interpreter", compile_mode: CompileMode::Interpreter },
    Mode { name: "jit", compile_mode: CompileMode::Jit },
    Mode { name: "aot", compile_mode: CompileMode::Aot },
];

/// Runs the cargo-nextest state test harness.
pub fn run() -> ExitCode {
    let mut args = Arguments::from_args();
    if !args.list && env::var_os(NEXTEST_ENV).is_none() {
        eprintln!("Skipping statetests: run this target through cargo nextest.");
        return ExitCode::SUCCESS;
    }

    let roots = state_test_roots();
    let trials = collect_trials(&args, roots).unwrap_or_else(|err| {
        eprintln!("{err}");
        Vec::new()
    });

    args.test_threads = Some(1);
    libtest_mimic::run(&args, trials).exit_code()
}

fn collect_trials(args: &Arguments, roots: Vec<TestRoot>) -> Result<Vec<Trial>, String> {
    if roots.is_empty() {
        return Ok(Vec::new());
    }

    if args.exact
        && let Some(filter) = &args.filter
    {
        return Ok(exact_trial(&roots, filter).into_iter().collect());
    }

    let mut trials = Vec::new();
    for mode in MODES {
        for root in &roots {
            let files = find_json_tests(std::slice::from_ref(&root.path), descend_all)?;
            for path in files {
                let name = test_name(*mode, root, &path);
                let ignored = should_ignore(&name);
                let compile_mode = mode.compile_mode;
                trials.push(
                    Trial::test(name, move || run_file(path, compile_mode))
                        .with_ignored_flag(ignored),
                );
            }
        }
    }
    Ok(trials)
}

fn exact_trial(roots: &[TestRoot], name: &str) -> Option<Trial> {
    let (mode, root, relative) = MODES
        .iter()
        .flat_map(|mode| roots.iter().map(move |root| (*mode, root)))
        .filter_map(|(mode, root)| {
            let prefix = format!("statetest::{}::{}", mode.name, root.name);
            name.strip_prefix(&prefix).map(|relative| (mode, root, relative))
        })
        .filter_map(|(mode, root, relative)| {
            relative.strip_prefix("::").map(|relative| (mode, root, relative))
        })
        .max_by_key(|(_, root, _)| root.name.len())?;
    let path = root.path.join(relative);
    path.is_file().then(|| {
        let name = name.to_string();
        let ignored = should_ignore(&name);
        let compile_mode = mode.compile_mode;
        Trial::test(name, move || run_file(path, compile_mode)).with_ignored_flag(ignored)
    })
}

fn run_file(path: PathBuf, mode: CompileMode) -> Result<(), Failed> {
    compiled::run(vec![path], true, false, mode).map(|_| ()).map_err(|err| err.to_string().into())
}

fn test_name(mode: Mode, root: &TestRoot, path: &Path) -> String {
    let relative = path.strip_prefix(&root.path).unwrap_or(path);
    format!("statetest::{}::{}::{}", mode.name, root.name, path_name(relative))
}

fn path_name(path: &Path) -> String {
    path.iter().map(|component| component.to_string_lossy()).collect::<Vec<_>>().join("/")
}

const fn descend_all(_: &Path) -> bool {
    true
}

#[rustfmt::skip]
const IGNORED_TESTS: &[&str] = &[
    "stTimeConsuming/static_Call50000_sha256.json",
    "CALLBlake2f_MaxRounds.json",
    "loopExp",
    "loopMul.json",
    "stQuadraticComplexityTest/Call1MB1024Calldepth.json",
    "stQuadraticComplexityTest/Create1000",
    "stRecursiveCreate/recursiveCreate",
    "stRevertTest/LoopCallsDepthThenRevert",
    "stRevertTest/LoopDelegateCallsDepthThenRevert",
    "stSolidityTest/RecursiveCreateContracts",
    "stStaticCall/static_Call1MB1024Calldepth.json",
    "stStaticCall/static_Call50000_sha256",
    "stStaticCall/static_CallRecursiveBomb",
    "stStaticCall/static_LoopCallsDepthThenRevert",
    "stSystemOperationsTest/CallRecursiveBomb",

    "eip7610_create_collision",
    "InitCollision.json",
    "InitCollisionParis.json",
    "RevertInCreateInInit.json",
    "RevertInCreateInInit_Paris.json",
    "RevertInCreateInInitCreate2.json",
    "RevertInCreateInInitCreate2Paris.json",
    "create2collisionStorage.json",
    "create2collisionStorageParis.json",
    "dynamicAccountOverwriteEmpty.json",
    "dynamicAccountOverwriteEmpty_Paris.json",
];

fn should_ignore(name: &str) -> bool {
    IGNORED_TESTS.iter().any(|pattern| name.contains(pattern))
}

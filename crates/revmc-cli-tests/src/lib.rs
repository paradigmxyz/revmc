//! CLI tests runner.

use tester as test;

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};
use test::{ShouldPanic, TestDesc, TestDescAndFn, TestFn, TestName, TestType};
use walkdir::{DirEntry, WalkDir};

/// Run all tests with the given command.
pub fn run_tests(cmd: &'static Path) -> i32 {
    let args = std::env::args().collect::<Vec<_>>();
    let mut opts = match test::test::parse_opts(&args) {
        Some(Ok(o)) => o,
        Some(Err(msg)) => {
            eprintln!("error: {msg}");
            return 101;
        }
        None => return 0,
    };
    /*
    // Condense output if not explicitly requested.
    let requested_pretty = || args.iter().any(|x| x.contains("--format"));
    if opts.format == test::OutputFormat::Pretty && !requested_pretty() {
        opts.format = test::OutputFormat::Terse;
    }
    */
    // [`tester`] currently (0.9.1) uses `num_cpus::get_physical`;
    // use all available threads instead.
    if opts.test_threads.is_none() {
        opts.test_threads = std::thread::available_parallelism().map(|x| x.get()).ok();
    }

    let mut tests = Vec::new();
    make_tests(cmd, &mut tests);
    tests.sort_by(|a, b| a.desc.name.as_slice().cmp(b.desc.name.as_slice()));

    match test::run_tests_console(&opts, tests) {
        Ok(true) => 0,
        Ok(false) => {
            eprintln!("Some tests failed");
            1
        }
        Err(e) => {
            eprintln!("I/O failure during tests: {e}");
            101
        }
    }
}

fn make_tests(cmd: &'static Path, tests: &mut Vec<TestDescAndFn>) {
    let config = Arc::new(Config::new(cmd));

    let codegen = config.root.join("tests/codegen");
    for entry in collect_tests(&codegen) {
        let config = config.clone();
        let path = entry.path().to_path_buf();
        let stripped = path.strip_prefix(config.root).unwrap();
        let name = stripped.display().to_string();
        tests.push(TestDescAndFn {
            desc: TestDesc {
                name: TestName::DynTestName(name),
                allow_fail: false,
                ignore: false,
                should_panic: ShouldPanic::No,
                test_type: TestType::Unknown,
            },
            testfn: TestFn::DynTestFn(Box::new(move || run_test(&config, &path))),
        });
    }
}

fn collect_tests(root: &Path) -> impl Iterator<Item = DirEntry> {
    WalkDir::new(root)
        .sort_by_file_name()
        .into_iter()
        .map(Result::unwrap)
        .filter(|e| e.file_type().is_file())
}

fn run_test(config: &Config, path: &Path) {
    let test_name = path.file_stem().unwrap().to_str().unwrap();

    // let s = fs::read_to_string(path).unwrap();
    // let comment = if path.extension() == Some("evm".as_ref()) { "#"} else { "//"};

    // let lines = s.lines().map(str::trim).filter(|s| !s.is_empty());
    // let comments = lines.filter_map(|s| s.strip_prefix(comment)).map(str::trim_start);
    // let mut commands = comments.filter_map(|s| s.strip_prefix("RUN:")).map(str::trim_start);
    // let command = commands.next().expect("no `RUN:` directive provided");
    // assert!(commands.next().is_none(), "multiple `RUN:` directives provided");

    let build_dir = &config.build_base;

    let mut compiler = Command::new(config.cmd);
    fs::create_dir_all(build_dir).unwrap();
    compiler.arg(path).arg("-o").arg(build_dir);
    // eprintln!("running compiler: {compiler:?}");
    let output = compiler.output().expect("failed to run test");
    assert!(
        output.status.success(),
        "compiler failed with {}:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let out_dir = build_dir.join(test_name);
    assert!(out_dir.exists(), "no output produced");

    let input_path = out_dir.join("opt.ll");

    let mut filecheck = Command::new(config.filecheck.as_deref().unwrap_or("FileCheck".as_ref()));
    filecheck.arg(path).arg("--input-file").arg(&input_path);
    // eprintln!("running filecheck: {filecheck:?}");
    let output = filecheck.output().expect("failed to run FileCheck");
    assert!(
        output.status.success(),
        "FileCheck failed with {}:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

struct Config {
    cmd: &'static Path,
    root: &'static Path,
    build_base: PathBuf,
    filecheck: Option<PathBuf>,
}

impl Config {
    fn new(cmd: &'static Path) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
        let build_base = root.join("target/tester");
        fs::create_dir_all(&build_base).unwrap();
        Self { root, cmd, build_base, filecheck: None }
    }
}

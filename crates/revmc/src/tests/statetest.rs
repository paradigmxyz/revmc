use revmc_statetest::runner::CompileMode;

fn run_state_tests(mode: CompileMode) {
    let Some(mut path) = revmc_statetest::get_general_state_tests_path() else {
        eprintln!("Skipping: ethereum/tests not available (run `git submodule update --init --checkout --depth 1 tests/ethereum-tests`)");
        return;
    };

    if let Ok(subdir) = std::env::var("SUBDIR") {
        path = path.join(subdir);
    }

    let test_files = revmc_statetest::find_all_json_tests(&path);
    if test_files.is_empty() {
        eprintln!("No JSON test files found in {}", path.display());
        return;
    }

    revmc_statetest::runner::run(test_files, false, false, mode).unwrap();
}

#[test]
#[ignore = "requires ethereum/tests checkout"]
fn interpreter() {
    run_state_tests(CompileMode::Interpreter);
}

#[test]
#[ignore = "requires ethereum/tests checkout"]
fn jit() {
    run_state_tests(CompileMode::Jit);
}

#[test]
#[ignore = "requires ethereum/tests checkout"]
fn aot() {
    run_state_tests(CompileMode::Aot);
}

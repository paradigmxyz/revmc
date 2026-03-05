use revmc_statetest::{find_all_json_tests, get_ethtests_path, runner::CompileMode};

fn run_state_tests(mode: CompileMode) {
    let mut path = get_ethtests_path().join("GeneralStateTests");

    if let Ok(subdir) = std::env::var("SUBDIR") {
        path = path.join(subdir);
    }

    if !path.exists() {
        eprintln!("Skipping: {} does not exist", path.display());
        return;
    }

    let test_files = find_all_json_tests(&path);
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

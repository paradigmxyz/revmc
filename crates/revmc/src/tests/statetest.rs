use revmc_statetest::compiled::CompileMode;

fn run_state_tests(mode: CompileMode) {
    let Some(mut path) = revmc_statetest::get_general_state_tests_path() else {
        eprintln!(
            "Skipping: ethereum/tests not available (run `git submodule update --init --checkout --depth 1 tests/ethereum-tests`)"
        );
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

    let single_thread = std::env::var("SINGLE_THREAD").is_ok();
    revmc_statetest::compiled::run(test_files, single_thread, false, mode).unwrap();
}

#[test]
fn interpreter() {
    run_state_tests(CompileMode::Interpreter);
}

#[test]
fn jit() {
    run_state_tests(CompileMode::Jit);
}

#[test]
fn aot() {
    run_state_tests(CompileMode::Aot);
}

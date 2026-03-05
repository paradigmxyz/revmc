pub mod merkle_trie;
pub mod runner;
pub mod utils;

use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

/// Find all JSON test files in the given path.
/// If path is a file, returns it in a vector.
/// If path is a directory, recursively finds all .json files.
pub fn find_all_json_tests(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        vec![path.to_path_buf()]
    } else {
        WalkDir::new(path)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension() == Some("json".as_ref()))
            .map(DirEntry::into_path)
            .collect()
    }
}

/// Default path to ethereum/tests repository.
const DEFAULT_ETHTESTS_PATH: &str = "tests/ethereum-tests";

/// Get the path to ethereum/tests.
pub fn get_ethtests_path() -> PathBuf {
    std::env::var("ETHTESTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_ETHTESTS_PATH))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runner::CompileMode;

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

        runner::run(test_files, false, false, mode).unwrap();
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
}

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

const STATE_TESTS_TARBALL: &str = "fixtures_general_state_tests.tgz";

/// Get the path to ethereum/tests.
pub fn get_ethtests_path() -> PathBuf {
    if let Ok(path) = std::env::var("ETHTESTS") {
        return PathBuf::from(path);
    }
    // Resolve relative to the workspace root via CARGO_MANIFEST_DIR.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    Path::new(&manifest_dir)
        .ancestors()
        .find(|p| p.join(DEFAULT_ETHTESTS_PATH).exists())
        .map(|p| p.join(DEFAULT_ETHTESTS_PATH))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_ETHTESTS_PATH))
}

/// Get the path to `GeneralStateTests`, extracting the tarball if necessary.
///
/// The ethereum/tests repo ships the fixtures as `.tgz` archives.
/// We extract into the parent of the submodule (`tests/`) rather than inside
/// the submodule itself, so the root `.gitignore` can cover the extracted
/// directory and the submodule stays clean.
pub fn get_general_state_tests_path() -> Option<PathBuf> {
    let root = get_ethtests_path();

    // Check next to the submodule first (extracted location).
    if let Some(parent) = root.parent() {
        let dir = parent.join("GeneralStateTests");
        if dir.is_dir() {
            return Some(dir);
        }
    }

    // Also check inside the submodule (manual extraction or old layout).
    let dir = root.join("GeneralStateTests");
    if dir.is_dir() {
        return Some(dir);
    }

    let tarball = root.join(STATE_TESTS_TARBALL);
    if !tarball.is_file() {
        return None;
    }

    // Extract next to the submodule so the root .gitignore covers it.
    let extract_dir = root.parent()?;
    let status = std::process::Command::new("tar")
        .arg("xzf")
        .arg(&tarball)
        .arg("-C")
        .arg(extract_dir)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }

    let dir = extract_dir.join("GeneralStateTests");
    dir.is_dir().then_some(dir)
}

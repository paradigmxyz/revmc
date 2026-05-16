use crate::harness::TestRoot;
use std::{
    env,
    path::{Path, PathBuf},
};

/// Environment variable for the state test root.
pub const STATE_TEST_ROOT_ENV: &str = "REVMC_STATETEST_ROOT";

/// Fallback environment variable for the state test root.
pub const ETHEREUM_TESTS_ENV: &str = "ETHEREUM_TESTS";

/// Environment variable for selecting stable EEST fixtures instead of develop.
pub const STATETEST_STABLE_ENV: &str = "REVMC_STATETEST_STABLE";

/// Generic EEST stable selector.
pub const EEST_STABLE_ENV: &str = "REVMC_EEST_STABLE";

/// Optional environment variable for selecting a subdirectory under the test root.
pub const STATE_TEST_SUBDIR_ENV: &str = "SUBDIR";

/// Repo-relative ethereum/tests checkout path supported for compatibility.
pub const DEFAULT_ETHEREUM_TESTS_PATH: &str = "tests/ethereum-tests";

/// Repo-relative fixture root used by the setup script and CI.
pub const DEFAULT_FIXTURES_PATH: &str = "test-fixtures";

/// A named state-test root.
pub type StateTestRoot = TestRoot;

/// Return the explicit state-test root configured through environment variables.
pub fn explicit_state_test_root_from_env() -> Option<PathBuf> {
    env::var_os(STATE_TEST_ROOT_ENV)
        .or_else(|| env::var_os(ETHEREUM_TESTS_ENV))
        .or_else(|| env::var_os("ETHTESTS"))
        .map(PathBuf::from)
        .map(workspace_relative)
        .map(|mut path| {
            apply_subdir(&mut path, STATE_TEST_SUBDIR_ENV);
            path
        })
}

/// Return the state-test roots to run by default.
pub fn state_test_roots() -> Vec<StateTestRoot> {
    if let Some(path) = explicit_state_test_root_from_env() {
        return vec![StateTestRoot { name: "custom", label: "custom state tests", path }];
    }

    default_state_test_roots().into_iter().filter(|root| root.path.is_dir()).collect()
}

/// Return the default repo-relative state-test roots, whether or not they exist.
pub fn default_state_test_roots() -> Vec<StateTestRoot> {
    let fixtures = fixtures_root();
    let ethereum_tests = workspace_root().join(DEFAULT_ETHEREUM_TESTS_PATH);
    let main_path = if env_flag(STATETEST_STABLE_ENV) || env_flag(EEST_STABLE_ENV) {
        fixtures.join("main/stable/state_tests")
    } else {
        fixtures.join("main/develop/state_tests")
    };

    let mut roots = vec![
        StateTestRoot { name: "eest", label: "execution-spec-tests", path: main_path },
        StateTestRoot {
            name: "eest::devnet",
            label: "execution-spec-tests devnet",
            path: fixtures.join("devnet/state_tests"),
        },
        StateTestRoot {
            name: "legacy::cancun",
            label: "legacy Cancun",
            path: fixtures.join("legacytests/Cancun/GeneralStateTests"),
        },
        StateTestRoot {
            name: "legacy::constantinople",
            label: "legacy Constantinople",
            path: fixtures.join("legacytests/Constantinople/GeneralStateTests"),
        },
    ];

    if let Some(path) = general_state_tests_path(&ethereum_tests) {
        roots.push(StateTestRoot {
            name: "legacy::ethereum_tests",
            label: "ethereum/tests GeneralStateTests",
            path,
        });
    }

    for root in &mut roots {
        apply_subdir(&mut root.path, STATE_TEST_SUBDIR_ENV);
    }
    roots
}

/// Resolve the workspace root by walking up from this crate.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|path| path.join("Cargo.toml").is_file() && path.join("crates").is_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

/// Return the root of downloaded test fixtures.
pub fn fixtures_root() -> PathBuf {
    env::var_os("REVMC_TEST_FIXTURES")
        .map(PathBuf::from)
        .map(workspace_relative)
        .unwrap_or_else(|| workspace_root().join(DEFAULT_FIXTURES_PATH))
}

/// Return whether an environment flag is set to a truthy value.
pub fn env_flag(name: &str) -> bool {
    env::var_os(name).is_some_and(|value| !value.is_empty() && value.to_str() != Some("0"))
}

/// Appends `SUBDIR`-style filters to a root path.
pub fn apply_subdir(root: &mut PathBuf, name: &str) {
    if let Some(subdir) = env::var_os(name)
        && !subdir.is_empty()
    {
        root.push(subdir);
    }
}

/// Resolves a path against the workspace root if it is relative.
pub fn workspace_relative(path: PathBuf) -> PathBuf {
    if path.is_absolute() { path } else { workspace_root().join(path) }
}

fn general_state_tests_path(root: &Path) -> Option<PathBuf> {
    let sibling = root.parent().map(|parent| parent.join("GeneralStateTests"));
    if let Some(path) = sibling
        && path.is_dir()
    {
        return Some(path);
    }

    let path = root.join("GeneralStateTests");
    path.is_dir().then_some(path)
}

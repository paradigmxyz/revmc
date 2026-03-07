/// Verify that `__revmc_builtin_*` symbols are exported from this binary so that
/// AOT-compiled shared libraries can resolve them at load time.
#[test]
fn builtin_symbols_exported() {
    let exe = std::env::current_exe().unwrap();
    let args: &[&str] =
        if cfg!(target_os = "macos") { &["-gU"] } else { &["-D", "--defined-only"] };
    let output = std::process::Command::new("nm").args(args).arg(&exe).output().unwrap();
    assert!(output.status.success(), "nm failed: {}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let exported: Vec<&str> = stdout.lines().filter(|l| l.contains("__revmc_builtin_")).collect();
    assert!(
        !exported.is_empty(),
        "no __revmc_builtin_* symbols exported from test binary; \
         AOT shared libraries will fail to resolve builtins at runtime"
    );
    for name in ["__revmc_builtin_mstore", "__revmc_builtin_call", "__revmc_builtin_create"] {
        assert!(
            exported.iter().any(|l| l.contains(name)),
            "{name} not exported; found: {exported:?}"
        );
    }
}

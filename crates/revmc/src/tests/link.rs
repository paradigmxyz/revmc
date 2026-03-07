/// Verify that `__revmc_builtin_*` symbols are exported from this binary so that
/// AOT-compiled shared libraries can resolve them at load time.
///
/// On macOS with `strip = true` (release profile), `nm` cannot see symbols because the
/// symbol table (nlist) is stripped, even though they remain in the Mach-O export trie.
/// We use `objdump --exports-trie` as a fallback in that case.
#[test]
fn builtin_symbols_exported() {
    let exe = std::env::current_exe().unwrap();

    // Try nm first.
    let (args, stdout) = if cfg!(target_os = "macos") {
        let output = std::process::Command::new("nm").args(["-gU"]).arg(&exe).output().unwrap();
        assert!(output.status.success(), "nm failed: {}", String::from_utf8_lossy(&output.stderr));
        let stdout = String::from_utf8(output.stdout).unwrap();
        if stdout.lines().any(|l| l.contains("__revmc_builtin_")) {
            ("-gU", stdout)
        } else {
            // Stripped binary: fall back to objdump --exports-trie.
            let output = std::process::Command::new("objdump")
                .args(["--exports-trie", exe.to_str().unwrap()])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "objdump --exports-trie failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            ("--exports-trie", String::from_utf8(output.stdout).unwrap())
        }
    } else {
        let output = std::process::Command::new("nm")
            .args(["-D", "--defined-only"])
            .arg(&exe)
            .output()
            .unwrap();
        assert!(output.status.success(), "nm failed: {}", String::from_utf8_lossy(&output.stderr));
        ("-D", String::from_utf8(output.stdout).unwrap())
    };

    let exported: Vec<&str> = stdout.lines().filter(|l| l.contains("__revmc_builtin_")).collect();
    assert!(
        !exported.is_empty(),
        "no __revmc_builtin_* symbols exported from test binary (checked with {args}); \
         AOT shared libraries will fail to resolve builtins at runtime"
    );
    for name in ["__revmc_builtin_mstore", "__revmc_builtin_call", "__revmc_builtin_create"] {
        assert!(
            exported.iter().any(|l| l.contains(name)),
            "{name} not exported; found: {exported:?}"
        );
    }
}

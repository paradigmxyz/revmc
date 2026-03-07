/// Verify that `__revmc_builtin_*` symbols are exported from this binary so that
/// AOT-compiled shared libraries can resolve them at load time.
///
/// On macOS, `nm -gU` reads `LC_SYMTAB` which is removed by `strip = true` in release builds.
/// The symbols remain in the Mach-O exports trie (what dyld uses at runtime), so we fall back
/// to `llvm-objdump --macho --exports-trie` or `dyld_info -exports` for stripped binaries.
#[test]
fn builtin_symbols_exported() {
    let exe = std::env::current_exe().unwrap();
    let stdout = exported_symbols_output(&exe);
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

#[cfg(not(target_os = "macos"))]
fn exported_symbols_output(exe: &std::path::Path) -> String {
    let output =
        std::process::Command::new("nm").args(["-D", "--defined-only"]).arg(exe).output().unwrap();
    assert!(output.status.success(), "nm failed: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).unwrap()
}

#[cfg(target_os = "macos")]
fn exported_symbols_output(exe: &std::path::Path) -> String {
    // nm -gU works on non-stripped binaries (dev builds).
    let output = std::process::Command::new("nm").args(["-gU"]).arg(exe).output().unwrap();
    if output.status.success() {
        let stdout = String::from_utf8(output.stdout).unwrap();
        if stdout.contains("__revmc_builtin_") {
            return stdout;
        }
    }

    // Stripped binary: read the exports trie instead.
    // Try llvm-objdump (available when LLVM is installed).
    if let Ok(out) = std::process::Command::new("llvm-objdump")
        .args(["--macho", "--exports-trie"])
        .arg(exe)
        .output()
    {
        if out.status.success() {
            return String::from_utf8(out.stdout).unwrap();
        }
    }

    // Try dyld_info as last resort.
    if let Ok(out) = std::process::Command::new("dyld_info").args(["-exports"]).arg(exe).output() {
        if out.status.success() {
            return String::from_utf8(out.stdout).unwrap();
        }
    }

    panic!(
        "could not inspect exported symbols: \
         nm found no symbols (binary may be stripped), \
         and neither llvm-objdump nor dyld_info are available"
    );
}

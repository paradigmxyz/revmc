#![allow(missing_docs, unused_crate_dependencies)]

const CMD: &str = env!("CARGO_BIN_EXE_revmc-cli");

fn main() {
    // Force the linker to include all builtins.
    // Even though this test only invokes the CLI binary, cargo links the crate's
    // dependencies (including revmc-builtins) and the linker needs these symbols defined.
    revmc_builtins::force_link();

    let code = revmc_cli_tests::run_tests(CMD.as_ref());
    std::process::exit(code);
}

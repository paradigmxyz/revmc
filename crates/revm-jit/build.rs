#![allow(missing_docs)]

use std::{env, path::PathBuf, process::Command};

const LLVM_VERSION: usize = 17;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../revm-jit-callbacks/");

    if !cfg!(feature = "llvm") {
        println!("skipping: LLVM feature is disabled");
        return;
    }

    if !rustc_llvm_matches() {
        println!("skipping: LLVM version mismatch");
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root_manifest = manifest_dir.parent().unwrap().parent().unwrap().join("Cargo.toml");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out_path = out_dir.join("callbacks.bc");

    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut cmd = Command::new(cargo);
    cmd.arg("rustc");
    cmd.arg("--manifest-path").arg(root_manifest);
    cmd.arg("--package=revm-jit-callbacks");
    cmd.arg("--profile=bitcode");
    cmd.arg("--no-default-features");
    cmd.arg("--");
    cmd.arg(format!("--emit=llvm-bc={}", out_path.display()));
    let status = cmd.status().unwrap();
    assert!(status.success(), "{status}: {cmd:?}");

    assert!(out_path.exists());

    println!("cargo:rustc-cfg=llvm_bc");
}

fn rustc_llvm_matches() -> bool {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let mut cmd = Command::new(rustc);
    cmd.arg("-Vv");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "{}: {cmd:?}", output.status);
    let stdout = String::from_utf8(output.stdout).expect("invalid `rustc -Vv` output");
    let llvm_version_line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("LLVM version: "))
        .expect("no LLVM version in `rustc -Vv` output");
    let llvm_major_version =
        llvm_version_line.split('.').next().expect("no major version in LLVM version");
    let llvm_major_version =
        llvm_major_version.parse::<usize>().expect("invalid major version in LLVM version");
    llvm_major_version == LLVM_VERSION
}

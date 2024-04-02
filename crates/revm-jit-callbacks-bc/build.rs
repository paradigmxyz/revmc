#![allow(missing_docs)]

use std::{env, path::PathBuf, process::Command};

const LLVM_VERSION: usize = 17;
const PROFILE: &str = "bitcode";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../revm-jit-callbacks/");
    println!("cargo:rerun-if-changed=../../Cargo.toml");

    if !rustc_llvm_matches() {
        println!("skipping: LLVM version mismatch");
        return;
    }

    need_cmd("llvm-link");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root_dir = manifest_dir.parent().unwrap().parent().unwrap();
    let root_manifest = root_dir.join("Cargo.toml");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out_path = out_dir.join("callbacks.bc");

    let target = &env::var("TARGET").unwrap()[..];
    let profile = PROFILE;

    // Compile to LLVM bitcode.
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut cmd = Command::new(cargo);
    cmd.arg("rustc");
    cmd.arg("--manifest-path").arg(root_manifest);
    cmd.arg("--package=revm-jit-callbacks");
    cmd.arg("--profile").arg(profile);
    cmd.arg("--no-default-features");
    cmd.arg("--features=__internal");
    cmd.arg("-Zbuild-std");
    cmd.arg("--target").arg(target);

    // NOTE: We need to use `CARGO_ENCODED_RUSTFLAGS` as `RUSTFLAGS` alone is ignored due to
    // `--target`.
    let rustflags = if let Ok(rustflags) = env::var("CARGO_ENCODED_RUSTFLAGS") {
        rustflags + "\u{1f}"
    } else {
        String::new()
    };
    let rustflags = rustflags + "--emit=llvm-bc";
    cmd.env("CARGO_ENCODED_RUSTFLAGS", rustflags);

    let status = cmd.status();
    let status = status.unwrap();

    assert!(status.success(), "{status}: {cmd:?}");

    // Link the bitcode files.
    let mut deps_dir = root_dir.to_path_buf();
    deps_dir.push("target");
    deps_dir.push(target);
    deps_dir.push(profile);
    deps_dir.push("deps");
    let deps_dir_entries =
        deps_dir.read_dir().unwrap().map(|entry| entry.unwrap().path()).collect::<Vec<_>>();
    let to_link_cnames = ["core", "alloc", "revm_jit_callbacks"];
    let to_link_bc = to_link_cnames.iter().copied().map(|to_link| {
        let to_link_dash = &format!("{to_link}-")[..];
        deps_dir_entries
            .iter()
            .find(|path| {
                let fname = path.file_name().unwrap().to_str().unwrap();
                fname.starts_with(to_link_dash) && fname.ends_with(".bc")
            })
            .unwrap_or_else(|| panic!("no bitcode file found for `{to_link}`"))
    });
    let mut cmd = Command::new("llvm-link");
    cmd.arg("-o").arg(&out_path);
    cmd.args(to_link_bc);
    let status = cmd.status().unwrap();
    assert!(status.success(), "{status}: {cmd:?}");

    assert!(out_path.exists(), "llvm-link didn't produce the output file");

    println!("cargo:rustc-cfg=llvm_bc");
}

fn need_cmd(cmd: &str) {
    if Command::new(cmd).output().is_err() {
        panic!("need `{cmd}` to be available");
    }
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

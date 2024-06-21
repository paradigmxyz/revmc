#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

// Must be kept in sync with `remvc-builtins`.
const MANGLE_PREFIX: &str = "__revmc_builtin_";

/// Emits the linker flag to export all the necessary symbols.
pub fn emit() {
    let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap();
    let flag =
        if target_vendor == "apple" { "-exported_symbol" } else { "--export-dynamic-symbol" };
    println!("cargo:rustc-link-arg=-Wl,{flag},{MANGLE_PREFIX}*");
}

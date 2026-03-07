#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Must be kept in sync with `remvc-builtins`.
const MANGLE_PREFIX: &str = "__revmc_builtin_";

/// Emits the linker flag to export all the necessary symbols.
pub fn emit() {
    let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap();
    if target_vendor == "apple" {
        // Mach-O C symbols have a leading `_`, so `__revmc_builtin_*` becomes `___revmc_builtin_*`.
        println!("cargo:rustc-link-arg=-Wl,-exported_symbol,_{MANGLE_PREFIX}*");
        // Preserve global symbols in executables during LTO dead-stripping.
        println!("cargo:rustc-link-arg=-Wl,-export_dynamic");
    } else {
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic-symbol,{MANGLE_PREFIX}*");
    }
}

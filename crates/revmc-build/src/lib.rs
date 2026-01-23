#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg))]

use std::io::Write;

/// All builtin function names that need to be exported.
/// Must be kept in sync with `revmc-builtins`.
const BUILTIN_SYMBOLS: &[&str] = &[
    "__revmc_builtin_addmod",
    "__revmc_builtin_balance",
    "__revmc_builtin_basefee",
    "__revmc_builtin_blob_base_fee",
    "__revmc_builtin_blob_hash",
    "__revmc_builtin_blockhash",
    "__revmc_builtin_call",
    "__revmc_builtin_calldatacopy",
    "__revmc_builtin_calldataload",
    "__revmc_builtin_calldatasize",
    "__revmc_builtin_chainid",
    "__revmc_builtin_codecopy",
    "__revmc_builtin_codesize",
    "__revmc_builtin_coinbase",
    "__revmc_builtin_create",
    "__revmc_builtin_difficulty",
    "__revmc_builtin_do_return",
    "__revmc_builtin_exp",
    "__revmc_builtin_extcodecopy",
    "__revmc_builtin_extcodehash",
    "__revmc_builtin_extcodesize",
    "__revmc_builtin_gas_price",
    "__revmc_builtin_gaslimit",
    "__revmc_builtin_keccak256",
    "__revmc_builtin_log",
    "__revmc_builtin_mcopy",
    "__revmc_builtin_mload",
    "__revmc_builtin_msize",
    "__revmc_builtin_mstore",
    "__revmc_builtin_mstore8",
    "__revmc_builtin_mulmod",
    "__revmc_builtin_number",
    "__revmc_builtin_origin",
    "__revmc_builtin_panic",
    "__revmc_builtin_resize_memory",
    "__revmc_builtin_returndatacopy",
    "__revmc_builtin_self_balance",
    "__revmc_builtin_selfdestruct",
    "__revmc_builtin_sload",
    "__revmc_builtin_sstore",
    "__revmc_builtin_timestamp",
    "__revmc_builtin_tload",
    "__revmc_builtin_tstore",
];

/// Emits the linker flags to export all the necessary builtin symbols for binary targets.
///
/// On macOS, this generates an exported symbols list file since `-exported_symbol`
/// does not support wildcards. On other platforms, it uses `--export-dynamic-symbol`.
///
/// Call [`emit_tests`] separately if you also need to export symbols for test targets.
pub fn emit() {
    emit_inner(false);
}

/// Emits the linker flags to export all the necessary builtin symbols for both binary
/// and test targets.
///
/// Use this instead of [`emit`] when your crate has tests that need the builtin symbols.
pub fn emit_tests() {
    emit_inner(true);
}

fn emit_inner(include_tests: bool) {
    let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap();

    if target_vendor == "apple" {
        // macOS: Generate an exported symbols list file.
        // Mach-O symbols have an extra leading underscore.
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let symbols_path = std::path::Path::new(&out_dir).join("exported_symbols.txt");
        let mut file = std::fs::File::create(&symbols_path).unwrap();
        for sym in BUILTIN_SYMBOLS {
            // Add Mach-O underscore prefix
            writeln!(file, "_{sym}").unwrap();
        }
        let flag = format!("-Wl,-exported_symbols_list,{}", symbols_path.display());
        println!("cargo:rustc-link-arg={flag}");
        if include_tests {
            println!("cargo:rustc-link-arg-tests={flag}");
        }
    } else {
        // Linux/other: Use --export-dynamic-symbol for each symbol.
        // Wildcards work on Linux, but explicit is more portable.
        for sym in BUILTIN_SYMBOLS {
            let flag = format!("-Wl,--export-dynamic-symbol,{sym}");
            println!("cargo:rustc-link-arg={flag}");
            if include_tests {
                println!("cargo:rustc-link-arg-tests={flag}");
            }
        }
    }
}

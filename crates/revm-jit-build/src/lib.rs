#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

use std::path::PathBuf;

/// The linker dynamic list containing all the necessary symbols.
pub const DYNAMIC_LIST: &str = include_str!("dynamic_list.txt");

/// Emits the linker flag to export all the necessary symbols.
pub fn emit() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let dynamic_list = out_dir.join("dynamic_list.txt");
    std::fs::write(&dynamic_list, DYNAMIC_LIST).unwrap();
    println!("cargo:rustc-link-arg=-Wl,--dynamic-list={}", dynamic_list.display());
}

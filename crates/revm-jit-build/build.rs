#![allow(missing_docs)]

use std::fmt::Write;

const MANGLE_PREFIX: &str = "__revm_jit_builtin_";

fn main() {
    let input_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../revm-jit-builtins/src/ir.rs");
    let output_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/dynamic_list.txt");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={input_path}");

    let input = std::fs::read_to_string(input_path).unwrap();
    // Skip the first which is the string itself as a variable.
    let symbols = input.match_indices(MANGLE_PREFIX).skip(1).map(|(i, _)| {
        let start_search = i + MANGLE_PREFIX.len();
        let end = start_search + input[start_search..].find('(').unwrap();
        &input[i..end]
    });
    let symbols = symbols.collect::<Vec<_>>();
    assert!(!symbols.is_empty(), "No symbols found in the input file");

    let mut list = String::new();
    writeln!(list, "{{").unwrap();
    for symbol in symbols {
        writeln!(list, "    {symbol};").unwrap();
    }
    writeln!(list, "}};").unwrap();

    std::fs::write(output_path, list).unwrap();
}

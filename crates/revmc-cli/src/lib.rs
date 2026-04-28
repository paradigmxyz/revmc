#![allow(missing_docs)]

use revm_primitives::hex;
use revmc::eyre::{Result, WrapErr, eyre};
use std::path::Path;

mod benches;
pub use benches::*;

#[cfg(feature = "llvm")]
mod fixture;
#[cfg(feature = "llvm")]
pub use fixture::*;

/// Reads bytecode from `path`, auto-detecting hex/asm/binary by extension and contents.
pub fn read_code_path(path: &Path) -> Result<Vec<u8>> {
    let contents = std::fs::read(path)?;
    let ext = path.extension().and_then(|s| s.to_str());
    read_code_string(&contents, ext)
}

pub fn read_code_string(contents: &[u8], ext: Option<&str>) -> Result<Vec<u8>> {
    let has_prefix = contents.starts_with(b"0x") || contents.starts_with(b"0X");
    let is_hex = ext != Some("bin") && (ext == Some("hex") || has_prefix);
    let utf8 =
        || std::str::from_utf8(contents).wrap_err("given code is not valid UTF-8").map(str::trim);
    if is_hex {
        let input = utf8()?;
        let mut lines = input.lines().map(str::trim);
        let first_line = lines.next().unwrap_or_default();
        hex::decode(first_line).wrap_err("given code is not valid hex")
    } else if ext == Some("bin") || !contents.is_ascii() {
        Ok(contents.to_vec())
    } else if ext == Some("evm") {
        revmc::parse_asm(utf8()?)
    } else if contents.is_ascii() {
        let s = utf8()?;
        revmc::parse_asm(s).or_else(|e1| match hex::decode(s) {
            Ok(b) => Ok(b),
            Err(e2) => {
                Err(eyre::eyre!("input is not valid EVM bytecode or hex:\n1. {e1}\n2. {e2}"))
            }
        })
    } else {
        Err(eyre!("could not determine bytecode type"))
    }
}

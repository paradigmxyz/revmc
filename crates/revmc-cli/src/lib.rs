#![allow(missing_docs)]

use revm_primitives::hex;
use revmc::eyre::{Result, WrapErr, eyre};
use std::path::Path;

mod benches;
pub use benches::*;

mod host;
pub use host::BenchHost;

#[cfg(feature = "llvm")]
mod fixture;
#[cfg(feature = "llvm")]
pub use fixture::*;

pub fn read_code(code: Option<&str>, code_path: Option<&Path>) -> Result<Vec<u8>> {
    if let Some(code) = code {
        return read_code_string(code.trim().as_bytes(), None);
    }

    if let Some(code_path) = code_path {
        let contents = std::fs::read(code_path)?;
        let ext = code_path.extension().and_then(|s| s.to_str());
        return read_code_string(&contents, ext);
    }

    Err(eyre!("one of --code, --code-path is required when argument is 'custom'"))
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
        revmc::parse_asm(s).or_else(|_| hex::decode(s).wrap_err("given code is not valid hex"))
    } else {
        Err(eyre!("could not determine bytecode type"))
    }
}

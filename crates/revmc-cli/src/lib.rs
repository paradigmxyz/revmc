#![allow(missing_docs)]

use revm_interpreter::OpCode;
use revm_primitives::hex;
use revmc::{
    eyre::{bail, eyre, Result, WrapErr},
    U256,
};
use std::{cmp::Ordering, path::Path, str::FromStr};

mod benches;
pub use benches::*;

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
        parse_evm_dsl(utf8()?)
    } else if contents.is_ascii() {
        let s = utf8()?;
        parse_evm_dsl(s).or_else(|_| hex::decode(s).wrap_err("given code is not valid hex"))
    } else {
        Err(eyre!("could not determine bytecode type"))
    }
}

/// Parse EVM code from a string.
fn parse_evm_dsl(s: &str) -> Result<Vec<u8>> {
    const COM: char = ';';

    let mut code = Vec::with_capacity(32);

    let lines = s.lines().map(str::trim).filter(|s| !s.is_empty());
    let words = lines.flat_map(|s| s.split_whitespace().take_while(|s| !s.starts_with(COM)));
    let mut words = words.peekable();
    while let Some(word) = words.next() {
        if word == "PUSH" {
            let next = words.next().ok_or_else(|| eyre!("missing immediate for opcode PUSH"))?;
            let imm_bytes = parse_imm(next, None)?;
            code.push(OpCode::PUSH0.get() + imm_bytes.len() as u8);
            code.extend_from_slice(&imm_bytes);
        } else {
            let op = OpCode::parse(word).ok_or_else(|| eyre!("invalid opcode: {word:?}"))?;
            code.push(op.get());
            let imm_len = op.info().immediate_size();
            if imm_len > 0 {
                let imm = words.next().ok_or_else(|| eyre!("missing immediate for opcode {op}"))?;
                let imm_bytes = parse_imm(imm, Some(imm_len))?;
                code.extend_from_slice(&imm_bytes);
            } else if let Some(next) = words.peek() {
                if U256::from_str(next).is_ok() {
                    bail!("unexpected immediate for opcode {op}");
                }
            }
        }
    }

    Ok(code)
}

fn parse_imm(s: &str, size: Option<u8>) -> Result<Vec<u8>> {
    let num: U256 = s.parse().wrap_err("failed to parse immediate")?;
    let mut imm_bytes = num.to_be_bytes_trimmed_vec();
    if let Some(size) = size {
        debug_assert!(size <= 32);
        match imm_bytes.len().cmp(&(size as usize)) {
            Ordering::Less => {
                let extend = size as usize - imm_bytes.len();
                imm_bytes.splice(0..0, std::iter::repeat(0).take(extend));
            }
            Ordering::Equal => {}
            Ordering::Greater => {
                bail!("expected at most {size} immediate bytes, got {}", imm_bytes.len())
            }
        }
    }
    debug_assert!(imm_bytes.len() <= 32);
    Ok(imm_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm_interpreter::opcode as op;

    #[test]
    fn test_evm_dsl() {
        let cases: &[(&str, Vec<u8>)] = &[
            ("ADD ; ADD\n ADD", vec![op::ADD, op::ADD]),
            ("PUSH1 0", vec![op::PUSH1, 0]),
            ("PUSH3 0x000069", vec![op::PUSH3, 0, 0, 0x69]),
            ("PUSH3 0x69 ; padded", vec![op::PUSH3, 0, 0, 0x69]),
            ("PUSH 0", vec![op::PUSH0]),
            ("PUSH 1", vec![op::PUSH1, 1]),
            ("PUSH 2", vec![op::PUSH1, 2]),
            ("PUSH 69", vec![op::PUSH1, 69]),
            ("PUSH 0x2222", vec![op::PUSH2, 0x22, 0x22]),
        ];
        for (s, expected) in cases.iter() {
            let code = match parse_evm_dsl(s) {
                Ok(code) => code,
                Err(e) => panic!("code: {s:?}\n\n err: {e}"),
            };
            assert_eq!(code, *expected, "{s:?}");
        }
    }
}

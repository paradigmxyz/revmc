//! EVM bytecode assembler.
//!
//! Assembles EVM mnemonics from a string into raw bytecode. Supports:
//! - Standard EVM opcodes (`ADD`, `PUSH1 0x42`, etc.)
//! - Auto-sized pushes (`PUSH 0x1234` picks the smallest encoding)
//! - Labels: `name:` defines a label at the current PC, `PUSH name` / `PUSHn name` resolves to the
//!   label's byte offset
//! - Comments starting with `;`

use crate::{
    U256,
    eyre::{self, Result},
};
use revm_bytecode::opcode::OpCode;
use std::{cmp::Ordering, collections::HashMap, str::FromStr};

/// Parse EVM assembly from a string into bytecode.
///
/// See [module docs](self) for syntax.
pub fn parse_asm(s: &str) -> Result<Vec<u8>> {
    let items = parse_items(s)?;
    layout_and_emit(&items)
}

/// Comment character.
const COM: char = ';';

/// A parsed item from the source.
enum Item<'a> {
    /// A label definition (`name:`).
    Label(&'a str),
    /// An instruction.
    Inst(Inst<'a>),
}

/// A parsed instruction.
struct Inst<'a> {
    opcode: u8,
    imm: Option<Imm<'a>>,
    push_kind: PushKind,
}

/// An immediate value.
enum Imm<'a> {
    /// A numeric literal.
    Number(U256),
    /// A label reference (resolved during layout).
    Label(&'a str),
}

/// How the push width is determined.
enum PushKind {
    /// Not a push instruction.
    None,
    /// Fixed width (`PUSH1`..`PUSH32`).
    Fixed(u8),
    /// Auto-sized (`PUSH`).
    Auto,
}

/// Parse source text into items.
fn parse_items<'a>(s: &'a str) -> Result<Vec<Item<'a>>> {
    let mut items = Vec::new();
    let lines = s.lines().map(str::trim).filter(|s| !s.is_empty());
    let words = lines.flat_map(|s| s.split_whitespace().take_while(|s| !s.starts_with(COM)));
    let mut words = words.peekable();
    while let Some(word) = words.next() {
        // Label definition.
        if let Some(name) = word.strip_suffix(':') {
            eyre::ensure!(!name.is_empty(), "empty label name");
            items.push(Item::Label(name));
            continue;
        }

        if word == "PUSH" {
            let next =
                words.next().ok_or_else(|| eyre::eyre!("missing immediate for opcode PUSH"))?;
            let imm = parse_imm_or_label(next);
            items.push(Item::Inst(Inst { opcode: 0, imm: Some(imm), push_kind: PushKind::Auto }));
        } else {
            let op = OpCode::parse(word).ok_or_else(|| eyre::eyre!("invalid opcode: {word:?}"))?;
            let opcode = op.get();
            let imm_len = op.info().immediate_size();
            if imm_len > 0 {
                let next =
                    words.next().ok_or_else(|| eyre::eyre!("missing immediate for opcode {op}"))?;
                let imm = parse_imm_or_label(next);
                items.push(Item::Inst(Inst {
                    opcode,
                    imm: Some(imm),
                    push_kind: PushKind::Fixed(imm_len),
                }));
            } else {
                if let Some(next) = words.peek()
                    && U256::from_str(next).is_ok()
                {
                    eyre::bail!("unexpected immediate for opcode {op}");
                }
                items.push(Item::Inst(Inst { opcode, imm: None, push_kind: PushKind::None }));
            }
        }
    }
    Ok(items)
}

/// Try to parse as a number, fall back to label reference.
fn parse_imm_or_label<'a>(s: &'a str) -> Imm<'a> {
    match U256::from_str(s) {
        Ok(n) => Imm::Number(n),
        Err(_) => Imm::Label(s),
    }
}

/// Encode a U256 as big-endian bytes with optional fixed size.
fn encode_imm(num: U256, size: Option<u8>) -> Result<Vec<u8>> {
    let mut bytes = num.to_be_bytes_trimmed_vec();
    if let Some(size) = size {
        debug_assert!(size <= 32);
        match bytes.len().cmp(&(size as usize)) {
            Ordering::Less => {
                let extend = size as usize - bytes.len();
                bytes.splice(0..0, std::iter::repeat_n(0, extend));
            }
            Ordering::Equal => {}
            Ordering::Greater => {
                eyre::bail!("expected at most {size} immediate bytes, got {}", bytes.len());
            }
        }
    }
    debug_assert!(bytes.len() <= 32);
    Ok(bytes)
}

/// Compute the minimum push width for a value (0 for zero, 1 for 1..=0xff, etc.).
fn min_push_width(val: usize) -> u8 {
    if val == 0 {
        0
    } else {
        let bits = usize::BITS - val.leading_zeros();
        bits.div_ceil(8) as u8
    }
}

/// Layout items with label resolution (fixed-point for auto-sized label pushes) and emit bytecode.
fn layout_and_emit(items: &[Item<'_>]) -> Result<Vec<u8>> {
    // Collect auto-push-label indices for the fixpoint.
    let mut auto_label_indices: Vec<usize> = Vec::new();
    let mut has_any_label = false;

    for (i, item) in items.iter().enumerate() {
        match item {
            Item::Label(_) => has_any_label = true,
            Item::Inst(inst) => {
                if matches!(inst.imm, Some(Imm::Label(_))) {
                    has_any_label = true;
                    if matches!(inst.push_kind, PushKind::Auto) {
                        auto_label_indices.push(i);
                    }
                }
            }
        }
    }

    // If no labels at all, just emit directly.
    if !has_any_label {
        let mut code = Vec::with_capacity(32);
        for item in items {
            if let Item::Inst(inst) = item {
                emit_inst_no_labels(inst, &mut code)?;
            }
        }
        return Ok(code);
    }

    // Fixed-point layout: auto-push widths start at 0 and grow monotonically.
    let mut auto_widths: Vec<u8> = vec![0; items.len()];
    let mut label_pcs: HashMap<&str, usize> = HashMap::new();

    loop {
        // Compute PCs with current widths.
        label_pcs.clear();
        let mut pc = 0usize;
        for (i, item) in items.iter().enumerate() {
            match item {
                Item::Label(name) => {
                    label_pcs.insert(name, pc);
                }
                Item::Inst(inst) => {
                    pc += 1; // opcode byte
                    match &inst.push_kind {
                        PushKind::None => {}
                        PushKind::Fixed(n) => pc += *n as usize,
                        PushKind::Auto => {
                            let width = match &inst.imm {
                                Some(Imm::Label(_)) => auto_widths[i],
                                Some(Imm::Number(n)) => {
                                    let bytes = n.to_be_bytes_trimmed_vec();
                                    bytes.len() as u8
                                }
                                None => 0,
                            };
                            pc += width as usize;
                        }
                    }
                }
            }
        }

        // Check if any auto-width needs to grow.
        let mut changed = false;
        for &i in &auto_label_indices {
            if let Item::Inst(inst) = &items[i]
                && let Some(Imm::Label(name)) = &inst.imm
            {
                let target_pc = *label_pcs
                    .get(name)
                    .ok_or_else(|| eyre::eyre!("undefined label: {name:?}"))?;
                let needed = min_push_width(target_pc);
                if needed > auto_widths[i] {
                    auto_widths[i] = needed;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Final emit.
    let mut code = Vec::with_capacity(64);
    for (i, item) in items.iter().enumerate() {
        if let Item::Inst(inst) = item {
            match &inst.push_kind {
                PushKind::None => {
                    code.push(inst.opcode);
                }
                PushKind::Fixed(size) => {
                    code.push(inst.opcode);
                    let val = resolve_imm(inst.imm.as_ref().unwrap(), &label_pcs)?;
                    let bytes = encode_imm(val, Some(*size))?;
                    code.extend_from_slice(&bytes);
                }
                PushKind::Auto => {
                    let val = resolve_imm(inst.imm.as_ref().unwrap(), &label_pcs)?;
                    let width = match &inst.imm {
                        Some(Imm::Label(_)) => auto_widths[i],
                        _ => {
                            let bytes = val.to_be_bytes_trimmed_vec();
                            bytes.len() as u8
                        }
                    };
                    let push0 = OpCode::PUSH0.get();
                    code.push(push0 + width);
                    if width > 0 {
                        let bytes = encode_imm(val, Some(width))?;
                        code.extend_from_slice(&bytes);
                    }
                }
            }
        }
    }

    Ok(code)
}

/// Resolve an immediate value, substituting label PCs.
fn resolve_imm(imm: &Imm<'_>, label_pcs: &HashMap<&str, usize>) -> Result<U256> {
    match imm {
        Imm::Number(n) => Ok(*n),
        Imm::Label(name) => {
            let pc = label_pcs.get(name).ok_or_else(|| eyre::eyre!("undefined label: {name:?}"))?;
            Ok(U256::from(*pc))
        }
    }
}

/// Emit a single instruction (no-label fast path).
fn emit_inst_no_labels(inst: &Inst<'_>, code: &mut Vec<u8>) -> Result<()> {
    match &inst.push_kind {
        PushKind::None => {
            code.push(inst.opcode);
        }
        PushKind::Fixed(size) => {
            code.push(inst.opcode);
            let Imm::Number(n) = inst.imm.as_ref().unwrap() else {
                unreachable!();
            };
            let bytes = encode_imm(*n, Some(*size))?;
            code.extend_from_slice(&bytes);
        }
        PushKind::Auto => {
            let Imm::Number(n) = inst.imm.as_ref().unwrap() else {
                unreachable!();
            };
            let bytes = encode_imm(*n, None)?;
            let push0 = OpCode::PUSH0.get();
            code.push(push0 + bytes.len() as u8);
            code.extend_from_slice(&bytes);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm_bytecode::opcode as op;

    #[test]
    fn basic_opcodes() {
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
            let code = match parse_asm(s) {
                Ok(code) => code,
                Err(e) => panic!("code: {s:?}\n\n err: {e}"),
            };
            assert_eq!(code, *expected, "{s:?}");
        }
    }

    #[test]
    fn label_forward_ref() {
        // PUSH target / JUMP / JUMPDEST / STOP
        // target: is at pc=3 (PUSH1 takes 2 bytes, JUMP takes 1).
        let code = parse_asm(
            "PUSH target
             JUMP
             target:
             JUMPDEST
             STOP",
        )
        .unwrap();
        assert_eq!(code, vec![op::PUSH1, 3, op::JUMP, op::JUMPDEST, op::STOP]);
    }

    #[test]
    fn label_backward_ref() {
        // target: JUMPDEST / PUSH target / JUMP
        let code = parse_asm(
            "target:
             JUMPDEST
             PUSH target
             JUMP",
        )
        .unwrap();
        // target is at pc=0.
        assert_eq!(code, vec![op::JUMPDEST, op::PUSH0, op::JUMP]);
    }

    #[test]
    fn label_fixed_width() {
        let code = parse_asm(
            "PUSH1 target
             JUMP
             target:
             JUMPDEST
             STOP",
        )
        .unwrap();
        assert_eq!(code, vec![op::PUSH1, 3, op::JUMP, op::JUMPDEST, op::STOP]);
    }

    #[test]
    fn multiple_labels_same_pc() {
        let code = parse_asm(
            "a:
             b:
             JUMPDEST
             PUSH a
             PUSH b
             STOP",
        )
        .unwrap();
        // Both labels at pc=0.
        assert_eq!(code, vec![op::JUMPDEST, op::PUSH0, op::PUSH0, op::STOP]);
    }

    #[test]
    fn undefined_label() {
        assert!(parse_asm("PUSH missing JUMP").is_err());
    }

    #[test]
    fn empty_label() {
        assert!(parse_asm(": STOP").is_err());
    }
}

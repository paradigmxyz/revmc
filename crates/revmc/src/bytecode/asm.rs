//! EVM bytecode assembler.

use crate::{
    U256, encode_pair, encode_single,
    eyre::{self, Result},
};
use revm_bytecode::opcode::{self as op, OpCode};
use revm_primitives::map::HashMap;
use std::{cmp::Ordering, str::FromStr};

/// Parse EVM assembly from a string into bytecode.
///
/// Assembles EVM mnemonics from a string into raw bytecode. Supports:
/// - Standard EVM opcodes (`ADD`, `PUSH1 0x42`, etc.)
/// - Auto-sized pushes (`PUSH 0x1234` picks the smallest encoding)
/// - Labels: `name:` defines a label at the current PC, `PUSH %name` / `PUSHn %name` resolves to
///   the label's byte offset
/// - Comments starting with `;`
/// - C-style `#define` macros (textual expansion before parsing), with optional parameters
///
/// ```evm
/// #define PUSH_TWO(a, b) PUSH $a PUSH $b
///
/// entry:
///   PUSH_TWO(1, 2)
///   ADD
///   PUSH %target
///   JUMP
///
/// target:
///   JUMPDEST
///   STOP
/// ```
pub fn parse_asm(s: &str) -> Result<Vec<u8>> {
    let expanded = preprocess(s)?;
    let tokens = tokenize(&expanded)?;
    let items = parse_items(&tokens)?;
    layout_and_emit(&items)
}

// ———————————————————————————————————————————————————————————————————————
// Tokenizer
// ———————————————————————————————————————————————————————————————————————

/// Comment character.
const COM: char = ';';

/// A token produced by the tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Token<'a> {
    /// An identifier (opcode name, macro name, etc.).
    Ident(&'a str),
    /// A label definition (`name:`).
    Label(&'a str),
    /// A label reference (`%name`).
    LabelRef(&'a str),
    /// A numeric literal.
    Number(&'a str),
    /// A comma-separated pair like `1,2` (used by EXCHANGE).
    Pair(&'a str, &'a str),
}

/// Tokenize source text into a flat sequence of tokens.
///
/// Strips comments and whitespace. The input should already be preprocessed
/// (macros expanded).
fn tokenize<'a>(s: &'a str) -> Result<Vec<Token<'a>>> {
    let mut tokens = Vec::new();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Strip trailing comment.
        let code = line.split_once(COM).map_or(line, |(l, _)| l.trim_end());
        if code.is_empty() {
            continue;
        }
        tokenize_line(code, &mut tokens)?;
    }
    Ok(tokens)
}

/// Tokenize a single line of code (no comments) into tokens.
fn tokenize_line<'a>(line: &'a str, tokens: &mut Vec<Token<'a>>) -> Result<()> {
    let mut rest = line;
    while !rest.is_empty() {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }

        // Label reference: %name.
        if rest.starts_with('%') {
            rest = &rest[1..];
            let end =
                rest.find(|c: char| !c.is_ascii_alphanumeric() && c != '_').unwrap_or(rest.len());
            let name = &rest[..end];
            eyre::ensure!(!name.is_empty(), "empty label reference");
            tokens.push(Token::LabelRef(name));
            rest = &rest[end..];
            continue;
        }

        // Find word boundary.
        let end = rest.find(|c: char| c.is_ascii_whitespace()).unwrap_or(rest.len());
        let word = &rest[..end];
        rest = &rest[end..];

        // Label definition: name:.
        if let Some(name) = word.strip_suffix(':') {
            eyre::ensure!(!name.is_empty(), "empty label name");
            tokens.push(Token::Label(name));
            continue;
        }

        // Comma-separated pair (e.g. `1,2` for EXCHANGE).
        if let Some((a, b)) = word.split_once(',') {
            tokens.push(Token::Pair(a, b));
            continue;
        }

        // Number (starts with digit or 0x).
        if U256::from_str(word).is_ok() {
            tokens.push(Token::Number(word));
            continue;
        }

        // Everything else is an identifier.
        tokens.push(Token::Ident(word));
    }
    Ok(())
}

// ———————————————————————————————————————————————————————————————————————
// Preprocessor (#define macros)
// ———————————————————————————————————————————————————————————————————————

/// A macro definition: optional parameter names and a body template.
struct Macro<'a> {
    params: Vec<&'a str>,
    body: &'a str,
}

/// Builtin macros available in all assembly sources.
fn builtin_macros() -> Vec<(&'static str, Macro<'static>)> {
    vec![
        // Store the top-of-stack word to memory offset 0 and return 32 bytes.
        ("RET_WORD", Macro { params: vec![], body: "PUSH0 MSTORE PUSH1 0x20 PUSH0 RETURN" }),
    ]
}

fn is_valid_macro_name(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Parse a `#define` directive into (name, Macro).
fn parse_define<'a>(rest: &'a str) -> Result<(&'a str, Macro<'a>)> {
    let rest = rest.trim_start();
    eyre::ensure!(!rest.is_empty(), "empty #define name");

    // Check for parameterized macro: NAME(a, b, c).
    if let Some(paren) = rest.find('(') {
        let name = &rest[..paren];
        eyre::ensure!(is_valid_macro_name(name), "invalid #define name: {name:?}");

        let close = rest.find(')').ok_or_else(|| eyre::eyre!("unclosed '(' in #define {name}"))?;
        let params_str = &rest[paren + 1..close];
        let params: Vec<&str> = if params_str.trim().is_empty() {
            vec![]
        } else {
            params_str.split(',').map(str::trim).collect()
        };
        for p in &params {
            eyre::ensure!(is_valid_macro_name(p), "invalid macro parameter name: {p:?}");
        }

        let body = rest[close + 1..].trim();
        Ok((name, Macro { params, body }))
    } else {
        let (name, body) = rest
            .split_once(|c: char| c.is_ascii_whitespace())
            .map(|(n, b)| (n, b.trim()))
            .unwrap_or((rest, ""));
        eyre::ensure!(is_valid_macro_name(name), "invalid #define name: {name:?}");
        Ok((name, Macro { params: vec![], body }))
    }
}

/// Expand a macro body, substituting `$param` references with the provided arguments.
fn expand_macro(mac: &Macro<'_>, args: &[&str]) -> Result<String> {
    eyre::ensure!(
        args.len() == mac.params.len(),
        "macro expects {} argument(s), got {}",
        mac.params.len(),
        args.len()
    );
    if mac.params.is_empty() {
        return Ok(mac.body.to_string());
    }
    let mut out = String::with_capacity(mac.body.len());
    let mut chars = mac.body.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '$' {
            // Collect the parameter name.
            let start = out.len();
            while let Some(&c) = chars.peek()
                && (c.is_ascii_alphanumeric() || c == '_')
            {
                out.push(c);
                chars.next();
            }
            let param_name = &out[start..];
            if let Some(idx) = mac.params.iter().position(|&p| p == param_name) {
                out.truncate(start);
                out.push_str(args[idx]);
            } else {
                // Not a known param, keep `$name` literally.
                out.insert(start, '$');
            }
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

/// Preprocess source text: collect `#define` macros and expand all macro references.
fn preprocess(s: &str) -> Result<String> {
    let mut macros = HashMap::<&str, Macro<'_>>::default();
    for (name, mac) in builtin_macros() {
        macros.insert(name, mac);
    }

    // First pass: collect user-defined macros and non-directive lines.
    let mut lines = Vec::new();
    for line in s.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("#define") {
            let (name, mac) = parse_define(rest)?;
            macros.insert(name, mac);
        } else {
            lines.push(line);
        }
    }

    if macros.is_empty() {
        return Ok(lines.join("\n"));
    }

    // Second pass: expand macro references in each word.
    let mut out = String::with_capacity(s.len());
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let trimmed = line.trim();
        // Preserve empty lines and pure comment lines as-is.
        if trimmed.is_empty() || trimmed.starts_with(COM) {
            out.push_str(line);
            continue;
        }
        // Strip trailing comment.
        let code = trimmed.split_once(COM).map_or(trimmed, |(l, _)| l.trim_end());
        expand_line(code, &macros, &mut out)?;
    }
    Ok(out)
}

/// Expand macros in a single line of code (no comments), appending to `out`.
fn expand_line(code: &str, macros: &HashMap<&str, Macro<'_>>, out: &mut String) -> Result<()> {
    let mut rest = code;
    let mut first = true;
    while !rest.is_empty() {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }

        // Extract next word.
        let word_end =
            rest.find(|c: char| c.is_ascii_whitespace() || c == '(').unwrap_or(rest.len());
        let word = &rest[..word_end];
        if word.is_empty() {
            break;
        }

        if !first {
            out.push(' ');
        }
        first = false;

        if let Some(mac) = macros.get(word) {
            rest = &rest[word_end..];
            if !mac.params.is_empty() {
                // Parse parenthesized arguments: NAME(arg1, arg2).
                let rest_trimmed = rest.trim_start();
                eyre::ensure!(rest_trimmed.starts_with('('), "macro {word:?} expects arguments");
                let close = rest_trimmed
                    .find(')')
                    .ok_or_else(|| eyre::eyre!("unclosed '(' in macro invocation {word:?}"))?;
                let args_str = &rest_trimmed[1..close];
                let args: Vec<&str> = args_str.split(',').map(str::trim).collect();
                rest = rest_trimmed[close + 1..].trim_start();
                let expanded = expand_macro(mac, &args)?;
                out.push_str(&expanded);
            } else {
                let expanded = expand_macro(mac, &[])?;
                out.push_str(&expanded);
            }
        } else {
            out.push_str(word);
            rest = &rest[word_end..];
        }
    }
    Ok(())
}

// ———————————————————————————————————————————————————————————————————————
// Parser (tokens → items)
// ———————————————————————————————————————————————————————————————————————

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

/// Parse a token stream into items.
fn parse_items<'a>(tokens: &[Token<'a>]) -> Result<Vec<Item<'a>>> {
    let mut items = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i] {
            Token::Label(name) => {
                items.push(Item::Label(name));
                i += 1;
            }
            Token::Ident(word) => {
                if *word == "PUSH" {
                    i += 1;
                    let imm = expect_imm(tokens, &mut i, "PUSH")?;
                    items.push(Item::Inst(Inst {
                        opcode: 0,
                        imm: Some(imm),
                        push_kind: PushKind::Auto,
                    }));
                } else {
                    let opc = OpCode::parse(word)
                        .ok_or_else(|| eyre::eyre!("invalid opcode: {word:?}"))?;
                    let opcode = opc.get();
                    i += 1;

                    if opcode == op::DUPN || opcode == op::SWAPN {
                        let next = expect_token(tokens, &mut i, opc)?;
                        let Token::Number(s) = next else {
                            eyre::bail!("expected numeric immediate for {opc}, got {next:?}");
                        };
                        let n: u8 =
                            s.parse().map_err(|_| eyre::eyre!("invalid {opc} immediate: {s:?}"))?;
                        let raw = encode_single(n).ok_or_else(|| {
                            eyre::eyre!("{opc} index {n} out of valid range [17, 235]")
                        })?;
                        items.push(Item::Inst(Inst {
                            opcode,
                            imm: Some(Imm::Number(U256::from(raw))),
                            push_kind: PushKind::Fixed(1),
                        }));
                    } else if opcode == op::EXCHANGE {
                        let next = expect_token(tokens, &mut i, opc)?;
                        let Token::Pair(n_str, m_str) = next else {
                            eyre::bail!("EXCHANGE requires n,m format (e.g. `1,2`), got {next:?}");
                        };
                        let n: u8 = n_str
                            .parse()
                            .map_err(|_| eyre::eyre!("invalid EXCHANGE index: {n_str:?}"))?;
                        let m: u8 = m_str
                            .parse()
                            .map_err(|_| eyre::eyre!("invalid EXCHANGE index: {m_str:?}"))?;
                        let raw = encode_pair(n, m).ok_or_else(|| {
                            eyre::eyre!("EXCHANGE pair ({n}, {m}) cannot be encoded")
                        })?;
                        items.push(Item::Inst(Inst {
                            opcode,
                            imm: Some(Imm::Number(U256::from(raw))),
                            push_kind: PushKind::Fixed(1),
                        }));
                    } else {
                        let imm_len = opc.info().immediate_size();
                        if imm_len > 0 {
                            let imm = expect_imm(tokens, &mut i, opc)?;
                            items.push(Item::Inst(Inst {
                                opcode,
                                imm: Some(imm),
                                push_kind: PushKind::Fixed(imm_len),
                            }));
                        } else {
                            if let Some(Token::Number(_)) = tokens.get(i) {
                                eyre::bail!("unexpected immediate for opcode {opc}");
                            }
                            items.push(Item::Inst(Inst {
                                opcode,
                                imm: None,
                                push_kind: PushKind::None,
                            }));
                        }
                    }
                }
            }
            other => eyre::bail!("unexpected token: {other:?}"),
        }
    }
    Ok(items)
}

/// Consume the next token, returning an error if the stream is exhausted.
fn expect_token<'a>(
    tokens: &[Token<'a>],
    i: &mut usize,
    ctx: impl std::fmt::Display,
) -> Result<Token<'a>> {
    let tok =
        tokens.get(*i).cloned().ok_or_else(|| eyre::eyre!("missing immediate for opcode {ctx}"))?;
    *i += 1;
    Ok(tok)
}

/// Consume the next token as an immediate (number or label ref).
fn expect_imm<'a>(
    tokens: &[Token<'a>],
    i: &mut usize,
    ctx: impl std::fmt::Display,
) -> Result<Imm<'a>> {
    let tok = expect_token(tokens, i, &ctx)?;
    match tok {
        Token::Number(s) => {
            let n: U256 = s.parse().map_err(|_| eyre::eyre!("invalid immediate: {s:?}"))?;
            Ok(Imm::Number(n))
        }
        Token::LabelRef(name) => Ok(Imm::Label(name)),
        other => eyre::bail!("expected immediate for {ctx}, got {other:?}"),
    }
}

// ———————————————————————————————————————————————————————————————————————
// Layout and emit
// ———————————————————————————————————————————————————————————————————————

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
    let mut auto_label_indices = Vec::new();
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
    let mut auto_widths = vec![0u8; items.len()];
    let mut label_pcs = HashMap::<&str, usize>::default();

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
                let target_pc =
                    *label_pcs.get(name).ok_or_else(|| eyre::eyre!("undefined label: {name:?}"))?;
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
        let code = parse_asm(
            "
            PUSH %target
            JUMP
        target:
            JUMPDEST
            STOP
        ",
        )
        .unwrap();
        assert_eq!(code, vec![op::PUSH1, 3, op::JUMP, op::JUMPDEST, op::STOP]);
    }

    #[test]
    fn label_backward_ref() {
        let code = parse_asm(
            "
        target:
            JUMPDEST
            PUSH %target
            JUMP
        ",
        )
        .unwrap();
        assert_eq!(code, vec![op::JUMPDEST, op::PUSH0, op::JUMP]);
    }

    #[test]
    fn label_fixed_width() {
        let code = parse_asm(
            "
            PUSH1 %target
            JUMP
        target:
            JUMPDEST
            STOP
        ",
        )
        .unwrap();
        assert_eq!(code, vec![op::PUSH1, 3, op::JUMP, op::JUMPDEST, op::STOP]);
    }

    #[test]
    fn multiple_labels_same_pc() {
        let code = parse_asm(
            "
        a:
        b:
            JUMPDEST
            PUSH %a
            PUSH %b
            STOP
        ",
        )
        .unwrap();
        assert_eq!(code, vec![op::JUMPDEST, op::PUSH0, op::PUSH0, op::STOP]);
    }

    #[test]
    fn dupn() {
        // Decoded index 17 → raw byte 0x00.
        assert_eq!(parse_asm("DUPN 17").unwrap(), vec![op::DUPN, 0x00]);
        // Decoded index 108 → raw byte 128.
        assert_eq!(parse_asm("DUPN 108").unwrap(), vec![op::DUPN, 128]);
        // Index 16 is out of range.
        assert!(parse_asm("DUPN 16").is_err());
        // Index 0 is out of range.
        assert!(parse_asm("DUPN 0").is_err());
        // 236 is out of range.
        assert!(parse_asm("DUPN 236").is_err());
        // Non-numeric.
        assert!(parse_asm("DUPN abc").is_err());
    }

    #[test]
    fn swapn() {
        assert_eq!(parse_asm("SWAPN 17").unwrap(), vec![op::SWAPN, 0x00]);
        assert_eq!(parse_asm("SWAPN 108").unwrap(), vec![op::SWAPN, 128]);
        assert!(parse_asm("SWAPN 16").is_err());
        assert!(parse_asm("SWAPN 0").is_err());
    }

    #[test]
    fn exchange() {
        // (1, 2) → raw byte 0x01.
        assert_eq!(parse_asm("EXCHANGE 1,2").unwrap(), vec![op::EXCHANGE, 0x01]);
        // (1, 14) uses case 2 (q >= r).
        assert!(parse_asm("EXCHANGE 1,14").is_ok());
        // (2, 1) cannot be encoded.
        assert!(parse_asm("EXCHANGE 2,1").is_err());
        // (0, 1) is invalid (zero index).
        assert!(parse_asm("EXCHANGE 0,1").is_err());
        // Missing comma.
        assert!(parse_asm("EXCHANGE 1").is_err());
        // Non-numeric.
        assert!(parse_asm("EXCHANGE a,b").is_err());
    }

    #[test]
    fn slotnum() {
        assert_eq!(parse_asm("SLOTNUM").unwrap(), vec![op::SLOTNUM]);
    }

    #[test]
    fn undefined_label() {
        assert!(parse_asm("PUSH %missing JUMP").is_err());
    }

    #[test]
    fn empty_label() {
        assert!(parse_asm(": STOP").is_err());
    }

    #[test]
    fn empty_label_ref() {
        assert!(parse_asm("PUSH % JUMP").is_err());
    }

    #[test]
    fn define_macro() {
        let code = parse_asm(
            "
            #define TWO PUSH 2
            TWO
            TWO
            ADD
        ",
        )
        .unwrap();
        assert_eq!(code, vec![op::PUSH1, 2, op::PUSH1, 2, op::ADD]);
    }

    #[test]
    fn builtin_ret_word() {
        let code = parse_asm("CALLVALUE RET_WORD").unwrap();
        assert_eq!(
            code,
            vec![op::CALLVALUE, op::PUSH0, op::MSTORE, op::PUSH1, 0x20, op::PUSH0, op::RETURN]
        );
    }

    #[test]
    fn define_override_builtin() {
        let code = parse_asm(
            "
            #define RET_WORD STOP
            RET_WORD
        ",
        )
        .unwrap();
        assert_eq!(code, vec![op::STOP]);
    }

    #[test]
    fn define_with_args() {
        let code = parse_asm(
            "
            #define PUSH_TWO(a, b) PUSH $a PUSH $b
            PUSH_TWO(1, 2)
            ADD
        ",
        )
        .unwrap();
        assert_eq!(code, vec![op::PUSH1, 1, op::PUSH1, 2, op::ADD]);
    }

    #[test]
    fn define_single_arg() {
        let code = parse_asm(
            "
            #define PUSH_AND_STORE(val) PUSH $val PUSH0 MSTORE
            PUSH_AND_STORE(0x42)
        ",
        )
        .unwrap();
        assert_eq!(code, vec![op::PUSH1, 0x42, op::PUSH0, op::MSTORE]);
    }

    #[test]
    fn define_missing_args() {
        assert!(
            parse_asm(
                "
            #define FOO(a) PUSH $a
            FOO
        "
            )
            .is_err()
        );
    }

    #[test]
    fn define_wrong_arg_count() {
        assert!(
            parse_asm(
                "
            #define FOO(a, b) PUSH $a PUSH $b
            FOO(1)
        "
            )
            .is_err()
        );
    }

    #[test]
    fn define_empty_name() {
        assert!(parse_asm("#define").is_err());
        assert!(parse_asm("#define  ").is_err());
    }

    #[test]
    fn define_invalid_name() {
        assert!(parse_asm("#define foo-bar STOP").is_err());
    }
}

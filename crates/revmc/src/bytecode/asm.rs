//! EVM bytecode assembler.

use crate::{
    U256, encode_pair, encode_single,
    eyre::{self, Result},
};
use revm_bytecode::opcode::{self as op, OpCode};
use revm_primitives::map::HashMap;
use std::cmp::Ordering;

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
    let tokens = preprocess(s)?;
    let items = parse_items(&tokens)?;
    layout_and_emit(&items)
}

// ———————————————————————————————————————————————————————————————————————
// Tokenizer
// ———————————————————————————————————————————————————————————————————————

/// A token produced by the tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    /// An identifier (opcode name, macro name, etc.).
    Ident(String),
    /// A label definition (`name:`).
    Label(String),
    /// A label reference (`%name`).
    LabelRef(String),
    /// A numeric literal.
    Number(U256),
    /// A comma.
    Comma,
    /// Opening parenthesis.
    LParen,
    /// Closing parenthesis.
    RParen,
    /// A macro parameter reference (`$name`).
    ParamRef(String),
    /// Unrecognized character.
    Unknown(char),
}

/// Character-by-character tokenizer over source text.
struct Tokenizer<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn remaining(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn peek_char(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn skip_whitespace(&mut self) {
        let rest = self.remaining();
        let trimmed = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
        self.advance(rest.len() - trimmed.len());
    }

    fn skip_line(&mut self) {
        if let Some(nl) = self.remaining().find('\n') {
            self.advance(nl + 1);
        } else {
            self.pos = self.src.len();
        }
    }

    /// Read a contiguous word of alphanumeric/underscore characters.
    fn read_word(&mut self) -> String {
        let rest = self.remaining();
        let end = rest.find(|c: char| !c.is_ascii_alphanumeric() && c != '_').unwrap_or(rest.len());
        let word = rest[..end].to_string();
        self.advance(end);
        word
    }

    /// Read a numeric literal (decimal or 0x hex).
    fn read_number(&mut self) -> Token {
        let rest = self.remaining();
        let end = if rest.starts_with("0x") || rest.starts_with("0X") {
            2 + rest[2..]
                .find(|c: char| !c.is_ascii_hexdigit())
                .unwrap_or(rest.len() - 2)
        } else {
            rest.find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len())
        };
        let s = &rest[..end];
        self.advance(end);
        match s.parse::<U256>() {
            Ok(n) => Token::Number(n),
            Err(_) => Token::Unknown(s.chars().next().unwrap_or('?')),
        }
    }
}

impl Iterator for Tokenizer<'_> {
    type Item = Token;

    fn next(&mut self) -> Option<Token> {
        loop {
            self.skip_whitespace();
            let c = self.peek_char()?;

            match c {
                ';' => self.skip_line(),

                '%' => {
                    self.advance(1);
                    let name = self.read_word();
                    return Some(if name.is_empty() {
                        Token::Unknown('%')
                    } else {
                        Token::LabelRef(name)
                    });
                }

                '$' => {
                    self.advance(1);
                    let name = self.read_word();
                    return Some(if name.is_empty() {
                        Token::Unknown('$')
                    } else {
                        Token::ParamRef(name)
                    });
                }

                ',' => {
                    self.advance(1);
                    return Some(Token::Comma);
                }
                '(' => {
                    self.advance(1);
                    return Some(Token::LParen);
                }
                ')' => {
                    self.advance(1);
                    return Some(Token::RParen);
                }

                '0'..='9' => return Some(self.read_number()),

                _ if c.is_ascii_alphabetic() || c == '_' => {
                    let word = self.read_word();
                    if self.peek_char() == Some(':') {
                        self.advance(1);
                        return Some(Token::Label(word));
                    }
                    return Some(Token::Ident(word));
                }

                ':' => {
                    self.advance(1);
                    return Some(Token::Label(String::new()));
                }

                other => {
                    self.advance(other.len_utf8());
                    return Some(Token::Unknown(other));
                }
            }
        }
    }
}

// ———————————————————————————————————————————————————————————————————————
// Preprocessor (#define macros)
// ———————————————————————————————————————————————————————————————————————

/// A macro definition: parameter names and body tokens.
struct MacroDef {
    params: Vec<String>,
    body: Vec<Token>,
}

/// Builtin macros available in all assembly sources.
fn builtin_macros() -> HashMap<String, MacroDef> {
    let mut m = HashMap::default();
    m.insert(
        "RET_WORD".into(),
        MacroDef {
            params: vec![],
            body: vec![
                Token::Ident("PUSH0".into()),
                Token::Ident("MSTORE".into()),
                Token::Ident("PUSH1".into()),
                Token::Number(U256::from(0x20)),
                Token::Ident("PUSH0".into()),
                Token::Ident("RETURN".into()),
            ],
        },
    );
    m
}

/// Preprocess source text: extract `#define` directives (line-scoped), tokenize everything
/// with the char-by-char tokenizer, then expand macro invocations on the token stream.
fn preprocess(s: &str) -> Result<Vec<Token>> {
    let mut macros = builtin_macros();

    // Extract #define lines and tokenize their bodies; collect remaining source.
    let mut rest_lines = String::with_capacity(s.len());
    for line in s.lines() {
        let trimmed = line.trim();
        if let Some(after) = trimmed.strip_prefix("#define") {
            parse_define(after, &mut macros)?;
        } else {
            rest_lines.push_str(line);
            rest_lines.push('\n');
        }
    }

    // Tokenize the non-directive source.
    let raw: Vec<Token> = Tokenizer::new(&rest_lines).collect();

    if macros.is_empty() {
        return Ok(raw);
    }

    // Expand macro invocations.
    expand_macros(raw, &macros)
}

/// Parse a `#define` directive body (everything after `#define`) into the macro table.
fn parse_define(after: &str, macros: &mut HashMap<String, MacroDef>) -> Result<()> {
    let mut tokens = Tokenizer::new(after);

    let name = match tokens.next() {
        Some(Token::Ident(name)) => name,
        _ => eyre::bail!("expected macro name after #define"),
    };

    // Optional parameter list: NAME(a, b).
    let mut params = Vec::new();
    if tokens.peek_char() == Some('(') {
        // The tokenizer already consumed the ident; '(' is next.
        // But peek_char looks at the raw source. The ident was consumed, so pos is past it.
        // Let's just check the next token.
        if let Some(Token::LParen) = {
            let saved = tokens.pos;
            let tok = tokens.next();
            if !matches!(tok, Some(Token::LParen)) {
                tokens.pos = saved;
                None::<Token>
            } else {
                tok
            }
        } {
            loop {
                match tokens.next() {
                    Some(Token::Ident(p)) => params.push(p),
                    Some(Token::RParen) => break,
                    _ => break,
                }
                match tokens.next() {
                    Some(Token::Comma) => {}
                    Some(Token::RParen) => break,
                    _ => break,
                }
            }
        }
    }

    // Body: remaining tokens on this line.
    let body: Vec<Token> = tokens.collect();

    macros.insert(name, MacroDef { params, body });
    Ok(())
}

/// Expand macro invocations in a token stream.
fn expand_macros(tokens: Vec<Token>, macros: &HashMap<String, MacroDef>) -> Result<Vec<Token>> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut iter = tokens.into_iter().peekable();

    while let Some(tok) = iter.next() {
        let Token::Ident(ref name) = tok else {
            out.push(tok);
            continue;
        };
        let Some(mac) = macros.get(name.as_str()) else {
            out.push(tok);
            continue;
        };

        if mac.params.is_empty() {
            out.extend(mac.body.iter().cloned());
        } else {
            // Consume `(arg1, arg2, ...)`.
            eyre::ensure!(iter.next() == Some(Token::LParen), "macro {name:?} expects arguments");
            let mut args: Vec<Vec<Token>> = vec![vec![]];
            let mut depth = 1u32;
            loop {
                let t = iter
                    .next()
                    .ok_or_else(|| eyre::eyre!("unclosed '(' in macro invocation {name:?}"))?;
                match &t {
                    Token::LParen => {
                        depth += 1;
                        args.last_mut().unwrap().push(t);
                    }
                    Token::RParen => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                        args.last_mut().unwrap().push(t);
                    }
                    Token::Comma if depth == 1 => {
                        args.push(vec![]);
                    }
                    _ => args.last_mut().unwrap().push(t),
                }
            }
            eyre::ensure!(
                args.len() == mac.params.len(),
                "macro {name:?} expects {} argument(s), got {}",
                mac.params.len(),
                args.len()
            );
            // Substitute $param refs in the body.
            for body_tok in &mac.body {
                if let Token::ParamRef(pname) = body_tok
                    && let Some(idx) = mac.params.iter().position(|p| p == pname)
                {
                    out.extend(args[idx].iter().cloned());
                } else {
                    out.push(body_tok.clone());
                }
            }
        }
    }
    Ok(out)
}

// ———————————————————————————————————————————————————————————————————————
// Parser (tokens → items)
// ———————————————————————————————————————————————————————————————————————

/// A parsed item from the source.
enum Item {
    /// A label definition (`name:`).
    Label(String),
    /// An instruction.
    Inst(Inst),
}

/// A parsed instruction.
struct Inst {
    opcode: u8,
    imm: Option<Imm>,
    push_kind: PushKind,
}

/// An immediate value.
enum Imm {
    /// A numeric literal.
    Number(U256),
    /// A label reference (resolved during layout).
    Label(String),
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
fn parse_items(tokens: &[Token]) -> Result<Vec<Item>> {
    let mut items = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i] {
            Token::Label(name) => {
                eyre::ensure!(!name.is_empty(), "empty label name");
                items.push(Item::Label(name.clone()));
                i += 1;
            }
            Token::Ident(word) => {
                if word == "PUSH" {
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
                        let n = expect_number_u8(tokens, &mut i, opc)?;
                        let raw = encode_single(n).ok_or_else(|| {
                            eyre::eyre!("{opc} index {n} out of valid range [17, 235]")
                        })?;
                        items.push(Item::Inst(Inst {
                            opcode,
                            imm: Some(Imm::Number(U256::from(raw))),
                            push_kind: PushKind::Fixed(1),
                        }));
                    } else if opcode == op::EXCHANGE {
                        let n = expect_number_u8(tokens, &mut i, opc)?;
                        eyre::ensure!(
                            i < tokens.len() && tokens[i] == Token::Comma,
                            "EXCHANGE requires n,m format (e.g. `1,2`)"
                        );
                        i += 1; // comma
                        let m = expect_number_u8(tokens, &mut i, opc)?;
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
                            if matches!(tokens.get(i), Some(Token::Number(_))) {
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
            Token::Unknown(c) => eyre::bail!("unexpected character: {c:?}"),
            other => eyre::bail!("unexpected token: {other:?}"),
        }
    }
    Ok(items)
}

/// Consume the next token as a number and convert to u8.
fn expect_number_u8(tokens: &[Token], i: &mut usize, ctx: impl std::fmt::Display) -> Result<u8> {
    let tok = tokens.get(*i).ok_or_else(|| eyre::eyre!("missing immediate for opcode {ctx}"))?;
    *i += 1;
    match tok {
        Token::Number(n) => {
            let v: u64 =
                n.try_into().map_err(|_| eyre::eyre!("invalid {ctx} immediate: too large"))?;
            u8::try_from(v).map_err(|_| eyre::eyre!("invalid {ctx} immediate: too large"))
        }
        _ => eyre::bail!("expected numeric immediate for {ctx}, got {tok:?}"),
    }
}

/// Consume the next token as an immediate (number or label ref).
fn expect_imm(tokens: &[Token], i: &mut usize, ctx: impl std::fmt::Display) -> Result<Imm> {
    let tok = tokens.get(*i).ok_or_else(|| eyre::eyre!("missing immediate for opcode {ctx}"))?;
    *i += 1;
    match tok {
        Token::Number(n) => Ok(Imm::Number(*n)),
        Token::LabelRef(name) => Ok(Imm::Label(name.clone())),
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
fn layout_and_emit(items: &[Item]) -> Result<Vec<u8>> {
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
        label_pcs.clear();
        let mut pc = 0usize;
        for (i, item) in items.iter().enumerate() {
            match item {
                Item::Label(name) => {
                    label_pcs.insert(name, pc);
                }
                Item::Inst(inst) => {
                    pc += 1;
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

        let mut changed = false;
        for &i in &auto_label_indices {
            if let Item::Inst(inst) = &items[i]
                && let Some(Imm::Label(name)) = &inst.imm
            {
                let target_pc = *label_pcs
                    .get(name.as_str())
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
fn resolve_imm(imm: &Imm, label_pcs: &HashMap<&str, usize>) -> Result<U256> {
    match imm {
        Imm::Number(n) => Ok(*n),
        Imm::Label(name) => {
            let pc = label_pcs
                .get(name.as_str())
                .ok_or_else(|| eyre::eyre!("undefined label: {name:?}"))?;
            Ok(U256::from(*pc))
        }
    }
}

/// Emit a single instruction (no-label fast path).
fn emit_inst_no_labels(inst: &Inst, code: &mut Vec<u8>) -> Result<()> {
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
        // `-` is tokenized as Unknown, so it can't be a macro name.
        assert!(parse_asm("#define - STOP").is_err());
        // `#define` with no name at all.
        assert!(parse_asm("#define (a) STOP").is_err());
    }
}

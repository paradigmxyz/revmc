use super::{Bytecode, InstData, InstFlags, bitvec_as_bytes};
use revm_bytecode::opcode as op;
use revm_primitives::hex;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::fmt;

/// Basic block info collected from bytecode analysis.
struct BlockInfo {
    /// `(block_idx, first_inst, last_inst)` for each block.
    blocks: Vec<(usize, usize, usize)>,
    /// Maps instruction index to block index.
    inst_to_block: FxHashMap<usize, usize>,
}

impl Bytecode<'_> {
    fn collect_blocks(&self) -> BlockInfo {
        let mut blocks = Vec::new();
        let mut inst_to_block = FxHashMap::default();
        let mut block_idx = 0usize;
        let mut need_header = true;
        for (inst, data) in self.iter_all_insts() {
            if data.is_dead_code() {
                continue;
            }
            if !data.section.is_empty() || need_header {
                inst_to_block.insert(inst, block_idx);
                blocks.push((block_idx, inst, inst));
                block_idx += 1;
                need_header = false;
            }
            if let Some(b) = blocks.last_mut() {
                b.2 = inst;
            }
            if data.is_branching() {
                need_header = true;
            }
        }
        BlockInfo { blocks, inst_to_block }
    }
}

impl fmt::Display for Bytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use std::fmt::Write;

        let info = self.collect_blocks();

        // First pass: collect lines with their text and comments.
        let mut lines: Vec<(String, String)> = Vec::new();

        lines.push((
            String::new(),
            format!(
                "spec_id={}, has_dynamic_jumps={}, may_suspend={}",
                self.spec_id, self.has_dynamic_jumps, self.may_suspend,
            ),
        ));
        lines.push((String::new(), String::new()));

        for &(block_idx, first_inst, last_inst) in &info.blocks {
            // Blank line between blocks.
            if first_inst > 0 {
                lines.push((String::new(), String::new()));
            }

            // Block header.
            let first = self.inst(first_inst);
            let mut header = format!("bb{block_idx}:");
            let mut comment = String::new();
            if !first.section.is_empty() {
                write!(
                    comment,
                    "gas={}, stack_in={}, max_growth={}",
                    first.section.gas_cost, first.section.inputs, first.section.max_growth,
                )
                .unwrap();
            }
            // Pad header to align with indented instructions.
            while header.len() < 2 {
                header.push(' ');
            }
            lines.push((header, comment));

            // Instructions.
            for inst in first_inst..=last_inst {
                let data = self.inst(inst);
                if data.is_dead_code() {
                    continue;
                }

                // Instruction text.
                let mut text = String::from("  ");
                let opcode = data.to_op_in(self);
                write!(text, "{opcode}").unwrap();
                if data.flags.contains(InstFlags::INVALID_JUMP) {
                    text.push_str(" -> INVALID");
                } else if data.is_legacy_static_jump() {
                    let target = data.data as usize;
                    match info.inst_to_block.get(&target) {
                        Some(b) => write!(text, " bb{b}").unwrap(),
                        None => write!(text, " inst {target}").unwrap(),
                    }
                }

                // Comment with pc and flags/behavior.
                let mut comment = format!("pc={}", data.pc);
                let flags = data.flags;
                if flags.contains(InstFlags::SKIP_LOGIC) {
                    comment.push_str(", skip");
                }
                if flags.contains(InstFlags::DEAD_CODE) {
                    comment.push_str(", dead");
                }
                if flags.contains(InstFlags::DISABLED) {
                    comment.push_str(", disabled");
                }
                if flags.contains(InstFlags::UNKNOWN) {
                    comment.push_str(", unknown");
                }
                if flags.contains(InstFlags::INVALID_JUMP) {
                    comment.push_str(", invalid_jump");
                }
                if data.may_suspend() {
                    comment.push_str(", suspends");
                }
                if data.is_reachable_jumpdest(self.has_dynamic_jumps) {
                    comment.push_str(", reachable");
                }

                lines.push((text, comment));
            }
        }

        // Second pass: find max text width and write with aligned comments.
        let max_text_width = lines.iter().map(|(t, _)| t.len()).max().unwrap_or(0);
        let comment_col = max_text_width.clamp(4, 20);
        for (text, comment) in &lines {
            if text.is_empty() && comment.is_empty() {
                writeln!(f)?;
            } else if comment.is_empty() {
                writeln!(f, "{text}")?;
            } else {
                writeln!(f, "{text:<comment_col$} ; {comment}")?;
            }
        }

        Ok(())
    }
}

impl fmt::Debug for Bytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bytecode")
            .field("code", &hex::encode(self.code))
            .field("insts", &self.insts)
            .field("jumpdests", &hex::encode(bitvec_as_bytes(&self.jumpdests)))
            .field("spec_id", &self.spec_id)
            .field("has_dynamic_jumps", &self.has_dynamic_jumps)
            .field("may_suspend", &self.may_suspend)
            .finish()
    }
}

impl fmt::Debug for InstData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InstData")
            .field("opcode", &self.to_op())
            .field("flags", &format_args!("{:?}", self.flags))
            .field("data", &self.data)
            .field("pc", &self.pc)
            .field("section", &self.section)
            .finish()
    }
}

impl<'a> Bytecode<'a> {
    /// Writes the bytecode as a DOT graph to the given writer.
    #[doc(hidden)]
    pub fn write_dot<W: fmt::Write>(&self, w: &mut W) -> fmt::Result {
        let info = self.collect_blocks();

        writeln!(w, "digraph bytecode {{")?;
        writeln!(w, "  graph [bgcolor=\"#1a1a2e\" rankdir=TB];")?;
        writeln!(
            w,
            "  node [shape=Mrecord fontname=\"Courier\" fontsize=10 \
             style=filled fillcolor=\"#16213e\" fontcolor=\"#e0e0e0\" \
             color=\"#0f3460\" penwidth=1.5];"
        )?;
        writeln!(
            w,
            "  edge [fontname=\"Courier\" fontsize=9 color=\"#555577\" \
             fontcolor=\"#8888aa\"];"
        )?;

        // Emit nodes.
        for &(block_idx, first_inst, last_inst) in &info.blocks {
            let last = self.inst(last_inst);
            let first = self.inst(first_inst);

            // Color based on block terminator.
            let (fill, border) = if last.is_diverging() && !last.is_legacy_jump() {
                ("#2d1b2e", "#e94560") // exit blocks: dark red
            } else if last.is_legacy_jump() {
                ("#1a2340", "#53a8b6") // branching blocks: teal
            } else if first.is_reachable_jumpdest(self.has_dynamic_jumps) {
                ("#1a2e1a", "#5cdb95") // jump targets: green
            } else {
                ("#16213e", "#0f3460") // default: dark blue
            };

            write!(
                w,
                "  bb{block_idx} [fillcolor=\"{fill}\" color=\"{border}\" \
                 label=\"{{bb{block_idx}",
            )?;

            if !first.section.is_empty() {
                write!(
                    w,
                    " | gas={} in={} growth={}",
                    first.section.gas_cost, first.section.inputs, first.section.max_growth,
                )?;
            }

            write!(w, " |")?;
            for inst in first_inst..=last_inst {
                let data = self.inst(inst);
                if data.is_dead_code() {
                    continue;
                }
                let opcode = data.to_op_in(self);
                let op_str = abbreviate_hex(&opcode.to_string())
                    .replace('>', "\\>")
                    .replace('<', "\\<");
                write!(w, "{op_str}\\l")?;
            }
            writeln!(w, "}}\"];")?;
        }

        // Emit edges.
        for (i, &(block_idx, _, last_inst)) in info.blocks.iter().enumerate() {
            let last = self.inst(last_inst);

            // Jump edge.
            if last.is_legacy_static_jump() && !last.flags.contains(InstFlags::INVALID_JUMP) {
                let target = last.data as usize;
                if let Some(&target_block) = info.inst_to_block.get(&target) {
                    let (label, color) = if last.opcode == op::JUMPI {
                        ("true", "#5cdb95")
                    } else {
                        ("", "#53a8b6")
                    };
                    writeln!(
                        w,
                        "  bb{block_idx} -> bb{target_block} \
                         [label=\"{label}\" color=\"{color}\" fontcolor=\"{color}\"];"
                    )?;
                }
            } else if last.is_legacy_jump() && !last.is_legacy_static_jump() {
                writeln!(
                    w,
                    "  bb{block_idx} -> dynamic \
                     [label=\"dynamic\" color=\"#e94560\" fontcolor=\"#e94560\" style=dashed];"
                )?;
            }

            // Fallthrough edge.
            let has_fallthrough = if last.opcode == op::JUMPI {
                true
            } else {
                !last.is_diverging() && last.opcode != op::JUMP
            };
            if has_fallthrough && let Some(&(next_block, _, _)) = info.blocks.get(i + 1) {
                let (label, color) =
                    if last.opcode == op::JUMPI { ("false", "#e94560") } else { ("", "#555577") };
                writeln!(
                    w,
                    "  bb{block_idx} -> bb{next_block} \
                     [label=\"{label}\" color=\"{color}\" fontcolor=\"{color}\"];"
                )?;
            }
        }

        // Dynamic jump table.
        if self.has_dynamic_jumps {
            writeln!(
                w,
                "  dynamic [shape=diamond style=filled fillcolor=\"#2d1b2e\" \
                 color=\"#e94560\" fontcolor=\"#e0e0e0\" \
                 label=\"dynamic\\njump table\"];"
            )?;
            for &(block_idx, first_inst, _) in &info.blocks {
                let first = self.inst(first_inst);
                if first.is_reachable_jumpdest(self.has_dynamic_jumps) {
                    writeln!(
                        w,
                        "  dynamic -> bb{block_idx} \
                         [color=\"#e94560\" style=dashed];"
                    )?;
                }
            }
        }

        writeln!(w, "}}")
    }

    /// Returns the bytecode as a DOT graph string.
    #[cfg(test)]
    fn to_dot(&self) -> String {
        let mut s = String::new();
        self.write_dot(&mut s).unwrap();
        s
    }
}

/// Abbreviates hex strings with repeated leading byte pairs.
/// E.g. `"PUSH32 0xffffffffff...ffe0"` → `"PUSH32 0xff..ffe0"`.
fn abbreviate_hex(s: &str) -> Cow<'_, str> {
    let Some(hex_start) = s.find("0x") else {
        return Cow::Borrowed(s);
    };
    let hex = &s[hex_start + 2..];
    // Need at least 2 byte pairs (4 hex chars) of repetition to abbreviate.
    if hex.len() < 8 {
        return Cow::Borrowed(s);
    }
    let prefix = &hex[..2];
    let run_len = hex
        .as_bytes()
        .chunks(2)
        .take_while(|chunk| chunk.len() == 2 && *chunk == prefix.as_bytes())
        .count();
    if run_len < 4 {
        return Cow::Borrowed(s);
    }
    let suffix = &hex[run_len * 2..];
    Cow::Owned(format!("{}0x{prefix}..{suffix}", &s[..hex_start]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm_bytecode::opcode as op;
    use revm_primitives::hardfork::SpecId;

    fn test_bytecode() -> Bytecode<'static> {
        let code: &[u8] = &[
            op::PUSH1,
            0x01,
            op::PUSH1,
            0x07,
            op::JUMPI,
            op::STOP,
            op::ADD,
            op::JUMPDEST,
            op::PUSH1,
            0x02,
            op::STOP,
        ];
        let mut bytecode = Bytecode::new(code, SpecId::OSAKA);
        bytecode.analyze().unwrap();
        bytecode
    }

    #[test]
    fn display_format() {
        let bytecode = test_bytecode();
        let actual = format!("{bytecode}");
        snapbox::assert_data_eq!(
            actual,
            snapbox::str![[r#"
             ; spec_id=Osaka, has_dynamic_jumps=false, may_suspend=false

bb0:         ; gas=16, stack_in=0, max_growth=2
  PUSH1 0x01 ; pc=0
  PUSH1 0x07 ; pc=2, skip
  JUMPI bb2  ; pc=4

bb1:
  STOP       ; pc=5

bb2:         ; gas=4, stack_in=0, max_growth=1
  JUMPDEST   ; pc=7, reachable
  PUSH1 0x02 ; pc=8
  STOP       ; pc=10

"#]]
        );
    }

    #[test]
    fn dot_format() {
        let bytecode = test_bytecode();
        let dot = bytecode.to_dot();
        assert!(dot.starts_with("digraph bytecode {"));
        assert!(dot.contains("bb0"));
        assert!(dot.contains("bb1"));
        assert!(dot.contains("bb2"));
        assert!(dot.contains("PUSH1 0x01"));
        assert!(dot.contains("JUMPI"));
        assert!(dot.contains("JUMPDEST"));
        assert!(dot.contains("bb0 -> bb2"), "missing true edge");
        assert!(dot.contains("bb0 -> bb1"), "missing false edge");
        assert!(dot.contains("\"true\""), "missing true label");
        assert!(dot.contains("\"false\""), "missing false label");
        assert!(!dot.contains("dynamic"), "unexpected dynamic jump table");
    }
}

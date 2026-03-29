use super::{Bytecode, Inst, InstData, InstFlags, bitvec_as_bytes};
use crate::FxHashMap;
use oxc_index::IndexVec;
use revm_bytecode::opcode as op;
use revm_primitives::hex;
use std::{borrow::Cow, fmt, fmt::Write};

/// Basic block info collected from bytecode analysis.
struct BlockInfo {
    /// `(block_idx, first_inst, last_inst)` for each block.
    blocks: Vec<(usize, Inst, Inst)>,
    /// Maps instruction index to block index.
    inst_to_block: FxHashMap<Inst, usize>,
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
            if !data.stack_section.is_empty() || need_header {
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

    /// Collects formatted lines and builds the inst-to-line map stored in `self.inst_lines`.
    fn collect_lines(&self) -> Vec<(String, String)> {
        let info = self.collect_blocks();
        let mut lines: Vec<(String, String)> = Vec::new();
        let mut inst_lines = IndexVec::<Inst, u32>::from_vec(vec![0u32; self.insts.len()]);

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
            if first_inst.index() > 0 {
                lines.push((String::new(), String::new()));
            }

            // Block header.
            let first = self.inst(first_inst);
            let mut header = format!("bb{block_idx}:");
            let mut comment = String::new();
            if !first.stack_section.is_empty() {
                write!(
                    comment,
                    "stack_in={}, max_growth={}",
                    first.stack_section.inputs, first.stack_section.max_growth,
                )
                .unwrap();
            }
            // Pad header to align with indented instructions.
            while header.len() < 2 {
                header.push(' ');
            }
            lines.push((header, comment));

            // Instructions.
            for i in first_inst.index()..=last_inst.index() {
                let inst = Inst::from_usize(i);
                let data = self.inst(inst);
                if data.is_dead_code() {
                    continue;
                }

                // 1-based line number (lines.len() is the 0-based index of the next line).
                inst_lines[inst] = lines.len() as u32 + 1;

                // Instruction text.
                let mut text = String::from("  ");
                let opcode = data.to_op_in(self);
                write!(text, "{opcode}").unwrap();
                if data.flags.contains(InstFlags::INVALID_JUMP) {
                    text.push_str(" INVALID");
                } else if data.flags.contains(InstFlags::MULTI_JUMP) {
                    if let Some(targets) = self.multi_jump_targets(inst) {
                        text.push(' ');
                        for (i, &t) in targets.iter().enumerate() {
                            if i > 0 {
                                text.push_str(", ");
                            }
                            match info.inst_to_block.get(&t) {
                                Some(b) => write!(text, " bb{b}").unwrap(),
                                None => write!(text, " inst {t}").unwrap(),
                            }
                        }
                    }
                } else if data.is_static_jump() {
                    let target = Inst::from_usize(data.data as usize);
                    match info.inst_to_block.get(&target) {
                        Some(b) => write!(text, " bb{b}").unwrap(),
                        None => write!(text, " inst {target}").unwrap(),
                    }
                }

                // Comment with pc and flags/behavior.
                let mut comment = format!("pc={}", data.pc);
                if !data.gas_section.is_empty() {
                    write!(comment, ", gas={}", data.gas_section.gas_cost).unwrap();
                }
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
                if flags.contains(InstFlags::BLOCK_RESOLVED_JUMP) {
                    comment.push_str(", block_resolved");
                }
                if flags.contains(InstFlags::MULTI_JUMP) {
                    comment.push_str(", multi_jump");
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

        *self.inst_lines.borrow_mut() = inst_lines;
        lines
    }
}

impl fmt::Display for Bytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lines = self.collect_lines();

        // Find max text width and write with aligned comments.
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
            .field("gas_section", &self.gas_section)
            .field("stack_section", &self.stack_section)
            .finish()
    }
}

// DOT graph colors.
mod dot_colors {
    const DARK_NAVY: &str = "#1a1a2e";
    const DARK_BLUE: &str = "#16213e";
    const BLUE: &str = "#0f3460";
    const DARK_TEAL: &str = "#1a2340";
    const TEAL: &str = "#53a8b6";
    const GREEN: &str = "#5cdb95";
    const DARK_GREEN: &str = "#1a2e1a";
    const DARK_ORANGE: &str = "#2e2416";
    const ORANGE: &str = "#e0a030";
    const DARK_RED: &str = "#2d1b2e";
    const RED: &str = "#e94560";
    const GRAY: &str = "#555577";
    const LIGHT_GRAY: &str = "#e0e0e0";

    pub(super) const BG: &str = DARK_NAVY;
    pub(super) const TEXT: &str = LIGHT_GRAY;
    // Default node.
    pub(super) const NODE_FILL: &str = DARK_BLUE;
    pub(super) const NODE_BORDER: &str = BLUE;
    // Reverting/error blocks.
    pub(super) const REVERT_FILL: &str = DARK_RED;
    pub(super) const REVERT_BORDER: &str = RED;
    // Non-reverting exit blocks (STOP, RETURN).
    pub(super) const EXIT_FILL: &str = DARK_GREEN;
    pub(super) const EXIT_BORDER: &str = GREEN;
    // Suspending blocks (CALL, CREATE, ...).
    pub(super) const SUSPEND_FILL: &str = DARK_ORANGE;
    pub(super) const SUSPEND_BORDER: &str = ORANGE;
    // Branching blocks.
    pub(super) const BRANCH_FILL: &str = DARK_TEAL;
    pub(super) const BRANCH_BORDER: &str = TEAL;
    // Edges.
    pub(super) const EDGE: &str = GRAY;
    pub(super) const EDGE_JUMP: &str = TEAL;
    pub(super) const EDGE_COND_JUMP: &str = GREEN;
    pub(super) const EDGE_FALSE: &str = RED;
}

impl<'a> Bytecode<'a> {
    /// Writes the bytecode as a DOT graph to the given writer.
    #[doc(hidden)]
    pub fn write_dot<W: fmt::Write>(&self, w: &mut W) -> fmt::Result {
        use dot_colors::*;

        let info = self.collect_blocks();

        writeln!(w, "digraph bytecode {{")?;
        writeln!(w, "  graph [bgcolor=\"{BG}\" rankdir=TB];")?;
        writeln!(
            w,
            "  node [shape=Mrecord fontname=\"Courier\" fontsize=10 \
             style=filled fillcolor=\"{NODE_FILL}\" fontcolor=\"{TEXT}\" \
             color=\"{NODE_BORDER}\" penwidth=1.5];"
        )?;
        writeln!(w, "  edge [fontname=\"Courier\" fontsize=9 color=\"{EDGE}\"];")?;

        // Emit nodes.
        for &(block_idx, first_inst, last_inst) in &info.blocks {
            let last = self.inst(last_inst);
            let first = self.inst(first_inst);

            // Color based on block terminator.
            let (fill, border) = if matches!(last.opcode, op::STOP | op::RETURN) {
                (EXIT_FILL, EXIT_BORDER)
            } else if last.is_diverging() {
                (REVERT_FILL, REVERT_BORDER)
            } else if last.may_suspend() {
                (SUSPEND_FILL, SUSPEND_BORDER)
            } else if last.is_jump() {
                (BRANCH_FILL, BRANCH_BORDER)
            } else {
                (NODE_FILL, NODE_BORDER)
            };

            write!(
                w,
                "  bb{block_idx} [fillcolor=\"{fill}\" color=\"{border}\" \
                 label=\"{{bb{block_idx}",
            )?;

            if !first.stack_section.is_empty() {
                write!(
                    w,
                    " | in={} growth={}",
                    first.stack_section.inputs, first.stack_section.max_growth
                )?;
            }

            write!(w, " |")?;
            for i in first_inst.index()..=last_inst.index() {
                let data = self.inst(Inst::from_usize(i));
                if data.is_dead_code() {
                    continue;
                }
                let opcode = data.to_op_in(self);
                let mut op_str =
                    abbreviate_hex(&opcode.to_string()).replace('>', "\\>").replace('<', "\\<");
                if !data.gas_section.is_empty() {
                    write!(op_str, " [g={}]", data.gas_section.gas_cost).unwrap();
                }
                write!(w, "{op_str}\\l")?;
            }
            writeln!(w, "}}\"];")?;
        }

        // Emit edges.
        for (i, &(block_idx, _, last_inst)) in info.blocks.iter().enumerate() {
            let last = self.inst(last_inst);

            // Jump edge.
            if last.flags.contains(InstFlags::MULTI_JUMP) {
                if let Some(targets) = self.multi_jump_targets(last_inst) {
                    for &t in targets {
                        if let Some(&target_block) = info.inst_to_block.get(&t) {
                            let color = "#e2a93b";
                            let extra = if block_idx == target_block {
                                " tailport=s headport=e constraint=false"
                            } else if target_block <= block_idx {
                                " constraint=false"
                            } else {
                                ""
                            };
                            writeln!(
                                w,
                                "  bb{block_idx} -> bb{target_block} \
                                 [label=\"multi\" color=\"{color}\" fontcolor=\"{color}\"{extra}];"
                            )?;
                        }
                    }
                }
            } else if last.is_static_jump() && !last.flags.contains(InstFlags::INVALID_JUMP)
            {
                let target = Inst::from_usize(last.data as usize);
                if let Some(&target_block) = info.inst_to_block.get(&target) {
                    let color = if last.opcode == op::JUMPI { EDGE_COND_JUMP } else { EDGE_JUMP };
                    let extra = if block_idx == target_block {
                        " tailport=s headport=e constraint=false"
                    } else if target_block <= block_idx {
                        " constraint=false"
                    } else {
                        ""
                    };
                    writeln!(w, "  bb{block_idx} -> bb{target_block} [color=\"{color}\"{extra}];")?;
                }
            } else if last.is_jump() && !last.is_static_jump() {
                writeln!(w, "  bb{block_idx} -> dynamic [color=\"{EDGE_FALSE}\" style=dashed];")?;
            }

            // Fallthrough edge.
            let has_fallthrough = last.can_fall_through();
            if has_fallthrough && let Some(&(next_block, _, _)) = info.blocks.get(i + 1) {
                let color = if last.opcode == op::JUMPI { EDGE_FALSE } else { EDGE };
                writeln!(w, "  bb{block_idx} -> bb{next_block} [color=\"{color}\"];")?;
            }
        }

        // Dynamic jump table.
        if self.has_dynamic_jumps {
            writeln!(
                w,
                "  dynamic [shape=diamond style=filled fillcolor=\"{REVERT_FILL}\" \
                 color=\"{REVERT_BORDER}\" fontcolor=\"{TEXT}\" \
                 label=\"dynamic\\njump table\"];"
            )?;
            for &(block_idx, first_inst, _) in &info.blocks {
                let first = self.inst(first_inst);
                if first.is_reachable_jumpdest(self.has_dynamic_jumps) {
                    writeln!(
                        w,
                        "  dynamic -> bb{block_idx} [color=\"{EDGE_FALSE}\" style=dashed];"
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

    /// Test bytecode with SSTORE (splits gas but not stack), a loop (back-edge), and CALL
    /// (suspending instruction that splits both gas and stack sections).
    fn test_bytecode() -> Bytecode<'static> {
        #[rustfmt::skip]
        let code: &[u8] = &[
            op::PUSH1, 0x03,
            op::JUMP,
            op::JUMPDEST,
            op::PUSH1, 0x01,
            op::PUSH1, 0x00,
            op::SSTORE,
            op::PUSH1, 0x01,
            op::PUSH1, 0x03,
            op::JUMPI,
            op::PUSH1, 0x00,
            op::PUSH1, 0x00,
            op::PUSH1, 0x00,
            op::PUSH1, 0x00,
            op::PUSH1, 0x00,
            op::PUSH1, 0x42,
            op::PUSH2, 0xff, 0xff,
            op::CALL,
            op::POP,
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
               ; spec_id=Osaka, has_dynamic_jumps=false, may_suspend=true

bb0:           ; stack_in=0, max_growth=1
  PUSH1 0x03   ; pc=0, gas=11, skip
  JUMP bb1     ; pc=2

bb1:           ; stack_in=0, max_growth=2
  JUMPDEST     ; pc=3, gas=7, reachable
  PUSH1 0x01   ; pc=4
  PUSH1 0x00   ; pc=6
  SSTORE       ; pc=8
  PUSH1 0x01   ; pc=9, gas=16
  PUSH1 0x03   ; pc=11, skip
  JUMPI bb1    ; pc=13

bb2:           ; stack_in=0, max_growth=7
  PUSH1 0x00   ; pc=14, gas=121
  PUSH1 0x00   ; pc=16
  PUSH1 0x00   ; pc=18
  PUSH1 0x00   ; pc=20
  PUSH1 0x00   ; pc=22
  PUSH1 0x42   ; pc=24
  PUSH2 0xffff ; pc=26
  CALL         ; pc=29, suspends

bb3:           ; stack_in=1, max_growth=0
  POP          ; pc=30, gas=2
  STOP         ; pc=31

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
        assert!(dot.contains("bb3"));
        // SSTORE present in bb1.
        assert!(dot.contains("SSTORE"), "missing SSTORE");
        // SSTORE splits gas sections: two [g=] annotations in bb1.
        assert!(dot.contains("[g=7]"), "missing first gas section");
        assert!(dot.contains("[g=16]"), "missing second gas section");
        // CALL present in bb2, suspends and splits into bb3.
        assert!(dot.contains("CALL"), "missing CALL");
        assert!(dot.contains("[g=121]"), "missing CALL gas section");
        // bb0 -> bb1 (unconditional jump).
        assert!(dot.contains("bb0 -> bb1"), "missing jump edge");
        // bb1 -> bb1 (loop back-edge).
        assert!(dot.contains("bb1 -> bb1"), "missing loop back-edge");
        // bb1 -> bb2 (fallthrough on false).
        assert!(dot.contains("bb1 -> bb2"), "missing false edge");
        // bb2 -> bb3 (fallthrough after CALL).
        assert!(dot.contains("bb2 -> bb3"), "missing CALL fallthrough edge");
        assert!(!dot.contains("dynamic"), "unexpected dynamic jump table");
    }

    #[test]
    fn abbreviate_hex_repeated() {
        // 32 repeated ff bytes + suffix.
        assert_eq!(
            abbreviate_hex(
                "PUSH32 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe0"
            ),
            "PUSH32 0xff..e0",
        );
        assert_eq!(
            abbreviate_hex(
                "PUSH32 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeeee"
            ),
            "PUSH32 0xff..eeee",
        );
        assert_eq!(
            abbreviate_hex(
                "PUSH32 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeeeeee"
            ),
            "PUSH32 0xff..eeeeee",
        );
        // 32 repeated 00 bytes + suffix.
        assert_eq!(
            abbreviate_hex(
                "PUSH32 0x0000000000000000000000000000000000000000000000000000000000000001"
            ),
            "PUSH32 0x00..01",
        );
        // All repeated, no suffix.
        assert_eq!(
            abbreviate_hex(
                "PUSH32 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
            ),
            "PUSH32 0xff..",
        );
    }

    #[test]
    fn abbreviate_hex_short() {
        // Too few repeated pairs (< 4).
        assert_eq!(abbreviate_hex("PUSH3 0xffffff"), "PUSH3 0xffffff");
        // No repetition.
        assert_eq!(abbreviate_hex("PUSH4 0x30627b7c"), "PUSH4 0x30627b7c");
        // Short value.
        assert_eq!(abbreviate_hex("PUSH1 0x40"), "PUSH1 0x40");
        // No hex at all.
        assert_eq!(abbreviate_hex("STOP"), "STOP");
    }
}

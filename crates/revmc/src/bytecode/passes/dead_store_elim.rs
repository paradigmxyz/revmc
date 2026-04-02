//! Intra-block dead store elimination.
//!
//! Walks each basic block backward to determine which stack positions are live (will be read
//! by a later instruction or at the block exit). Instructions that are pure and whose outputs
//! are all dead are marked [`InstFlags::SKIP_LOGIC`].
//!
//! SWAP and DUP are handled conservatively: SWAP rearranges live positions and DUP copies a
//! value, but neither is eliminated because they affect stack layout that other instructions
//! depend on.

use super::StackSection;
use crate::bytecode::{Bytecode, Inst, InstFlags};
use bitvec::vec::BitVec;
use revm_bytecode::opcode as op;

/// Returns `true` if the opcode is pure: no side effects beyond stack I/O, no dynamic gas.
///
/// Pure instructions can be eliminated when their outputs are dead (never read).
fn is_pure_opcode(opcode: u8) -> bool {
    matches!(
        opcode,
        op::ADD
            | op::MUL
            | op::SUB
            | op::DIV
            | op::SDIV
            | op::MOD
            | op::SMOD
            | op::ADDMOD
            | op::MULMOD
            | op::SIGNEXTEND
            | op::LT
            | op::GT
            | op::SLT
            | op::SGT
            | op::EQ
            | op::ISZERO
            | op::AND
            | op::OR
            | op::XOR
            | op::NOT
            | op::BYTE
            | op::SHL
            | op::SHR
            | op::SAR
            | op::CLZ
            | op::PUSH0..=op::PUSH32
            | op::DUP1..=op::DUP16
            | op::SWAP1..=op::SWAP16
            | op::POP
            | op::PC
            | op::CODESIZE
            // EXP excluded: has dynamic gas.
    )
}

impl Bytecode<'_> {
    /// Intra-block dead store elimination.
    ///
    /// Walks each block backward to determine which stack positions are live (will be read
    /// by a later instruction or at the block exit). Instructions that are pure and whose
    /// outputs are all dead are marked `SKIP_LOGIC`.
    #[instrument(name = "dse", level = "debug", skip_all)]
    pub(crate) fn dead_store_elim(&mut self) {
        let mut eliminated = 0u32;

        for bid in self.cfg.blocks.indices() {
            let block = &self.cfg.blocks[bid];
            let insts: Vec<Inst> = block.insts().collect();
            if insts.is_empty() {
                continue;
            }

            // Compute the stack height at each instruction boundary by walking forward.
            // heights[i] = stack height *before* executing insts[i].
            // heights[insts.len()] = stack height after the last instruction.
            let mut heights: Vec<i32> = Vec::with_capacity(insts.len() + 1);
            let entry_height = {
                let section =
                    StackSection::from_stack_io(insts.iter().map(|&i| self.insts[i].stack_io()));
                section.inputs as i32
            };
            heights.push(entry_height);
            for &inst in &insts {
                let data = &self.insts[inst];
                if data.is_dead_code() || data.flags.contains(InstFlags::SKIP_LOGIC) {
                    heights.push(*heights.last().unwrap());
                    continue;
                }
                let (inp, out) = data.stack_io();
                let h = heights.last().unwrap() - inp as i32 + out as i32;
                heights.push(h);
            }

            // Size the liveness bitvec to the maximum stack height in the block.
            let max_height = *heights.iter().max().unwrap();
            let exit_height = *heights.last().unwrap();
            let live_len = max_height.max(0) as usize;
            let mut live: BitVec = BitVec::repeat(false, live_len);
            // At the block exit, all stack positions are conservatively live.
            for i in 0..exit_height.max(0) as usize {
                live.set(i, true);
            }

            // Walk backward.
            for (idx, &inst) in insts.iter().enumerate().rev() {
                let data = &self.insts[inst];
                if data.is_dead_code() || data.flags.contains(InstFlags::SKIP_LOGIC) {
                    continue;
                }

                let opcode = data.opcode;
                let (inp, out) = data.stack_io();
                let h_before = heights[idx];

                // Stack positions written: [h_before - inp .. h_before - inp + out).
                let write_base = h_before as usize - inp as usize;

                // Check if all outputs are dead.
                if out > 0
                    && is_pure_opcode(opcode)
                    && !data.is_branching()
                    && !data.is_diverging()
                    // Don't eliminate SWAP — it permutes positions, affecting liveness.
                    && !matches!(opcode, op::SWAP1..=op::SWAP16)
                    // Don't eliminate DUP — its copy is needed for stack layout.
                    && !matches!(opcode, op::DUP1..=op::DUP16)
                    // Don't eliminate POP — it has no outputs.
                    && opcode != op::POP
                {
                    let all_dead = (0..out as usize).all(|k| {
                        let pos = write_base + k;
                        pos < live.len() && !live[pos]
                    });

                    if all_dead {
                        self.insts[inst].flags |= InstFlags::SKIP_LOGIC;
                        eliminated += 1;
                        continue;
                    }
                }

                // Update liveness: outputs are defined (kill), inputs are used (gen).
                match opcode {
                    op::SWAP1..=op::SWAP16 => {
                        // SWAP exchanges TOS and TOS-(depth). The two swapped positions
                        // propagate liveness from their post-swap destination to their
                        // pre-swap source. Middle positions pass through unchanged.
                        let depth = (opcode - op::SWAP1 + 1) as usize;
                        let tos = h_before as usize - 1;
                        let other = tos - depth;
                        // After swap: post[tos] ← pre[other], post[other] ← pre[tos].
                        let tos_live = tos < live.len() && live[tos];
                        let other_live = other < live.len() && live[other];
                        if tos < live.len() {
                            live.set(tos, other_live);
                        }
                        if other < live.len() {
                            live.set(other, tos_live);
                        }
                    }
                    op::DUP1..=op::DUP16 => {
                        // DUP pushes a copy of stack[depth]. Kill the new TOS, gen
                        // the source.
                        let depth = (opcode - op::DUP1 + 1) as usize;
                        let new_tos = h_before as usize;
                        let src = h_before as usize - depth;
                        if new_tos < live.len() {
                            live.set(new_tos, false);
                        }
                        if src < live.len() {
                            live.set(src, true);
                        }
                    }
                    op::POP => {
                        // POP discards its input — kill the position instead of
                        // marking it live. This allows the producer to be eliminated.
                        let pos = h_before as usize - 1;
                        if pos < live.len() {
                            live.set(pos, false);
                        }
                    }
                    _ => {
                        // Kill outputs.
                        for k in 0..out as usize {
                            let pos = write_base + k;
                            if pos < live.len() {
                                live.set(pos, false);
                            }
                        }
                        // Gen inputs.
                        for k in 0..inp as usize {
                            let pos = h_before as usize - inp as usize + k;
                            if pos < live.len() {
                                live.set(pos, true);
                            }
                        }
                    }
                }
            }
        }

        if eliminated > 0 {
            debug!(eliminated, "dead stores eliminated");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::block_analysis::tests::{Inst, analyze_asm};
    use crate::bytecode::InstFlags;

    #[test]
    fn push_pop() {
        let bytecode = analyze_asm(
            "
            PUSH1 0x42
            POP
            STOP
        ",
        );
        assert!(
            bytecode.inst(Inst::from_usize(0)).flags.contains(InstFlags::SKIP_LOGIC),
            "PUSH should be skipped (dead store)"
        );
    }

    #[test]
    fn push_push_add_pop() {
        let bytecode = analyze_asm(
            "
            PUSH1 0x03
            PUSH1 0x04
            ADD
            POP
            STOP
        ",
        );
        for (i, name) in [(0, "PUSH 3"), (1, "PUSH 4"), (2, "ADD")] {
            assert!(
                bytecode.inst(Inst::from_usize(i)).flags.contains(InstFlags::SKIP_LOGIC),
                "{name} should be skipped"
            );
        }
    }

    #[test]
    fn partial() {
        let bytecode = analyze_asm(
            "
            PUSH1 0x01
            PUSH1 0x02
            POP
            PUSH0
            MSTORE
            STOP
        ",
        );
        assert!(
            !bytecode.inst(Inst::from_usize(0)).flags.contains(InstFlags::SKIP_LOGIC),
            "PUSH 1 should NOT be skipped"
        );
        assert!(
            bytecode.inst(Inst::from_usize(1)).flags.contains(InstFlags::SKIP_LOGIC),
            "PUSH 2 should be skipped"
        );
    }

    #[test]
    fn swap_preserves_liveness() {
        let bytecode = analyze_asm(
            "
            PUSH1 0x01
            PUSH1 0x02
            PUSH1 0x03
            SWAP2
            STOP
        ",
        );
        for i in 0..3 {
            assert!(
                !bytecode.inst(Inst::from_usize(i)).flags.contains(InstFlags::SKIP_LOGIC),
                "PUSH at {i} should NOT be skipped"
            );
        }
    }

    #[test]
    fn dup_keeps_source_live() {
        let bytecode = analyze_asm(
            "
            PUSH1 0x42
            DUP1
            POP
            PUSH0
            MSTORE
            STOP
        ",
        );
        assert!(
            !bytecode.inst(Inst::from_usize(0)).flags.contains(InstFlags::SKIP_LOGIC),
            "PUSH should NOT be skipped"
        );
    }

    #[test]
    fn no_side_effects() {
        let bytecode = analyze_asm(
            "
            PUSH1 0x42
            PUSH0
            MSTORE
            STOP
        ",
        );
        assert!(!bytecode.inst(Inst::from_usize(2)).flags.contains(InstFlags::SKIP_LOGIC));
    }
}

//! Intra-block dead store elimination.
//!
//! Walks each basic block backward to determine which stack positions are live (will be read
//! by a later instruction or at the block exit). Instructions whose outputs are all dead and
//! whose logic is safe to skip are marked [`InstFlags::SKIP_LOGIC`].
//!
//! DUP, SWAP, and POP require custom liveness transfer functions because generic `(inputs,
//! outputs)` arity doesn't capture how they map stack slots:
//! - SWAP permutes two positions without consuming or producing values.
//! - DUP copies a single deep source to a new TOS.
//! - POP kills its input (makes the position dead) rather than reading it.

use super::StackSection;
use crate::bytecode::{Bytecode, InstFlags};
use bitvec::vec::BitVec;
use revm_bytecode::opcode as op;

impl Bytecode<'_> {
    /// Intra-block dead store elimination.
    ///
    /// Walks each block backward to determine which stack positions are live (will be read
    /// by a later instruction or at the block exit). Instructions that are safe to skip and
    /// whose outputs are all dead are marked `SKIP_LOGIC`.
    #[instrument(name = "dse", level = "debug", skip_all)]
    pub(crate) fn dead_store_elim(&mut self) {
        let mut eliminated = 0u32;

        // Reusable buffers hoisted out of the per-block loop.
        let mut heights: Vec<i32> = Vec::new();
        let mut live = BitVec::new();

        for bid in self.cfg.blocks.indices() {
            let block = &self.cfg.blocks[bid];
            let insts = block.insts();

            // Compute the stack height at each instruction boundary by walking forward.
            // heights[i] = stack height *before* executing insts[i].
            // heights[insts.len()] = stack height after the last instruction.
            heights.clear();
            let entry_height = {
                let section =
                    StackSection::from_stack_io(insts.clone().map(|i| self.insts[i].stack_io()));
                section.inputs as i32
            };
            let mut max_height = entry_height;
            heights.push(entry_height);
            for inst in insts.clone() {
                let data = &self.insts[inst];
                let h = if data.is_dead_code() || data.flags.contains(InstFlags::SKIP_LOGIC) {
                    *heights.last().unwrap()
                } else {
                    let (inp, out) = data.stack_io();
                    heights.last().unwrap() - inp as i32 + out as i32
                };
                if h > max_height {
                    max_height = h;
                }
                heights.push(h);
            }

            let exit_height = *heights.last().unwrap();
            let live_len = max_height.max(0) as usize;

            // At the block exit, all stack positions are conservatively live.
            live.clear();
            live.resize(live_len, false);
            for i in 0..exit_height.max(0) as usize {
                live.set(i, true);
            }

            // Walk backward.
            for (idx, inst) in insts.clone().enumerate().rev() {
                let data = &self.insts[inst];
                if data.is_dead_code() || data.flags.contains(InstFlags::SKIP_LOGIC) {
                    continue;
                }

                let opcode = data.opcode;
                let (inp, out) = data.stack_io();
                let h_before = heights[idx] as usize;

                // Check if all outputs are dead and the instruction can be skipped.
                if out > 0
                    && can_skip_when_dead(opcode)
                    && !data.is_branching()
                    && !data.is_diverging()
                {
                    let write_base = h_before - inp as usize;
                    let all_dead = (0..out as usize)
                        .all(|k| {
                            let pos = write_base + k;
                            pos < live.len() && !live[pos]
                        });

                    if all_dead {
                        self.insts[inst].flags |= InstFlags::SKIP_LOGIC;
                        eliminated += 1;
                        continue;
                    }
                }

                transfer_liveness(&mut live, opcode, h_before, inp as usize, out as usize);
            }
        }

        if eliminated > 0 {
            debug!(eliminated, "dead stores eliminated");
        }
    }
}

/// Returns `true` if the opcode's logic is safe to skip when its outputs are dead.
///
/// This covers opcodes with no side effects beyond stack I/O and no dynamic gas. When
/// SKIP_LOGIC is set the instruction is still present (static gas is charged, stack
/// length is adjusted), only the computation is skipped.
///
/// Based on solc's `SemanticInformation::movable()` with additions for call-constant
/// environment reads that are pure in EVM semantics. Excludes opcodes with dynamic gas
/// (EXP), memory access (MLOAD, KECCAK256), storage/host reads (SLOAD, TLOAD, BALANCE,
/// etc.), and MSIZE/GAS (position-dependent).
fn can_skip_when_dead(opcode: u8) -> bool {
    matches!(
        opcode,
        // Arithmetic.
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
            // Comparison.
            | op::LT
            | op::GT
            | op::SLT
            | op::SGT
            | op::EQ
            | op::ISZERO
            // Bitwise.
            | op::AND
            | op::OR
            | op::XOR
            | op::NOT
            | op::BYTE
            | op::SHL
            | op::SHR
            | op::SAR
            | op::CLZ
            // Constants.
            | op::PUSH0..=op::PUSH32
            | op::PC
            | op::CODESIZE
            // Call-constant environment reads (no dynamic gas, no host calls).
            | op::ADDRESS
            | op::ORIGIN
            | op::CALLER
            | op::CALLVALUE
            | op::CALLDATALOAD
            | op::CALLDATASIZE
            | op::GASPRICE
            | op::COINBASE
            | op::TIMESTAMP
            | op::NUMBER
            | op::DIFFICULTY
            | op::GASLIMIT
            | op::CHAINID
            | op::BASEFEE
            | op::BLOBBASEFEE
    )
}

/// Backward liveness transfer for a single instruction.
///
/// Updates `live` to reflect liveness *before* the instruction executes, given
/// `h_before` (the stack height before the instruction).
fn transfer_liveness(live: &mut BitVec, opcode: u8, h_before: usize, inp: usize, out: usize) {
    match opcode {
        op::SWAP1..=op::SWAP16 => {
            // SWAP exchanges TOS and TOS-depth. Propagate liveness from post-swap
            // destination to pre-swap source.
            let depth = (opcode - op::SWAP1 + 1) as usize;
            let tos = h_before - 1;
            let other = tos - depth;
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
            // DUP pushes a copy of stack[depth] as new TOS. The source is live iff
            // it was already live as a pass-through OR the duplicated TOS is live.
            let depth = (opcode - op::DUP1 + 1) as usize;
            let new_tos = h_before;
            let src = h_before - depth;
            let new_tos_live = new_tos < live.len() && live[new_tos];
            let src_live = src < live.len() && live[src];
            if new_tos < live.len() {
                live.set(new_tos, false);
            }
            if src < live.len() {
                live.set(src, src_live || new_tos_live);
            }
        }
        op::POP => {
            // POP discards its input — kill the position to allow the producer to
            // be eliminated.
            let pos = h_before - 1;
            if pos < live.len() {
                live.set(pos, false);
            }
        }
        _ => {
            let write_base = h_before - inp;
            // Kill outputs.
            for k in 0..out {
                let pos = write_base + k;
                if pos < live.len() {
                    live.set(pos, false);
                }
            }
            // Gen inputs.
            for k in 0..inp {
                let pos = h_before - inp + k;
                if pos < live.len() {
                    live.set(pos, true);
                }
            }
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
    fn dup_both_dead() {
        // DUP copies source to new TOS; if both the copy and the source are popped,
        // the producer should be eliminable.
        let bytecode = analyze_asm(
            "
            PUSH1 0x42
            DUP1
            POP
            POP
            STOP
        ",
        );
        assert!(
            bytecode.inst(Inst::from_usize(0)).flags.contains(InstFlags::SKIP_LOGIC),
            "PUSH should be skipped when both DUP copy and source are dead"
        );
    }

    #[test]
    fn env_read_dead() {
        // Call-constant environment reads should be eliminable when dead.
        let bytecode = analyze_asm(
            "
            ADDRESS
            POP
            STOP
        ",
        );
        assert!(
            bytecode.inst(Inst::from_usize(0)).flags.contains(InstFlags::SKIP_LOGIC),
            "ADDRESS should be skipped when dead"
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

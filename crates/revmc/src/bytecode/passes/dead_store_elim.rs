//! Intra-block dead store elimination.
//!
//! Walks each basic block backward to determine which stack positions are live (will be read
//! by a later instruction or at the block exit). Instructions whose outputs are all dead and
//! whose logic is safe to skip are marked [`InstFlags::SKIP_LOGIC`].
//!
//! DUP, SWAP, POP, DUPN, SWAPN, and EXCHANGE require custom liveness transfer functions
//! because generic `(inputs, outputs)` arity doesn't capture how they map stack slots:
//! - SWAP/SWAPN permute two positions without consuming or producing values.
//! - DUP/DUPN copy a single deep source to a new TOS.
//! - EXCHANGE swaps two arbitrary non-TOS positions.
//! - POP kills its input (makes the position dead) rather than reading it.

use super::StackSection;
use crate::bytecode::{Bytecode, InstFlags};
use bitvec::vec::BitVec;
use revm_bytecode::opcode as op;

/// Decoded immediate for liveness transfer of DUPN/SWAPN/EXCHANGE.
enum DecodedImm {
    /// Single depth (DUPN, SWAPN).
    Single(usize),
    /// Pair of non-TOS indices (EXCHANGE).
    Pair(usize, usize),
}

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
            // Compute the stack height at each instruction boundary by walking forward.
            // heights[i] = stack height *before* executing insts[i].
            // heights[insts.len()] = stack height after the last instruction.
            heights.clear();
            let entry_height = {
                let section =
                    StackSection::from_stack_io(block.insts().map(|i| self.insts[i].stack_io()));
                section.inputs as i32
            };
            let mut max_height = entry_height;
            heights.push(entry_height);
            for inst in block.insts() {
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
            for (idx, inst) in block.insts().enumerate().rev() {
                let data = &self.insts[inst];
                if data.is_dead_code()
                    || data.flags.contains(InstFlags::SKIP_LOGIC)
                    || data.flags.contains(InstFlags::DISABLED)
                    || data.flags.contains(InstFlags::UNKNOWN)
                {
                    continue;
                }

                let opcode = data.opcode;
                let (inp, out) = data.stack_io();
                let h_before = heights[idx] as usize;

                // Decode immediate for DUPN/SWAPN/EXCHANGE.
                let imm = match opcode {
                    op::DUPN | op::SWAPN => self
                        .get_imm(data)
                        .and_then(|b| crate::decode_single(b[0]))
                        .map(|n| DecodedImm::Single(n as usize)),
                    op::EXCHANGE => self
                        .get_imm(data)
                        .and_then(|b| crate::decode_pair(b[0]))
                        .map(|(n, m)| DecodedImm::Pair(n as usize, m as usize)),
                    _ => None,
                };

                // Check if all outputs are dead and the instruction can be skipped.
                if out > 0
                    && can_skip_when_dead(opcode)
                    && !data.is_branching()
                    && !data.is_diverging()
                {
                    let write_base = h_before - inp as usize;
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

                transfer_liveness(&mut live, opcode, h_before, inp as usize, out as usize, &imm);
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
            | op::PUSH0
            ..=op::PUSH32
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
            | op::SLOTNUM
    )
}

/// Backward liveness transfer for a single instruction.
///
/// Updates `live` to reflect liveness *before* the instruction executes, given
/// `h_before` (the stack height before the instruction). All positions are guaranteed
/// in-bounds since `live` is sized to the block's max stack height.
///
/// `imm` carries the decoded immediate for DUPN/SWAPN/EXCHANGE; `None` for all other opcodes.
fn transfer_liveness(
    live: &mut BitVec,
    opcode: u8,
    h_before: usize,
    inp: usize,
    out: usize,
    imm: &Option<DecodedImm>,
) {
    match opcode {
        op::SWAP1..=op::SWAP16 => {
            let depth = (opcode - op::SWAP1 + 1) as usize;
            transfer_swap(live, h_before, depth);
        }
        op::SWAPN => match imm {
            Some(DecodedImm::Single(depth)) => transfer_swap(live, h_before, *depth),
            _ => generic_transfer(live, h_before, inp, out),
        },
        op::DUP1..=op::DUP16 => {
            let depth = (opcode - op::DUP1 + 1) as usize;
            transfer_dup(live, h_before, depth);
        }
        op::DUPN => match imm {
            Some(DecodedImm::Single(depth)) => transfer_dup(live, h_before, *depth),
            _ => generic_transfer(live, h_before, inp, out),
        },
        op::EXCHANGE => match imm {
            Some(DecodedImm::Pair(n, m)) => {
                let tos = h_before - 1;
                let pos_a = tos - n;
                let pos_b = tos - m;
                let (a, b) = (live[pos_a], live[pos_b]);
                live.set(pos_a, b);
                live.set(pos_b, a);
            }
            _ => generic_transfer(live, h_before, inp, out),
        },
        op::POP => {
            live.set(h_before - 1, false);
        }
        _ => generic_transfer(live, h_before, inp, out),
    }
}

/// Generic liveness transfer: kill all outputs, then mark all inputs as live.
fn generic_transfer(live: &mut BitVec, h_before: usize, inp: usize, out: usize) {
    let write_base = h_before - inp;
    for k in 0..out {
        live.set(write_base + k, false);
    }
    for k in 0..inp {
        live.set(h_before - inp + k, true);
    }
}

/// SWAP liveness: permute liveness of TOS and the swapped position.
fn transfer_swap(live: &mut BitVec, h_before: usize, depth: usize) {
    let tos = h_before - 1;
    let other = tos - depth;
    let (a, b) = (live[tos], live[other]);
    live.set(tos, b);
    live.set(other, a);
}

/// DUP liveness: the new TOS is killed; if it was live, the source becomes live.
fn transfer_dup(live: &mut BitVec, h_before: usize, depth: usize) {
    let new_tos = h_before;
    let src = h_before - depth;
    let new_tos_live = live[new_tos];
    live.set(new_tos, false);
    if new_tos_live {
        live.set(src, true);
    }
}

#[cfg(test)]
mod tests {
    use super::super::block_analysis::tests::{Inst, analyze_asm, analyze_code_spec};
    use crate::bytecode::InstFlags;
    use revm_bytecode::opcode as op;
    use revm_primitives::hardfork::SpecId;

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

    /// Helper: builds bytecode with `n` PUSH0s followed by `suffix`, using AMSTERDAM spec.
    fn eof_with_prefix(n: usize, suffix: &[u8]) -> crate::bytecode::Bytecode<'static> {
        let mut code = vec![op::PUSH0; n];
        code.extend_from_slice(suffix);
        analyze_code_spec(code, SpecId::AMSTERDAM)
    }

    #[test]
    fn swapn_preserves_liveness() {
        // 18 × PUSH0 (insts 0..18), SWAPN 17 (inst 18), STOP (inst 19).
        // SWAPN 17 needs TOS + 17 below = 18 items.
        let bytecode = eof_with_prefix(18, &[op::SWAPN, 0x00, op::STOP]);
        for i in 0..18 {
            assert!(
                !bytecode.inst(Inst::from_usize(i)).flags.contains(InstFlags::SKIP_LOGIC),
                "PUSH0 at {i} should NOT be skipped"
            );
        }
    }

    #[test]
    fn dupn_keeps_source_live() {
        // 17 × PUSH0, DUPN 17 (dup bottom), POP, STOP.
        // The DUP copy is popped but the source (inst 0) is still live at exit.
        let bytecode = eof_with_prefix(17, &[op::DUPN, 0x00, op::POP, op::STOP]);
        assert!(
            !bytecode.inst(Inst::from_usize(0)).flags.contains(InstFlags::SKIP_LOGIC),
            "source PUSH0 should NOT be skipped"
        );
    }

    #[test]
    fn dupn_dead_copy() {
        // 17 × PUSH0, DUPN 17 (dup bottom), POP, STOP.
        // The DUP copy is immediately popped. With precise DUPN liveness the source stays
        // live (it's still on the exit stack), and the DUPN itself can't be eliminated
        // because it's not in `can_skip_when_dead`. What we really test is that the
        // transfer doesn't incorrectly kill the source position.
        let bytecode = eof_with_prefix(17, &[op::DUPN, 0x00, op::POP, op::STOP]);
        // All 17 original PUSH0s should remain live (they're on the exit stack).
        for i in 0..17 {
            assert!(
                !bytecode.inst(Inst::from_usize(i)).flags.contains(InstFlags::SKIP_LOGIC),
                "PUSH0 at {i} should NOT be skipped"
            );
        }
    }

    #[test]
    fn exchange_preserves_liveness() {
        // 18 × PUSH0, EXCHANGE 1,2 (swap positions 1 and 2 from TOS), STOP.
        // EXCHANGE needs at least m+1 items. With (1,2): 3 items minimum, but we use 18
        // to match the DUPN/SWAPN test shape and ensure all are live.
        let bytecode = eof_with_prefix(18, &[op::EXCHANGE, 0x01, op::STOP]);
        for i in 0..18 {
            assert!(
                !bytecode.inst(Inst::from_usize(i)).flags.contains(InstFlags::SKIP_LOGIC),
                "PUSH0 at {i} should NOT be skipped"
            );
        }
    }
}

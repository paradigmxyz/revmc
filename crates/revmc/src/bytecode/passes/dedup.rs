//! Block deduplication pass.
//!
//! Identifies structurally identical non-fallthrough blocks (same opcode + immediate sequence)
//! and eliminates duplicates by marking them as dead code and redirecting predecessors to a
//! single canonical copy.

use super::block_analysis::{Block, BlockData, Snapshots};
use crate::bytecode::{Bytecode, Inst, InstData, InstFlags};
use alloy_primitives::map::HashMap;
use oxc_index::IndexVec;
use revm_bytecode::opcode as op;
use smallvec::SmallVec;

impl<'a> Bytecode<'a> {
    /// Deduplicate structurally identical non-fallthrough blocks.
    ///
    /// Eligible blocks are those whose terminator cannot fall through (diverging instructions
    /// like REVERT/STOP/RETURN, or unconditional JUMP). For every group of byte-identical
    /// blocks, keeps one canonical copy while marking the rest as dead code.
    /// Predecessors that reach a dead duplicate via a static jump or multi-jump are
    /// redirected to the canonical block. For predecessors that reach the duplicate via
    /// fallthrough, a redirect entry is stored in [`Bytecode::redirects`] so the
    /// translator can map the dead instruction to the canonical block's IR block.
    ///
    /// `local_snapshots` are the block-local snapshots computed by `block_analysis_local`
    /// (before the global fixpoint). After merging, the canonical block's snapshots are
    /// restored to these local values because the global snapshots may be context-specific
    /// and stale once multiple predecessors are merged.
    #[instrument(name = "dedup", level = "debug", skip_all)]
    pub(crate) fn dedup_blocks(&mut self, local_snapshots: &Snapshots) {
        // Group eligible (diverging, non-dead) blocks by their raw bytecode content.
        // We borrow `self.code` separately to avoid holding a `&self` borrow across mutations.
        let code = &*self.code;
        let mut key_to_blocks = HashMap::<&[u8], SmallVec<[Block; 4]>>::default();
        for bid in self.cfg.blocks.indices() {
            let block = &self.cfg.blocks[bid];
            // Only dedup blocks whose terminator cannot fall through (diverging or
            // unconditional JUMP). JUMPI blocks fall through to position-dependent targets.
            let term = &self.insts[block.terminator()];
            if term.can_fall_through() {
                continue;
            }

            // Reachable JUMPDESTs can only be deduped when:
            // - all jumps are statically resolved (no dynamic jumps), AND
            // - the block terminates execution (STOP/RETURN/REVERT/etc.), not JUMP.
            // Blocks ending in JUMP cannot be deduped by raw bytes alone because
            // block_analysis may have resolved different target sets for byte-identical
            // copies (e.g. different return-address contexts).
            let first = &self.insts[block.insts.start];
            if first.is_reachable_jumpdest(self.has_dynamic_jumps)
                && (self.has_dynamic_jumps || term.opcode == op::JUMP)
            {
                continue;
            }

            // Skip blocks containing PC — it returns the current program counter,
            // so identical bytecode at different positions produces different results.
            if block.insts().any(|i| self.insts[i].opcode == op::PC) {
                continue;
            }

            let bytes = block_bytes(code, &self.insts, block);
            if bytes.is_empty() {
                continue;
            }
            key_to_blocks.entry(bytes).or_default().push(bid);
        }

        let mut deduped = 0usize;
        for group in key_to_blocks.values() {
            if group.len() < 2 {
                continue;
            }
            let (&canonical, dups) = group.split_first().unwrap();
            let canonical_first_inst = self.cfg.blocks[canonical].insts.start;

            // Restore block-local snapshots for the canonical block. The global
            // fixpoint may have recorded context-specific constants that become stale
            // once the canonical block serves multiple predecessors after merging.
            // Block-local constants (computed without incoming stack state) remain valid.
            self.snapshots.restore_from(self.cfg.blocks[canonical].insts(), local_snapshots);

            for &dup in dups {
                deduped += 1;
                trace!("deduped: {from} -> {to}", from = dup, to = canonical);

                // Mark all instructions in the duplicate block as dead.
                let dup_block = &self.cfg.blocks[dup];
                for i in dup_block.insts() {
                    self.insts[i].flags |= InstFlags::DEAD_CODE;
                }

                let dup_first_inst = dup_block.insts.start;
                self.redirects.insert(dup_first_inst, canonical_first_inst);

                // If the duplicate was a reachable JUMPDEST, the canonical must be too
                // so that sections/gas-checks treat it as a jump target entry point.
                if self.insts[dup_first_inst].data == 1 {
                    self.insts[canonical_first_inst].data = 1;
                }

                // Redirect predecessors that reach the duplicate via static jumps.
                for &pred in &dup_block.preds {
                    let term_inst = self.cfg.blocks[pred].terminator();
                    let term = &self.insts[term_inst];
                    let is_static = term.is_static_jump()
                        && !term.flags.contains(InstFlags::INVALID_JUMP)
                        && !term.flags.contains(InstFlags::MULTI_JUMP)
                        && term.data == dup_first_inst.index() as u32;
                    if is_static {
                        self.insts[term_inst].data = canonical_first_inst.index() as u32;
                    }
                    // Multi-jump targets are NOT rewritten: each target carries a
                    // distinct PC that callers may push as a return address. The
                    // `inst_entries` redirect (translate.rs) already maps the dead
                    // instruction's IR entry to the canonical one, so the switch
                    // correctly emits cases for both PCs pointing to the same IR block.
                }
            }
        }

        debug!(deduped, "finished");
    }
}

/// Returns the raw bytecode bytes for a block's PC range, or empty if the block
/// extends past the original code (e.g. the synthetic STOP padding).
fn block_bytes<'a>(
    code: &'a [u8],
    insts: &IndexVec<Inst, InstData>,
    block: &BlockData,
) -> &'a [u8] {
    let start_pc = insts[block.insts.start].pc as usize;
    let end_inst = &insts[block.terminator()];
    let end_pc = end_inst.pc as usize + 1 + end_inst.imm_len() as usize;
    code.get(start_pc..end_pc).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use crate::bytecode::{AnalysisConfig, passes::block_analysis::tests::analyze_asm_with};

    #[test]
    fn dedup_identical_revert_blocks() {
        // Two JUMPI branches each fall through to an identical `PUSH1 0x00 / DUP1 / REVERT`.
        let bytecode = analyze_asm_with(
            "
            PUSH0
            CALLDATALOAD
            PUSH %bb2
            JUMPI
            ; revert A
            PUSH1 0x00
            DUP1
            REVERT
        bb2:
            JUMPDEST
            PUSH0
            CALLDATALOAD
            PUSH %bb4
            JUMPI
            ; revert B (identical to A)
            PUSH1 0x00
            DUP1
            REVERT
        bb4:
            JUMPDEST
            STOP
        ",
            AnalysisConfig::DEDUP,
        );

        // One of the two `PUSH1 0x00 / DUP1 / REVERT` blocks should be redirected.
        assert_eq!(
            bytecode.redirects.len(),
            1,
            "expected exactly 1 redirect for 2 identical revert blocks",
        );
    }

    #[test]
    fn dedup_preserves_unique_blocks() {
        // Two different diverging blocks — they should NOT be deduplicated.
        let bytecode = analyze_asm_with(
            "
            PUSH0
            CALLDATALOAD
            PUSH %target
            JUMPI
            ; revert with 0
            PUSH1 0x00
            DUP1
            REVERT
        target:
            JUMPDEST
            STOP
        ",
            AnalysisConfig::DEDUP,
        );

        assert!(bytecode.redirects.is_empty(), "should not dedup different blocks");
    }

    #[test]
    fn dedup_three_identical_blocks() {
        // Three identical `PUSH1 0x00 / DUP1 / REVERT` blocks.
        let bytecode = analyze_asm_with(
            "
            PUSH0
            CALLDATALOAD
            PUSH %bb2
            JUMPI
            ; revert A
            PUSH1 0x00
            DUP1
            REVERT
        bb2:
            JUMPDEST
            PUSH0
            CALLDATALOAD
            PUSH %bb4
            JUMPI
            ; revert B
            PUSH1 0x00
            DUP1
            REVERT
        bb4:
            JUMPDEST
            PUSH0
            CALLDATALOAD
            PUSH %bb6
            JUMPI
            ; revert C
            PUSH1 0x00
            DUP1
            REVERT
        bb6:
            JUMPDEST
            STOP
        ",
            AnalysisConfig::DEDUP,
        );

        let redirect_count = bytecode.redirects.len();
        assert_eq!(redirect_count, 2, "expected 2 redirects, got {redirect_count}");
    }

    #[test]
    fn dedup_skips_pc_opcode() {
        // Two identical `PC / PUSH1 0x00 / REVERT` blocks — PC is position-dependent,
        // so they must NOT be deduplicated.
        let bytecode = analyze_asm_with(
            "
            PUSH0
            CALLDATALOAD
            PUSH %bb2
            JUMPI
            ; PC + revert A
            PC
            PUSH1 0x00
            REVERT
        bb2:
            JUMPDEST
            PUSH0
            CALLDATALOAD
            PUSH %bb4
            JUMPI
            ; PC + revert B (same bytes, different PC value)
            PC
            PUSH1 0x00
            REVERT
        bb4:
            JUMPDEST
            STOP
        ",
            AnalysisConfig::DEDUP,
        );

        assert!(bytecode.redirects.is_empty(), "should not dedup blocks containing PC");
    }

    #[test]
    fn dedup_weth() {
        let code =
            revm_primitives::hex::decode(include_str!("../../../../../data/weth.rt.hex").trim())
                .unwrap();
        let mut bytecode = crate::Bytecode::test(code);
        bytecode.config = AnalysisConfig::DEDUP;
        bytecode.analyze().unwrap();

        assert_eq!(bytecode.redirects.len(), 13);
    }

    #[test]
    fn dedup_invalidates_stale_const_snapshots() {
        // Two byte-identical RETURN tails reached with different incoming constants.
        // After dedup the canonical block must NOT retain stale const snapshots.
        let bytecode = analyze_asm_with(
            "
            CALLDATASIZE
            PUSH %path1
            JUMPI

            PUSH1 0xAA
            PUSH %tail0
            JUMP

        path1:
            JUMPDEST
            PUSH1 0xBB
            PUSH %tail1
            JUMP

        tail0:
            JUMPDEST
            PUSH1 0x01
            ADD
            PUSH0
            MSTORE
            PUSH1 0x20
            PUSH0
            RETURN

        tail1:
            JUMPDEST
            PUSH1 0x01
            ADD
            PUSH0
            MSTORE
            PUSH1 0x20
            PUSH0
            RETURN
        ",
            AnalysisConfig::DEDUP,
        );

        // Dedup should have merged the two tails.
        assert_eq!(bytecode.redirects.len(), 1);

        // The canonical block's ADD must not carry a stale const_output
        // (it would be 0xAB from the first context, but the second expects 0xBC).
        let canonical_start = *bytecode.redirects.values().next().unwrap();
        let add_inst = canonical_start + 2; // JUMPDEST, PUSH1, ADD
        assert!(
            bytecode.const_output(add_inst).is_none(),
            "canonical ADD should have no const_output after dedup (was stale)",
        );
    }
}

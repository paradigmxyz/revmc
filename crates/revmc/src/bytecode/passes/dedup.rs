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

/// Dedup key: raw bytecode bytes plus CFG successor blocks.
///
/// Raw bytes alone are insufficient for JUMP-terminated blocks because block analysis
/// may resolve byte-identical copies to different static targets depending on incoming
/// stack context (e.g. different return-address values).
#[derive(Clone, PartialEq, Eq, Hash)]
struct DedupKey<'a> {
    bytes: &'a [u8],
    /// CFG successors for this block. Two byte-identical blocks are only merged when
    /// their successor sets also match.
    succs: SmallVec<[Block; 4]>,
}

impl<'a> Bytecode<'a> {
    /// Deduplicate structurally identical non-fallthrough blocks.
    ///
    /// Eligible blocks are those whose terminator cannot fall through (diverging instructions
    /// like REVERT/STOP/RETURN, or unconditional JUMP). For every group of byte-identical
    /// blocks, keeps one canonical copy while marking them as dead code.
    /// A redirect entry is stored in [`Bytecode::redirects`] so the translator can map
    /// dead instructions to the canonical block's IR block. The CFG is rebuilt after each
    /// iteration so successor/predecessor lists stay consistent.
    ///
    /// `local_snapshots` are the block-local snapshots computed by `block_analysis_local`
    /// (before the global fixpoint). After merging, the canonical block's snapshots are
    /// restored to these local values because the global snapshots may be context-specific
    /// and stale once multiple predecessors are merged.
    #[instrument(name = "dedup", level = "debug", skip_all)]
    pub(crate) fn dedup_blocks(&mut self, local_snapshots: &Snapshots) {
        // Borrow code separately so we can pass `&mut self` to `dedup_blocks_once`.
        // SAFETY: `self.code` is not modified during dedup.
        let code: &[u8] = unsafe { &*std::ptr::from_ref::<[u8]>(&self.code) };
        let mut key_to_blocks = HashMap::<DedupKey<'_>, SmallVec<[Block; 4]>>::default();
        let mut total_deduped = 0usize;
        loop {
            key_to_blocks.clear();
            let deduped = self.dedup_blocks_once(code, &mut key_to_blocks, local_snapshots);
            total_deduped += deduped;
            if deduped == 0 {
                break;
            }
            self.rebuild_cfg();
        }
        debug!(deduped = total_deduped, "finished");
    }

    /// Single dedup iteration. Returns the number of blocks deduped.
    fn dedup_blocks_once<'b>(
        &mut self,
        code: &'b [u8],
        key_to_blocks: &mut HashMap<DedupKey<'b>, SmallVec<[Block; 4]>>,
        local_snapshots: &Snapshots,
    ) -> usize {
        for bid in self.cfg.blocks.indices() {
            let block = &self.cfg.blocks[bid];
            let term = &self.insts[block.terminator()];

            // Only dedup blocks whose terminator cannot fall through (diverging or
            // unconditional JUMP). JUMPI blocks fall through to position-dependent targets.
            if term.can_fall_through() {
                continue;
            }

            // Reachable JUMPDESTs with dynamic jumps cannot be deduped because any
            // JUMPDEST could be reached from an unresolved dynamic JUMP with an
            // arbitrary stack context.
            let first = &self.insts[block.insts.start];
            if self.has_dynamic_jumps && first.is_reachable_jumpdest(true) {
                continue;
            }

            // Skip blocks containing PC — it returns the current program counter,
            // so identical bytecode at different positions produces different results.
            if block.insts().any(|i| self.insts[i].opcode == op::PC) {
                continue;
            }

            // Skip unresolved dynamic JUMPs — any target is possible at runtime.
            if term.opcode == op::JUMP && !term.flags.contains(InstFlags::STATIC_JUMP) {
                continue;
            }

            let bytes = block_bytes(code, &self.insts, block);
            if bytes.is_empty() {
                continue;
            }

            key_to_blocks
                .entry(DedupKey { bytes, succs: block.succs.clone() })
                .or_default()
                .push(bid);
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

                // Merge multi-jump targets from the duplicate into the canonical block.
                // Without this, valid case PCs unique to the duplicate dispatcher would
                // be lost and fall into the default InvalidJump path at runtime.
                let canonical_term = self.cfg.blocks[canonical].terminator();
                let dup_term = dup_block.terminator();
                if self.insts[dup_term].flags.contains(InstFlags::MULTI_JUMP) {
                    if let Some(dup_targets) = self.multi_jump_targets.remove(&dup_term) {
                        let canonical_targets =
                            self.multi_jump_targets.get_mut(&canonical_term).unwrap();
                        for t in dup_targets {
                            if !canonical_targets.contains(&t) {
                                canonical_targets.push(t);
                            }
                        }
                    }
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
                    // Multi-jump target PCs are not rewritten here: each carries a
                    // distinct PC that callers push as a return address. The targets
                    // were already merged into the canonical block above, and
                    // `inst_entries` redirects (translate.rs) map the dead
                    // instruction's IR entry to the canonical one.
                }
            }
        }

        deduped
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
    fn dedup_jump_same_target() {
        // Two byte-identical fallthrough JUMP tails resolved to the same target.
        let bytecode = analyze_asm_with(
            "
            CALLDATASIZE
            PUSH %path1
            JUMPI
            PUSH0
            PUSH1 0xFF
            JUMPI
            PUSH %target
            JUMP
        path1:
            JUMPDEST
            PUSH0
            PUSH1 0xFF
            JUMPI
            PUSH %target
            JUMP
        target:
            JUMPDEST
            STOP
        ",
            AnalysisConfig::DEDUP,
        );
        assert_eq!(bytecode.redirects.len(), 1, "same-target JUMP tails should be deduped");
    }

    #[test]
    fn dedup_jump_different_targets() {
        // Two byte-identical non-JUMPDEST JUMP tails with different resolved static targets.
        // Must NOT be merged — the JUMP target is context-sensitive.
        // Regression test for:
        // revmc-dedup-non-jumpdest-fallthrough-jump-tail-static-target-confusion
        let bytecode = analyze_asm_with(
            "
            CALLDATASIZE
            PUSH %path1
            JUMPI

            PUSH %ret0
            PUSH0
            PUSH1 0xFF
            JUMPI
            PUSH1 0x42
            SWAP1
            JUMP

        path1:
            JUMPDEST
            PUSH %ret1
            PUSH0
            PUSH1 0xFF
            JUMPI
            PUSH1 0x42
            SWAP1
            JUMP

        ret0:
            JUMPDEST
            POP
            PUSH1 0xAA
            STOP

        ret1:
            JUMPDEST
            POP
            PUSH1 0xBB
            STOP
        ",
            AnalysisConfig::DEDUP,
        );

        assert!(
            bytecode.redirects.is_empty(),
            "different-target JUMP tails must not be deduped, got {} redirect(s)",
            bytecode.redirects.len(),
        );
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
    fn dedup_skips_dynamic_jump_sstore_tail() {
        // Two non-JUMPDEST byte-identical `SSTORE ; JUMP` tails with unresolved dynamic
        // JUMP terminators must NOT be deduped.
        let bytecode = analyze_asm_with(
            "
            PUSH0
            CALLDATALOAD
            PUSH1 0x20
            CALLDATALOAD
            PUSH %path_b
            JUMPI

            ; path A
            PUSH1 0x11
            PUSH1 0x01
            PUSH0
            PUSH %not_taken_a
            JUMPI
            SSTORE
            JUMP

        not_taken_a:
            JUMPDEST
            INVALID

        path_b:
            JUMPDEST
            PUSH1 0x22
            PUSH1 0x02
            PUSH0
            PUSH %not_taken_b
            JUMPI
            SSTORE
            JUMP

        not_taken_b:
            JUMPDEST
            INVALID

        done:
            JUMPDEST
            STOP
        ",
            AnalysisConfig::DEDUP,
        );

        assert!(
            bytecode.redirects.is_empty(),
            "should not dedup blocks ending in unresolved dynamic JUMP",
        );
    }

    #[test]
    fn dedup_multi_jump_source_merges_targets() {
        // Two byte-identical MULTI_JUMP dispatcher blocks (dispatcher_a, dispatcher_b)
        // each route to two identical JUMPDEST+STOP return blocks. After the return
        // blocks are deduped, the two dispatchers become identical in (bytes, succs).
        // Dedup must merge their multi_jump_targets so the canonical switch emits cases
        // for ALL valid return PCs.
        let bytecode = analyze_asm_with(
            "
            ; Branch on first calldata word.
            PUSH0
            CALLDATALOAD
            PUSH %path_b
            JUMPI

            ; Branch on second calldata word.
            PUSH1 0x20
            CALLDATALOAD
            PUSH %path_a1
            JUMPI
            PUSH %ret0
            PUSH %dispatcher_a
            JUMP
        path_a1:
            JUMPDEST
            PUSH %ret1
            PUSH %dispatcher_a
            JUMP

        path_b:
            JUMPDEST
            PUSH1 0x20
            CALLDATALOAD
            PUSH %path_b1
            JUMPI
            PUSH %ret2
            PUSH %dispatcher_b
            JUMP
        path_b1:
            JUMPDEST
            PUSH %ret3
            PUSH %dispatcher_b
            JUMP

        ret0:
            JUMPDEST
            STOP
        ret1:
            JUMPDEST
            STOP
        ret2:
            JUMPDEST
            STOP
        ret3:
            JUMPDEST
            STOP

        dispatcher_a:
            JUMPDEST
            JUMP
        dispatcher_b:
            JUMPDEST
            JUMP
        ",
            AnalysisConfig::DEDUP,
        );

        // The four STOP blocks are deduped (3 redirects), and one dispatcher is deduped
        // (1 redirect) = 4 total.
        assert_eq!(bytecode.redirects.len(), 4);

        // Find the canonical dispatcher's multi_jump_targets — it must contain all 4
        // original return PCs (possibly redirected to the canonical STOP inst).
        let canonical_dispatcher_term = bytecode
            .multi_jump_targets
            .keys()
            .find(|&&inst| !bytecode.insts[inst].is_dead_code())
            .expect("should have a live MULTI_JUMP dispatcher");
        let targets = &bytecode.multi_jump_targets[canonical_dispatcher_term];
        assert_eq!(
            targets.len(),
            4,
            "canonical dispatcher should have 4 multi-jump targets (merged), got {targets:?}",
        );
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

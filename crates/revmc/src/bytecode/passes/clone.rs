//! Continuation block cloning pass.
//!
//! Inspired by Gigahorse's `HeuristicBlockCloner`, this pass identifies continuation blocks
//! (return targets) that are shared by multiple private function call sites and clones them
//! so each call site gets its own copy. This prevents the flat fixpoint from merging different
//! return addresses at a shared JUMPDEST, enabling more jump resolutions.
//!
//! The pass runs before `block_analysis` so the abstract interpreter sees separate targets.

use super::block_analysis::{Block, BlockLocalSummary};
use crate::bytecode::{Bytecode, Inst, U256Idx};
use oxc_index::IndexVec;
use revm_bytecode::opcode as op;
use revm_primitives::U256;
use smallvec::SmallVec;

/// Maximum number of instructions in a cloned continuation trace.
/// Keeps clone size bounded to avoid code bloat.
const MAX_CLONE_TRACE_INSTS: usize = 64;
/// Maximum total number of cloned instructions per contract.
const MAX_TOTAL_CLONED_INSTS: usize = 4096;

impl<'a> Bytecode<'a> {
    /// Clone shared continuation blocks so each call site gets a unique return target.
    ///
    /// This must run after `static_jump_analysis`, `mark_dead_code`, and `rebuild_cfg`,
    /// but before `block_analysis`.
    #[instrument(name = "clone_cont", level = "debug", skip_all)]
    pub(crate) fn clone_continuations(&mut self) {
        if self.cfg.blocks.is_empty() || !self.has_dynamic_jumps {
            return;
        }

        let summaries = self.compute_local_summaries();

        // Find shared continuation targets: continuation PCs pushed by multiple call sites.
        // Group by continuation U256Idx → list of (caller_block, PrivateCallSummary).
        let mut cont_to_callers: crate::FxHashMap<
            U256Idx,
            SmallVec<[(Block, super::block_analysis::PrivateCallSummary); 4]>,
        > = crate::FxHashMap::default();

        for (bid, summary) in summaries.iter_enumerated() {
            if let Some(ref call) = summary.private_call {
                cont_to_callers.entry(call.continuation).or_default().push((bid, *call));
            }
        }

        // Only clone continuations that have multiple callers.
        let shared: Vec<_> =
            cont_to_callers.into_iter().filter(|(_, callers)| callers.len() > 1).collect();

        if shared.is_empty() {
            return;
        }

        let mut total_cloned = 0usize;
        let mut cloned_count = 0usize;

        for (cont_idx, callers) in &shared {
            // Resolve the continuation PC to a block.
            let cont_pc = {
                let val = *self.u256_interner.borrow().get(*cont_idx);
                match usize::try_from(val) {
                    Ok(pc) if self.is_valid_jump(pc) => pc,
                    _ => continue,
                }
            };
            let cont_inst = self.pc_to_inst(cont_pc);
            let cont_inst = self.redirects.get(&cont_inst).copied().unwrap_or(cont_inst);
            let Some(cont_block) = self.cfg.inst_to_block.get(cont_inst).copied().flatten() else {
                continue;
            };

            // Collect the continuation trace: a linear sequence of blocks starting at the
            // continuation, following fallthroughs until we hit a non-fallthrough terminator.
            let trace = self.collect_clone_trace(cont_block, &summaries);
            if trace.is_empty() {
                continue;
            }
            let trace_inst_count: usize = trace
                .iter()
                .map(|&bid| {
                    let block = &self.cfg.blocks[bid];
                    block.insts().filter(|&i| !self.insts[i].is_dead_code()).count()
                })
                .sum();

            if trace_inst_count > MAX_CLONE_TRACE_INSTS {
                trace!(cont_pc, trace_inst_count, "skipping large continuation trace");
                continue;
            }
            if total_cloned + trace_inst_count * (callers.len() - 1) > MAX_TOTAL_CLONED_INSTS {
                trace!(cont_pc, "clone budget exhausted");
                break;
            }

            // Keep the first caller pointing to the original continuation.
            // Clone for all other callers.
            for (_, call) in &callers[1..] {
                let synth_pc = self.clone_trace(&trace, cont_pc);
                if let Some(synth_pc) = synth_pc {
                    // Rewrite the caller's continuation PUSH to point at the clone.
                    self.set_push_override(call.continuation_push, U256::from(synth_pc));
                    total_cloned += trace_inst_count;
                    cloned_count += 1;
                }
            }
        }

        debug!(cloned_count, total_cloned, "cloned continuations");
    }

    /// Collect a linear continuation trace starting at `start_block`.
    ///
    /// Follows fallthrough edges only. Stops when:
    /// - The terminator does not fall through (JUMP, REVERT, STOP, etc.)
    /// - The block is a private function call (its continuation is a separate concern)
    /// - A cycle is detected
    /// - The block contains a JUMPI (would need internal retargeting in the clone)
    fn collect_clone_trace(
        &self,
        start_block: Block,
        summaries: &IndexVec<Block, BlockLocalSummary>,
    ) -> SmallVec<[Block; 8]> {
        let mut trace = SmallVec::new();
        let mut bid = start_block;

        loop {
            let block = &self.cfg.blocks[bid];
            if block.dead {
                break;
            }

            // Don't clone blocks that are private function calls — they push their own
            // continuation which complicates cloning.
            if bid != start_block && summaries[bid].private_call.is_some() {
                break;
            }

            // Check for JUMPI inside the block — would need internal retargeting.
            let has_jumpi = block.insts().any(|i| {
                let inst = &self.insts[i];
                inst.opcode == op::JUMPI && !inst.is_dead_code()
            });
            if has_jumpi && bid != start_block {
                // Allow JUMPI in the start block only if it's the terminator
                // (we clone the whole block, including the JUMPI).
                break;
            }

            trace.push(bid);

            let term = &self.insts[block.terminator()];
            if !term.can_fall_through() {
                // End of trace (diverging or unconditional JUMP).
                break;
            }

            // Find the fallthrough successor.
            if let Some(&succ) = block.succs.first() {
                if trace.contains(&succ) {
                    // Cycle detected.
                    break;
                }
                bid = succ;
            } else {
                break;
            }
        }

        // Safety: a cloned trace must end with a non-fallthrough terminator.
        // If the trace ends with a fallthrough (e.g. JUMPI), trim it because
        // the cloned instructions won't have the fallthrough successor appended.
        while let Some(&last) = trace.last() {
            let term = &self.insts[self.cfg.blocks[last].terminator()];
            if term.can_fall_through() {
                trace.pop();
            } else {
                break;
            }
        }

        trace
    }

    /// Clone a trace of blocks, returning the synthetic PC of the cloned entry JUMPDEST.
    ///
    /// Appends cloned instructions to `self.insts` and registers the synthetic JUMPDEST
    /// in `jumpdests` and `pc_to_inst`.
    fn clone_trace(&mut self, trace: &[Block], original_pc: usize) -> Option<u32> {
        if trace.is_empty() {
            return None;
        }

        // Collect instruction ranges first to avoid borrow conflicts.
        let inst_ranges: SmallVec<[(Inst, Inst); 8]> = trace
            .iter()
            .map(|&bid| {
                let block = &self.cfg.blocks[bid];
                (block.insts.start, block.insts.end)
            })
            .collect();

        let mut synth_entry_pc = None;

        for (ti, &(range_start, range_end)) in inst_ranges.iter().enumerate() {
            let mut i = range_start;
            while i < range_end {
                let orig = &self.insts[i];
                if orig.is_dead_code() {
                    i += 1;
                    continue;
                }

                let mut cloned = orig.clone();
                // Reset sections — they'll be recomputed after rebuild_cfg.
                cloned.gas_section = Default::default();
                cloned.stack_section = Default::default();

                let cloned_inst = self.insts.next_idx();

                if orig.is_jumpdest() && ti == 0 && i == range_start {
                    // This is the entry JUMPDEST of the continuation trace.
                    // Give it a synthetic PC.
                    let pc = self.alloc_synth_pc();
                    cloned.pc = pc;
                    cloned.data = 1; // Mark as reachable.
                    self.pc_to_inst.insert(pc, cloned_inst);
                    synth_entry_pc = Some(pc);
                }

                self.insts.push(cloned);
                i += 1;
            }
        }

        if synth_entry_pc.is_none() {
            trace!(original_pc, "trace has no entry JUMPDEST, skipping clone");
        }

        synth_entry_pc
    }

    /// Allocate a synthetic PC beyond the original code boundary.
    fn alloc_synth_pc(&mut self) -> u32 {
        let pc = self.jumpdests.len() as u32;
        self.jumpdests.push(true);
        pc
    }
}

#[cfg(test)]
mod tests {
    use crate::bytecode::passes::block_analysis::tests::analyze_code;
    use revm_bytecode::opcode as op;

    #[test]
    fn clone_shared_continuation() {
        // Two call sites sharing the same continuation (ret).
        // After cloning, each call site should get its own continuation,
        // allowing the function's return JUMP to be fully resolved.
        let ret: u8 = 12;
        let func: u8 = 17;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            // Call site 1.
            op::PUSH1, ret,         // pc=0: push return address
            op::PUSH1, func,        // pc=2: push function entry
            op::JUMP,               // pc=4: call function
            // Call site 2.
            op::PUSH1, ret,         // pc=5: push SAME return address
            op::PUSH1, func,        // pc=7: push function entry
            op::JUMP,               // pc=9: call function
            op::INVALID,            // pc=10
            op::INVALID,            // pc=11
            // Shared continuation.
            op::JUMPDEST,           // pc=12: ret
            op::POP,                // pc=13: drop result
            op::STOP,               // pc=14
            op::INVALID,            // pc=15
            op::INVALID,            // pc=16
            // Internal function.
            op::JUMPDEST,           // pc=17: func entry
            op::PUSH1, 0x42,        // pc=18: push a result
            op::SWAP1,              // pc=20: swap result and return address
            op::JUMP,               // pc=21: return (dynamic)
        ]);
        eprintln!("{bytecode}");

        // The function's return JUMP should be fully resolved.
        // Without cloning, the two callers push the same return address so the
        // flat fixpoint couldn't distinguish them; with cloning, each call site
        // gets a unique continuation, enabling the return JUMP to resolve.
        assert!(!bytecode.has_dynamic_jumps, "expected all jumps to be resolved");
    }

    #[test]
    fn clone_does_not_affect_unique_continuations() {
        // Two call sites with DIFFERENT continuations — no cloning should occur.
        let ret1: u8 = 5;
        let ret2: u8 = 12;
        let func: u8 = 15;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            // Call site 1.
            op::PUSH1, ret1,
            op::PUSH1, func,
            op::JUMP,
            op::JUMPDEST,       // ret1
            op::POP,
            // Call site 2.
            op::PUSH1, ret2,
            op::PUSH1, func,
            op::JUMP,
            op::JUMPDEST,       // ret2
            op::POP,
            op::STOP,
            // Internal function.
            op::JUMPDEST,       // func
            op::PUSH1, 0x42,
            op::SWAP1,
            op::JUMP,
        ]);
        eprintln!("{bytecode}");

        // Each call site already has a unique continuation, no cloning needed.
        let has_override = bytecode.iter_insts().any(|(_, d)| d.has_push_override());
        assert!(!has_override, "should not clone unique continuations");
        assert!(!bytecode.has_dynamic_jumps, "expected all jumps to be resolved");
    }

    #[test]
    fn clone_three_callers_shared_continuation() {
        // Three call sites sharing the same continuation.
        let ret: u8 = 18;
        let func: u8 = 23;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            // Call site 1.
            op::PUSH1, ret,         // pc=0
            op::PUSH1, func,        // pc=2
            op::JUMP,               // pc=4
            // Call site 2.
            op::PUSH1, ret,         // pc=5
            op::PUSH1, func,        // pc=7
            op::JUMP,               // pc=9
            // Call site 3.
            op::PUSH1, ret,         // pc=10
            op::PUSH1, func,        // pc=12
            op::JUMP,               // pc=14
            op::INVALID,            // pc=15
            op::INVALID,            // pc=16
            op::INVALID,            // pc=17
            // Shared continuation.
            op::JUMPDEST,           // pc=18
            op::POP,                // pc=19
            op::STOP,               // pc=20
            op::INVALID,            // pc=21
            op::INVALID,            // pc=22
            // Internal function.
            op::JUMPDEST,           // pc=23
            op::PUSH1, 0x42,        // pc=24
            op::SWAP1,              // pc=26
            op::JUMP,               // pc=27: return
        ]);
        eprintln!("{bytecode}");

        assert!(!bytecode.has_dynamic_jumps, "expected all jumps to be resolved");
    }
}

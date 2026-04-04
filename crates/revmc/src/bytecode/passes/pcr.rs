//! Private call/return (PCR) detection and resolution.
//!
//! Identifies private function call/return patterns in the CFG and resolves
//! return edges using context-sensitive call-string analysis. The results are
//! returned as *hints* that are seeded into the abstract interpreter's fixpoint
//! to improve jump resolution.

use super::block_analysis::{AbsValue, Block, ConstSetInterner, JumpTarget};
use super::StackSection;
use crate::bytecode::{Bytecode, Inst, InstFlags};
use bitvec::vec::BitVec;
use oxc_index::IndexVec;
use revm_bytecode::opcode as op;
use smallvec::SmallVec;
use std::collections::VecDeque;
use tracing::{debug, instrument, trace};

/// Local per-block summary for private call/return detection.
///
/// Computed by a single-pass stack simulation of each block, without any fixpoint.
/// These properties mirror gigahorse's `PrivateFunctionCall` and `PrivateFunctionReturn`
/// relations from `local_components.dl`.
#[derive(Clone, Debug, Default)]
struct LocalBlockSummary {
    /// If this block is a private function call: `(callee_inst, continuation_inst)`.
    /// Detected when the block's terminator is a `STATIC_JUMP` (adjacent PUSH+JUMP)
    /// and the block also pushes a valid JUMPDEST label that survives to exit.
    private_call: Option<PrivateCallInfo>,
    /// Whether this block is a private function return: the terminator is a JUMP
    /// with no `STATIC_JUMP` flag (target comes from the stack).
    is_return: bool,
}

/// Information about a private function call detected in a block.
#[derive(Clone, Debug)]
struct PrivateCallInfo {
    /// The callee function entry instruction (JUMPDEST target of the static jump).
    callee: Inst,
    /// The continuation instruction (JUMPDEST where the callee should return to).
    continuation: Inst,
}

/// A PCR hint: a return jump and its resolved targets.
pub(super) struct PcrHint {
    /// The return jump instruction.
    pub(super) jump_inst: Inst,
    /// The resolved target instructions.
    pub(super) targets: SmallVec<[Inst; 4]>,
}

impl Bytecode<'_> {
    /// Computes private call/return hints for the abstract interpreter.
    ///
    /// Returns PCR hints containing return edges discovered by context-sensitive
    /// call-string analysis.
    #[instrument(name = "pcr", level = "debug", skip_all)]
    pub(super) fn compute_pcr_hints(&mut self) -> Vec<PcrHint> {
        if self.cfg.blocks.is_empty() || !self.has_dynamic_jumps {
            return Vec::new();
        }

        let summaries = self.compute_local_summaries();
        let hints = self.resolve_private_calls(&summaries);

        if !hints.is_empty() {
            debug!(n = hints.len(), "PCR hints for fixpoint");
        }
        hints
    }

    /// Compute local per-block summaries for private call/return detection.
    ///
    /// For each block, simulates the stack effect to determine:
    /// - Whether the block is a private function call (static jump + pushed label).
    /// - Whether the block is a private function return (dynamic jump from stack).
    fn compute_local_summaries(&mut self) -> IndexVec<Block, LocalBlockSummary> {
        let mut summaries =
            IndexVec::from_vec(vec![LocalBlockSummary::default(); self.cfg.blocks.len()]);

        let empty_sets = ConstSetInterner::new();
        let mut stack = Vec::new();

        for bid in self.cfg.blocks.indices() {
            let block = &self.cfg.blocks[bid];

            let term_inst = block.terminator();
            let term = &self.insts[term_inst];

            if !term.is_jump() {
                continue;
            }

            // Private function return: JUMP with no static target, and the block
            // only does stack manipulation (the jump target comes from the entry
            // stack, not from memory/storage/computation).
            if !term.flags.contains(InstFlags::STATIC_JUMP) {
                let is_pure_return = block.insts().all(|i| {
                    let inst = &self.insts[i];
                    if inst.is_dead_code() || inst.flags.contains(InstFlags::NOOP) {
                        return true;
                    }
                    matches!(
                        inst.opcode,
                        op::JUMPDEST
                            | op::JUMP
                            | op::POP
                            | op::DUP1..=op::DUP16
                            | op::SWAP1..=op::SWAP16
                            | op::PUSH0..=op::PUSH32
                    )
                });
                if is_pure_return {
                    summaries[bid].is_return = true;
                }
                continue;
            }

            // Private function call: STATIC_JUMP + a pushed label surviving to exit.
            // The callee is the static jump target.
            let callee = Inst::from_usize(term.data as usize);

            // Interpret the block with Top inputs to find which values survive to exit.
            // Reuses the same abstract interpreter as block_analysis_local.
            let section =
                StackSection::from_stack_io(block.insts().map(|i| self.insts[i].stack_io()));
            stack.clear();
            stack.resize(section.inputs as usize, AbsValue::Top);
            if !self.interpret_block(block.insts(), &mut stack) {
                continue;
            }

            // Find the deepest surviving label (the one pushed earliest, which is the
            // continuation/return address in the standard PUSH ret_addr; PUSH func; JUMP
            // pattern). In Solidity, the return address is typically pushed before the
            // function arguments and callee address, so it ends up deeper in the stack.
            let continuation = stack.iter().find_map(|v| {
                if let JumpTarget::Const(inst) = self.resolve_jump_operand(*v, &empty_sets) {
                    Some(inst)
                } else {
                    None
                }
            });
            if let Some(continuation) = continuation {
                summaries[bid].private_call = Some(PrivateCallInfo { callee, continuation });
            }
        }

        for (bid, summary) in summaries.iter_enumerated() {
            if let Some(ref call) = summary.private_call {
                trace!(
                    %bid,
                    callee = %call.callee,
                    continuation = %call.continuation,
                    "private call"
                );
            }
            if summary.is_return {
                trace!(%bid, "private return");
            }
        }

        summaries
    }

    /// Resolve private function return jumps using context-sensitive graph traversal.
    ///
    /// Uses the local summaries to trace call-strings through the CFG. When a
    /// `PrivateFunctionReturn` block is reached, the call-string context reveals which
    /// caller pushed the return address, allowing the return edge to be materialized.
    ///
    /// Returns resolved jump targets for return blocks.
    #[instrument(name = "resolve_calls", level = "debug", skip_all)]
    fn resolve_private_calls(
        &self,
        summaries: &IndexVec<Block, LocalBlockSummary>,
    ) -> Vec<PcrHint> {
        let num_blocks = self.cfg.blocks.len();

        // Maximum call-string depth. Each entry is a caller block ID.
        const MAX_CONTEXT_DEPTH: usize = 8;

        // Context = call-string (stack of caller block IDs, most recent first).
        type Context = SmallVec<[Block; 4]>;

        // Per (block, context): set of visited states.
        // We use a map from Block to set of contexts to avoid exponential blowup.
        let mut visited: IndexVec<Block, SmallVec<[Context; 2]>> =
            IndexVec::from_vec(vec![SmallVec::new(); num_blocks]);

        // Per return-block: set of discovered continuation targets.
        let mut return_targets: IndexVec<Block, SmallVec<[Inst; 4]>> =
            IndexVec::from_vec(vec![SmallVec::new(); num_blocks]);
        // Return blocks that were reached from a non-call path (opaque caller or empty context).
        let mut tainted_returns: BitVec = BitVec::repeat(false, num_blocks);

        // Worklist of (block, context) pairs.
        let mut worklist: VecDeque<(Block, Context)> = VecDeque::new();

        // Seed: entry block with empty context.
        let empty_ctx: Context = SmallVec::new();
        visited[Block::from_usize(0)].push(empty_ctx.clone());
        worklist.push_back((Block::from_usize(0), empty_ctx));

        let mut iterations = 0;
        let max_iterations = num_blocks * 64;

        while let Some((bid, ctx)) = worklist.pop_front() {
            iterations += 1;
            if iterations > max_iterations {
                debug!("resolve_private_calls: did not converge after {iterations} iterations");
                break;
            }

            let block = &self.cfg.blocks[bid];

            let summary = &summaries[bid];

            if let Some(ref call) = summary.private_call {
                // Private function call: push caller onto context, follow to callee.
                let mut new_ctx = ctx.clone();
                if new_ctx.len() >= MAX_CONTEXT_DEPTH {
                    // Truncate oldest entry (last element) to keep bounded.
                    new_ctx.pop();
                }
                new_ctx.insert(0, bid);

                // Follow to callee.
                if let Some(callee_block) = self.cfg.inst_to_block[call.callee]
                    && !visited[callee_block].contains(&new_ctx)
                {
                    visited[callee_block].push(new_ctx.clone());
                    worklist.push_back((callee_block, new_ctx));
                }

                // Also follow the fallthrough edge (for JUMPI terminators).
                let term = &self.insts[block.terminator()];
                if term.opcode == op::JUMPI {
                    for &succ in &block.succs {
                        if let Some(callee_block) = self.cfg.inst_to_block[call.callee]
                            && succ != callee_block
                            && !visited[succ].contains(&ctx)
                        {
                            visited[succ].push(ctx.clone());
                            worklist.push_back((succ, ctx.clone()));
                        }
                    }
                }
            } else if summary.is_return {
                // Private function return: look up the context to find the caller
                // that pushed the matching continuation address (CutToCaller).
                if let Some(caller_bid) = ctx.first().copied() {
                    if let Some(ref caller_call) = summaries[caller_bid].private_call {
                        let continuation = caller_call.continuation;

                        // Record this return -> continuation edge.
                        return_targets[bid].push(continuation);

                        // Pop the caller from context and continue at the continuation.
                        let new_ctx: Context = ctx[1..].into();
                        if let Some(cont_block) = self.cfg.inst_to_block[continuation]
                            && !visited[cont_block].contains(&new_ctx)
                        {
                            visited[cont_block].push(new_ctx.clone());
                            worklist.push_back((cont_block, new_ctx));
                        }
                    } else {
                        // Caller in context doesn't have a call summary — context is stale.
                        tainted_returns.set(bid.index(), true);
                    }
                } else {
                    // Empty context: reached from a non-call path (e.g. opaque caller).
                    tainted_returns.set(bid.index(), true);
                }
            } else {
                // Normal block: propagate to all successors with same context.
                for &succ in &block.succs {
                    if !visited[succ].contains(&ctx) {
                        visited[succ].push(ctx.clone());
                        worklist.push_back((succ, ctx.clone()));
                    }
                }
            }
        }

        debug!("resolve_private_calls: converged after {iterations} iterations");

        // Convert return_targets to PCR hints.
        // These are returned as hints — soundness is enforced downstream by the
        // abstract interpreter's `invalidate_suspect_jumps`.
        let mut hints = Vec::new();
        for (bid, targets) in return_targets.iter_enumerated() {
            if targets.is_empty() || tainted_returns[bid.index()] {
                continue;
            }
            let term_inst = self.cfg.blocks[bid].terminator();
            if self.insts[term_inst].flags.contains(InstFlags::STATIC_JUMP) {
                continue;
            }
            trace!(
                %bid,
                term = %term_inst,
                n_targets = targets.len(),
                "resolved private return"
            );
            hints.push(PcrHint { jump_inst: term_inst, targets: targets.clone() });
        }

        hints
    }
}

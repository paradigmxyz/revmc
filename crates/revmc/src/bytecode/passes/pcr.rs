//! Private call/return (PCR) detection and resolution.
//!
//! Identifies private function call/return patterns in the CFG and resolves
//! return edges using context-sensitive call-string analysis. The results are
//! returned as *hints* that are seeded into the abstract interpreter's fixpoint
//! to improve jump resolution.

use super::{
    StackSection,
    block_analysis::{AbsValue, Block, ConstSetInterner, JumpTarget, apply_stack_shuffle},
};
use crate::bytecode::{Bytecode, Inst, InstFlags};
use bitvec::vec::BitVec;
use oxc_index::IndexVec;
use revm_bytecode::opcode as op;
use smallvec::SmallVec;
use std::collections::VecDeque;
use tracing::{debug, instrument, trace};

/// Maximum call-string depth for context-sensitive traversal.
const MAX_CONTEXT_DEPTH: usize = 8;

/// Stack value provenance for return detection.
///
/// Tracks whether a stack slot originated from the block's entry stack or was
/// produced in-block. Only entry-stack provenance (`Input`) qualifies a dynamic
/// JUMP as a private function return.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Provenance {
    /// Value from the block's entry stack.
    Input,
    /// Value produced in-block (PUSH, arithmetic, memory/storage load, etc.).
    Local,
}

/// Call-string context: stack of caller block IDs, most recent first.
type Context = SmallVec<[Block; 8]>;

/// Context-sensitive worklist for the PCR graph traversal.
///
/// Tracks `(block, context)` pairs with per-block visited sets.
struct ContextWorklist {
    queue: VecDeque<(Block, Context)>,
    /// Per-block set of visited contexts.
    visited: IndexVec<Block, SmallVec<[Context; 2]>>,
}

impl ContextWorklist {
    fn new(num_blocks: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            visited: IndexVec::from_vec(vec![SmallVec::new(); num_blocks]),
        }
    }

    /// Enqueues `(block, ctx)` if not already visited with that context.
    fn push(&mut self, block: Block, ctx: Context) {
        if !self.visited[block].contains(&ctx) {
            self.visited[block].push(ctx.clone());
            self.queue.push_back((block, ctx));
        }
    }

    fn pop(&mut self) -> Option<(Block, Context)> {
        self.queue.pop_front()
    }
}

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
    /// whose operand has entry-stack provenance (passed by the caller).
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
            debug!(n = hints.len(), "hints for fixpoint");
        }
        hints
    }

    /// Simulates stack provenance through a block's instructions.
    ///
    /// Entry-stack slots start as `Input`; all in-block-produced values are `Local`.
    /// Stack-motion opcodes (DUP, SWAP, POP, PUSH) preserve or introduce provenance;
    /// all other opcodes pop their inputs and push `Local` outputs.
    ///
    /// Returns `false` if the simulation encounters an invalid stack underflow.
    fn simulate_provenance(
        &self,
        insts: impl IntoIterator<Item = Inst>,
        stack: &mut Vec<Provenance>,
    ) -> bool {
        for i in insts {
            let inst = &self.insts[i];
            if inst.is_dead_code() || inst.flags.contains(InstFlags::NOOP) {
                continue;
            }

            if let Some(ok) = apply_stack_shuffle(inst, &self.code, stack, Provenance::Input) {
                if !ok {
                    return false;
                }
            } else if matches!(inst.opcode, op::PUSH0..=op::PUSH32) {
                stack.push(Provenance::Local);
            } else {
                let (inp, out) = inst.stack_io();
                let inp = inp as usize;
                if stack.len() < inp {
                    return false;
                }
                stack.truncate(stack.len() - inp);
                for _ in 0..out {
                    stack.push(Provenance::Local);
                }
            }
        }
        true
    }

    /// Computes local per-block summaries for private call/return detection.
    ///
    /// For each block, uses provenance-based stack simulation to determine:
    /// - Whether the block is a private function call (static jump + pushed label).
    /// - Whether the block is a private function return (dynamic jump whose operand has entry-stack
    ///   provenance, i.e. the target was passed by the caller).
    fn compute_local_summaries(&mut self) -> IndexVec<Block, LocalBlockSummary> {
        let mut summaries =
            IndexVec::from_vec(vec![LocalBlockSummary::default(); self.cfg.blocks.len()]);

        let empty_sets = ConstSetInterner::new();
        let mut abs_stack = Vec::new();
        let mut prov_stack = Vec::new();

        for bid in self.cfg.blocks.indices() {
            let block = &self.cfg.blocks[bid];

            let term_inst = block.terminator();
            let term = &self.insts[term_inst];

            if !term.is_jump() {
                continue;
            }

            // Compute the block's entry stack size and simulate provenance.
            let section =
                StackSection::from_stack_io(block.insts().map(|i| self.insts[i].stack_io()));

            // Private function return: JUMP with no static target and the jump
            // operand has entry-stack provenance (was passed by the caller).
            if !term.flags.contains(InstFlags::STATIC_JUMP) {
                prov_stack.clear();
                prov_stack.resize(section.inputs as usize, Provenance::Input);
                // Simulate up to (but not including) the terminator JUMP, then
                // check if TOS has entry-stack provenance.
                let pre_term = block.insts().take_while(|&i| i != term_inst);
                if self.simulate_provenance(pre_term, &mut prov_stack)
                    && prov_stack.last() == Some(&Provenance::Input)
                {
                    summaries[bid].is_return = true;
                }
                continue;
            }

            // Private function call: single-target STATIC_JUMP + a pushed label
            // surviving to exit. Skip multi-target or invalid jumps since `term.data`
            // is not a valid callee for those.
            if term.flags.intersects(InstFlags::MULTI_JUMP | InstFlags::INVALID_JUMP) {
                continue;
            }
            let callee = Inst::from_usize(term.data as usize);

            // Interpret the block with Top inputs to find which values survive to exit.
            // Reuses the same abstract interpreter as block_analysis_local.
            abs_stack.clear();
            abs_stack.resize(section.inputs as usize, AbsValue::Top);
            if !self.interpret_block(block.insts(), &mut abs_stack) {
                continue;
            }

            // Find the deepest surviving label (the one pushed earliest, which is the
            // continuation/return address in the standard PUSH ret_addr; PUSH func; JUMP
            // pattern). In Solidity, the return address is typically pushed before the
            // function arguments and callee address, so it ends up deeper in the stack.
            let continuation = abs_stack.iter().find_map(|v| {
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

        if enabled!(tracing::Level::TRACE) {
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
        }

        summaries
    }

    /// Resolves private function return jumps using context-sensitive graph traversal.
    ///
    /// Uses the local summaries to trace call-strings through the CFG. When a
    /// `PrivateFunctionReturn` block is reached, the call-string context reveals which
    /// caller pushed the return address, allowing the return edge to be materialized.
    ///
    /// Returns are tainted (suppressed) when they are reachable from opaque entry points
    /// that PCR cannot model — ensuring soundness even with adversarial bytecode.
    #[instrument(name = "resolve_calls", level = "debug", skip_all)]
    fn resolve_private_calls(
        &self,
        summaries: &IndexVec<Block, LocalBlockSummary>,
    ) -> Vec<PcrHint> {
        let num_blocks = self.cfg.blocks.len();

        // Compute opaque entry points: blocks that might be entered by edges PCR
        // does not model, meaning return blocks reachable from them may have callers
        // PCR cannot discover.
        let tainted_returns = self.compute_opaque_taint(summaries);

        let mut wl = ContextWorklist::new(num_blocks);
        wl.push(Block::from_usize(0), SmallVec::new());

        // Per return-block: discovered continuation targets.
        let mut return_targets: IndexVec<Block, SmallVec<[Inst; 4]>> =
            IndexVec::from_vec(vec![SmallVec::new(); num_blocks]);

        let max_iterations = num_blocks * 64;
        let mut iterations = 0;
        let mut converged = true;

        while let Some((bid, ctx)) = wl.pop() {
            iterations += 1;
            if iterations > max_iterations {
                converged = false;
                break;
            }

            let block = &self.cfg.blocks[bid];
            let summary = &summaries[bid];

            if let Some(ref call) = summary.private_call {
                // Private function call: push caller onto context, follow to callee.
                let mut new_ctx = ctx.clone();
                if new_ctx.len() >= MAX_CONTEXT_DEPTH {
                    new_ctx.pop();
                }
                new_ctx.insert(0, bid);

                let callee_block = self.cfg.inst_to_block[call.callee];

                if let Some(callee_block) = callee_block {
                    wl.push(callee_block, new_ctx);
                }

                // Also follow the fallthrough edge (for JUMPI terminators).
                let term = &self.insts[block.terminator()];
                if term.opcode == op::JUMPI {
                    for &succ in &block.succs {
                        if callee_block != Some(succ) {
                            wl.push(succ, ctx.clone());
                        }
                    }
                }
            } else if summary.is_return {
                // Private function return: pop the caller from the context
                // to find the matching continuation address.
                if let Some(caller_bid) = ctx.first().copied()
                    && let Some(ref caller_call) = summaries[caller_bid].private_call
                {
                    let continuation = caller_call.continuation;
                    if !return_targets[bid].contains(&continuation) {
                        return_targets[bid].push(continuation);
                    }

                    let new_ctx: Context = ctx[1..].into();
                    if let Some(cont_block) = self.cfg.inst_to_block[continuation] {
                        wl.push(cont_block, new_ctx);
                    }
                }
                // Note: returns reached with empty/invalid context are not
                // propagated further. Tainting is handled structurally by
                // `compute_opaque_taint` instead.
            } else {
                // Normal block: propagate to all successors with same context.
                for &succ in &block.succs {
                    wl.push(succ, ctx.clone());
                }
            }
        }

        debug!(
            "{msg} after {iterations} iterations (max={max_iterations})",
            msg = if converged { "converged" } else { "did not converge" },
        );

        // Partial exploration can miss valid continuations, making the subset
        // unsound. Discard all hints on non-convergence.
        if !converged {
            return Vec::new();
        }

        // Collect hints from resolved return targets.
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

    /// Computes which candidate return blocks are tainted by opaque entry points.
    ///
    /// An opaque entry is a block that might be entered by edges PCR cannot model
    /// (unresolved dynamic jumps, non-private-call predecessors of callee blocks).
    /// Any candidate return reachable from an opaque entry is tainted because PCR
    /// might not have discovered all its callers.
    fn compute_opaque_taint(&self, summaries: &IndexVec<Block, LocalBlockSummary>) -> BitVec {
        let num_blocks = self.cfg.blocks.len();
        let mut opaque: BitVec = BitVec::repeat(false, num_blocks);

        // Collect callee blocks that are entered by detected private calls.
        let mut private_call_preds: IndexVec<Block, SmallVec<[Block; 2]>> =
            IndexVec::from_vec(vec![SmallVec::new(); num_blocks]);
        // Whether any unmodeled dynamic jump exists (not a private call, not a return).
        let mut has_unmodeled_jump = false;

        for (bid, summary) in summaries.iter_enumerated() {
            if let Some(ref call) = summary.private_call {
                if let Some(callee_block) = self.cfg.inst_to_block[call.callee] {
                    private_call_preds[callee_block].push(bid);
                }
            } else {
                let term_inst = self.cfg.blocks[bid].terminator();
                let term = &self.insts[term_inst];
                if term.is_jump()
                    && !term.flags.contains(InstFlags::STATIC_JUMP)
                    && !summary.is_return
                {
                    has_unmodeled_jump = true;
                }
            }
        }

        // Seed A: callee blocks with non-private-call predecessors in the static CFG.
        for bid in self.cfg.blocks.indices() {
            if private_call_preds[bid].is_empty() {
                continue;
            }
            let has_external_pred = self.cfg.blocks[bid]
                .preds
                .iter()
                .any(|pred| !private_call_preds[bid].contains(pred));
            if has_external_pred {
                opaque.set(bid.index(), true);
            }
        }

        // Seed B: if unmodeled dynamic jumps exist, any JUMPDEST block is a potential
        // target (bytecode is user-controlled — any JUMPDEST can be jumped to).
        if has_unmodeled_jump {
            for bid in self.cfg.blocks.indices() {
                if self.insts[self.cfg.blocks[bid].insts.start].is_jumpdest() {
                    opaque.set(bid.index(), true);
                }
            }
        }

        // Propagate opaque flag forward through static CFG edges and private-call
        // edges to find all candidate returns reachable from opaque entries.
        let mut tainted: BitVec = BitVec::repeat(false, num_blocks);
        let mut queue: VecDeque<Block> = VecDeque::new();
        for bid in self.cfg.blocks.indices() {
            if opaque[bid.index()] {
                queue.push_back(bid);
            }
        }
        while let Some(bid) = queue.pop_front() {
            let summary = &summaries[bid];

            if summary.is_return {
                tainted.set(bid.index(), true);
                // Don't propagate past return blocks — the return edge goes back
                // to the caller's continuation, which has its own entry analysis.
                continue;
            }

            // Follow static CFG successors.
            for &succ in &self.cfg.blocks[bid].succs {
                if !opaque[succ.index()] {
                    opaque.set(succ.index(), true);
                    queue.push_back(succ);
                }
            }

            // Follow private-call edges into callees.
            if let Some(ref call) = summary.private_call
                && let Some(callee_block) = self.cfg.inst_to_block[call.callee]
                && !opaque[callee_block.index()]
            {
                opaque.set(callee_block.index(), true);
                queue.push_back(callee_block);
            }
        }

        tainted
    }
}

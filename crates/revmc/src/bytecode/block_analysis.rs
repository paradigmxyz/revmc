//! Abstract stack interpretation for resolving dynamic jump targets and constant propagation.
//!
//! This pass builds a basic-block CFG and performs constant propagation over an abstract stack
//! to resolve jump targets that the simple `static_jump_analysis` (which only looks at adjacent
//! `PUSH + JUMP/JUMPI`) cannot handle.
//!
//! After the fixpoint converges, the abstract stack state at each instruction is persisted so
//! that later passes (e.g. code generation) can query the known-constant value of stack operands.
//!
//! The abstract domain is:
//! - `Const(U256)` — a known constant value.
//! - `Top` — reachable but unknown.
//!
//! Block input states use `Bottom` (unreachable) at the block level.

use super::{Bytecode, InstFlags};
use bitvec::vec::BitVec;
use revm_bytecode::opcode as op;
use revm_primitives::U256;
use smallvec::SmallVec;
use std::{collections::VecDeque, num::NonZeroU32, ops::Range};

/// Abstract value on the stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AbsValue {
    Const(U256),
    Top,
}

/// The abstract stack state at a particular instruction, before it executes.
///
/// Stored per-instruction after block analysis so that codegen can query operand constants.
#[derive(Clone, Debug)]
pub(crate) enum StackSnapshot {
    /// The instruction's abstract stack is known.
    Known(SmallVec<[Option<U256>; 2]>),
    /// The instruction is unreachable or the analysis gave up on this block.
    Unknown,
}

impl StackSnapshot {
    /// Returns the constant value of the operand at `depth` from the top of the stack.
    ///
    /// `depth` 0 is TOS (first popped), 1 is second, etc.
    pub(crate) fn operand(&self, depth: usize) -> Option<U256> {
        match self {
            Self::Known(stack) => {
                let idx = stack.len().checked_sub(1 + depth)?;
                stack[idx]
            }
            Self::Unknown => None,
        }
    }
}

impl AbsValue {
    fn as_const(self) -> Option<U256> {
        match self {
            Self::Const(v) => Some(v),
            Self::Top => None,
        }
    }

    /// Lattice join: two values merge to their least upper bound.
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) if a == b => Self::Const(a),
            _ => Self::Top,
        }
    }
}

/// Abstract state at the entry of a block.
#[derive(Clone, Debug)]
enum BlockState {
    /// Block has not been reached yet.
    Bottom,
    /// Block has been reached with a known stack state.
    Known(Vec<AbsValue>),
    /// Predecessors have incompatible stack heights; give up on this block.
    Conflict,
}

impl BlockState {
    /// Force this state to Conflict. Returns `true` if the state changed.
    fn conflict(&mut self) -> bool {
        if matches!(self, Self::Conflict) {
            false
        } else {
            *self = Self::Conflict;
            true
        }
    }

    /// Join another incoming state into this one. Returns `true` if the state changed.
    fn join(&mut self, incoming: &[AbsValue]) -> bool {
        match self {
            Self::Bottom => {
                *self = Self::Known(incoming.to_vec());
                true
            }
            Self::Known(existing) => {
                if existing.len() != incoming.len() {
                    *self = Self::Conflict;
                    return true;
                }
                let mut changed = false;
                for (slot, inc) in existing.iter_mut().zip(incoming) {
                    let joined = slot.join(*inc);
                    if joined != *slot {
                        *slot = joined;
                        changed = true;
                    }
                }
                changed
            }
            Self::Conflict => false,
        }
    }
}

/// A block index in the CFG, backed by `NonZeroU32` with +1 encoding.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
struct Block(NonZeroU32);

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Block({})", self.index())
    }
}

impl Block {
    /// Creates a new block index from a `usize`.
    #[inline]
    fn new(index: usize) -> Self {
        Self(NonZeroU32::new(index as u32 + 1).expect("block index overflow"))
    }

    /// Returns the index as a `usize`.
    #[inline]
    fn index(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

/// Sentinel value for instructions not mapped to any block.
const NO_BLOCK: Option<Block> = None;

/// FIFO worklist with deduplication.
struct Worklist {
    queue: VecDeque<Block>,
    in_queue: BitVec,
}

impl Worklist {
    fn new(size: usize) -> Self {
        Self { queue: VecDeque::new(), in_queue: BitVec::repeat(false, size) }
    }

    fn push(&mut self, id: Block) {
        let idx = id.index();
        if !self.in_queue[idx] {
            self.in_queue.set(idx, true);
            self.queue.push_back(id);
        }
    }

    fn pop(&mut self) -> Option<Block> {
        let id = self.queue.pop_front()?;
        self.in_queue.set(id.index(), false);
        Some(id)
    }
}

/// A basic block in the CFG.
struct BasicBlock {
    /// Instruction index range (exclusive end). The terminator is at `insts.end - 1`.
    insts: Range<usize>,
    /// Predecessor block IDs.
    preds: SmallVec<[Block; 2]>,
    /// Successor block IDs.
    succs: SmallVec<[Block; 2]>,
    /// Whether all instructions in this block are dead code.
    dead: bool,
}

/// Resolved jump target after fixpoint.
#[derive(Clone, Debug)]
enum JumpTarget {
    /// Not yet observed.
    Bottom,
    /// Known constant target instruction index.
    Const(usize),
    /// Known constant but invalid target.
    Invalid,
    /// Multiple different targets or unknown.
    Top,
}

/// CFG for abstract interpretation.
struct Cfg {
    blocks: Vec<BasicBlock>,
    /// Maps instruction index to block ID. `None` for dead-code instructions.
    inst_to_block: Vec<Option<Block>>,
}

impl Bytecode<'_> {
    /// Runs abstract stack interpretation to resolve additional jump targets.
    ///
    /// Also computes and stores per-instruction stack snapshots for constant propagation.
    #[instrument(name = "block_analysis", level = "debug", skip_all)]
    pub(crate) fn block_analysis(&mut self) {
        let cfg = self.build_cfg();
        if cfg.blocks.is_empty() {
            return;
        }

        let mut snapshots = vec![StackSnapshot::Unknown; self.insts.len()];
        let (resolved, count) = self.run_abstract_interp(&cfg, &mut snapshots);
        self.stack_snapshots = snapshots;

        if count == 0 {
            return;
        }

        // Check if any jump remains unresolved (Top).
        let has_top_jump = resolved.iter().any(|(_, t)| matches!(t, JumpTarget::Top));

        // Commit resolved targets.
        let mut newly_resolved = 0u32;
        for &(jump_inst, ref target) in &resolved {
            let jump = &self.insts[jump_inst];
            // Skip if already resolved by static_jump_analysis.
            if jump.flags.contains(InstFlags::STATIC_JUMP) {
                continue;
            }

            match target {
                JumpTarget::Const(target_inst) => {
                    let target_inst = *target_inst;
                    debug_assert_eq!(
                        self.insts[target_inst].opcode,
                        op::JUMPDEST,
                        "block_analysis resolved to non-JUMPDEST"
                    );
                    self.insts[jump_inst].flags |=
                        InstFlags::STATIC_JUMP | InstFlags::BLOCK_RESOLVED_JUMP;
                    self.insts[jump_inst].data = target_inst as u32;
                    // Mark JUMPDEST as reachable.
                    self.insts[target_inst].data = 1;
                    newly_resolved += 1;
                    trace!(jump_inst, target_inst, "resolved jump");
                }
                JumpTarget::Invalid => {
                    self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP
                        | InstFlags::INVALID_JUMP
                        | InstFlags::BLOCK_RESOLVED_JUMP;
                    newly_resolved += 1;
                    trace!(jump_inst, "resolved invalid jump");
                }
                JumpTarget::Bottom if !has_top_jump => {
                    // Truly unreachable: no unresolved jumps remain, so this
                    // code cannot be reached at runtime. Mark as invalid.
                    self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP | InstFlags::INVALID_JUMP;
                    trace!(jump_inst, "unreachable jump");
                }
                JumpTarget::Bottom => {
                    // Unreachable according to the analysis, but there are
                    // unresolved (Top) jumps that might reach this code at
                    // runtime. Leave as-is.
                    trace!(jump_inst, "unreachable jump (not marking, has_top_jump)");
                }
                JumpTarget::Top => {}
            }
        }

        debug!(newly_resolved, "block_analysis complete");

        // Recompute dynamic jumps flag.
        self.has_dynamic_jumps = self
            .insts
            .iter()
            .any(|inst| inst.is_legacy_jump() && !inst.flags.contains(InstFlags::STATIC_JUMP));
    }

    /// Build a basic-block CFG from the instruction list.
    fn build_cfg(&self) -> Cfg {
        let n = self.insts.len();
        if n == 0 {
            return Cfg { blocks: Vec::new(), inst_to_block: Vec::new() };
        }

        // Identify block leaders.
        let mut is_leader: BitVec = BitVec::repeat(false, n);
        is_leader.set(0, true);

        for (i, inst) in self.insts.iter().enumerate() {
            if inst.is_dead_code() {
                continue;
            }
            if inst.is_reachable_jumpdest(self.has_dynamic_jumps) {
                is_leader.set(i, true);
            }
            if inst.is_branching() || inst.is_diverging() {
                // Next instruction is a leader (if it exists).
                if i + 1 < n {
                    is_leader.set(i + 1, true);
                }
            }
        }

        // Build blocks.
        let mut blocks = Vec::new();
        let mut inst_to_block = vec![NO_BLOCK; n];
        let mut current_start = None;

        let finish_block = |start: usize,
                            end: usize,
                            blocks: &mut Vec<BasicBlock>,
                            inst_to_block: &mut [Option<Block>]| {
            let block_id = Block::new(blocks.len());
            let range = start..end;
            let dead = range.clone().all(|j| self.insts[j].is_dead_code());
            for j in range.clone() {
                if !self.insts[j].is_dead_code() {
                    inst_to_block[j] = Some(block_id);
                }
            }
            blocks.push(BasicBlock {
                insts: range,
                preds: SmallVec::new(),
                succs: SmallVec::new(),
                dead,
            });
        };

        for i in 0..n {
            if self.insts[i].is_dead_code() {
                continue;
            }

            if is_leader[i] || current_start.is_none() {
                if let Some(start) = current_start {
                    finish_block(start, i, &mut blocks, &mut inst_to_block);
                }
                current_start = Some(i);
            }
        }

        // Close the last block.
        if let Some(start) = current_start {
            finish_block(start, n, &mut blocks, &mut inst_to_block);
        }

        // Build edges based on known control flow.
        let num_blocks = blocks.len();
        for bid_idx in 0..num_blocks {
            let bid = Block::new(bid_idx);
            let block = &blocks[bid_idx];
            if block.dead {
                continue;
            }
            let term_idx = block.insts.end - 1;
            let term = &self.insts[term_idx];

            // Fallthrough edge: if the terminator doesn't unconditionally branch/diverge.
            let has_fallthrough = !term.is_diverging() && (term.opcode != op::JUMP);
            if has_fallthrough && block.insts.end < n {
                if let Some(next_block) = inst_to_block[block.insts.end] {
                    blocks[next_block.index()].preds.push(bid);
                    blocks[bid_idx].succs.push(next_block);
                }
            }

            // Static jump edges.
            if term.is_legacy_static_jump() && !term.flags.contains(InstFlags::INVALID_JUMP) {
                let target_inst = term.data as usize;
                if let Some(target_block) = inst_to_block[target_inst] {
                    blocks[target_block.index()].preds.push(bid);
                    blocks[bid_idx].succs.push(target_block);
                }
            }
        }

        Cfg { blocks, inst_to_block }
    }

    /// Run worklist-based abstract interpretation over the CFG.
    ///
    /// Returns a list of (jump_inst, resolved_target) pairs and the count of resolvable jumps.
    /// Stack snapshots are recorded into `snapshots` during the fixpoint.
    fn run_abstract_interp(
        &self,
        cfg: &Cfg,
        snapshots: &mut [StackSnapshot],
    ) -> (Vec<(usize, JumpTarget)>, usize) {
        let num_blocks = cfg.blocks.len();

        // Initialize block states. Entry block starts with an empty stack.
        let mut block_states: Vec<BlockState> = vec![BlockState::Bottom; num_blocks];
        block_states[0] = BlockState::Known(Vec::new());

        // Collect unresolved jumps.
        let mut jump_insts: Vec<usize> = Vec::new();
        for (i, inst) in self.insts.iter().enumerate() {
            if inst.is_legacy_jump() && !inst.flags.contains(InstFlags::STATIC_JUMP) {
                jump_insts.push(i);
            }
        }

        // Always run the fixpoint to propagate stack states through the CFG.
        // Snapshots are recorded during interpretation — last write wins (= converged state).
        let discovered_edges = self.run_fixpoint(cfg, &mut block_states, snapshots);

        if jump_insts.is_empty() {
            return (Vec::new(), 0);
        }

        // After convergence, resolve each dynamic jump using the final block states.
        // This avoids the problem of the fixpoint accumulating stale partial results.
        let mut jump_targets: Vec<(usize, JumpTarget)> = Vec::new();
        let mut has_top_jump = false;
        for &jump_inst in &jump_insts {
            let target = match cfg.inst_to_block[jump_inst] {
                None => JumpTarget::Bottom,
                Some(bid) => {
                    let bi = bid.index();
                    match &block_states[bi] {
                        BlockState::Bottom => JumpTarget::Bottom,
                        BlockState::Conflict => JumpTarget::Top,
                        BlockState::Known(input) => {
                            match self.jump_operand(&cfg.blocks[bi], input) {
                                Some(AbsValue::Const(val)) => match usize::try_from(val) {
                                    Ok(target_pc) if self.is_valid_jump(target_pc) => {
                                        JumpTarget::Const(self.pc_to_inst(target_pc))
                                    }
                                    _ => JumpTarget::Invalid,
                                },
                                _ => JumpTarget::Top,
                            }
                        }
                    }
                }
            };
            if matches!(target, JumpTarget::Top) {
                has_top_jump = true;
            }
            jump_targets.push((jump_inst, target));
        }

        // Invalidate resolutions that may be unsound due to incomplete analysis.
        //
        // When there are Top (unresolved) or Conflict dynamic jumps, some blocks
        // may have incomplete input states because:
        // 1. A Top jump discovered only a subset of its targets during the fixpoint.
        // 2. A Conflict block suppressed propagation, leaving successor blocks at Bottom
        //    (unreachable) even though they ARE reachable at runtime.
        //
        // A Const/Invalid resolution is invalidated if any block in the transitive
        // predecessor set (following both static CFG edges and discovered dynamic-jump
        // edges backwards) has a Bottom predecessor that starts with a JUMPDEST.
        // Such a predecessor could be reached at runtime by any Top (unresolved)
        // jump, making the resolved jump's input state potentially incomplete.
        if has_top_jump {
            // Build reverse discovered-edge map: for each target block, which source blocks have
            // discovered edges pointing to it.
            let mut disc_preds: Vec<SmallVec<[Block; 2]>> = vec![SmallVec::new(); num_blocks];
            for (src_idx, targets) in discovered_edges.iter().enumerate() {
                let src = Block::new(src_idx);
                for &tgt in targets {
                    disc_preds[tgt.index()].push(src);
                }
            }

            // Precompute: for each block, whether it has a suspect (Bottom + JUMPDEST) predecessor.
            let mut suspect: BitVec = BitVec::repeat(false, num_blocks);
            for bid in 0..num_blocks {
                let has_suspect =
                    cfg.blocks[bid].preds.iter().chain(disc_preds[bid].iter()).any(|&pred| {
                        let pi = pred.index();
                        if !matches!(block_states[pi], BlockState::Bottom) {
                            return false;
                        }
                        self.insts[cfg.blocks[pi].insts.start].is_jumpdest()
                    });
                if has_suspect {
                    suspect.set(bid, true);
                }
            }

            // Propagate suspect flag forward through the CFG: if a block is suspect,
            // all its successors (static + discovered) are also suspect.
            let mut propagate = Worklist::new(num_blocks);
            for bid in 0..num_blocks {
                if suspect[bid] {
                    propagate.push(Block::new(bid));
                }
            }
            while let Some(bid) = propagate.pop() {
                let bi = bid.index();
                for &succ in cfg.blocks[bi].succs.iter().chain(discovered_edges[bi].iter()) {
                    let si = succ.index();
                    if !suspect[si] {
                        suspect.set(si, true);
                        propagate.push(succ);
                    }
                }
            }

            for (inst, target) in jump_targets.iter_mut() {
                if !matches!(target, JumpTarget::Const(_) | JumpTarget::Invalid) {
                    continue;
                }
                if let Some(bid) = cfg.inst_to_block[*inst] {
                    if suspect[bid.index()] {
                        *target = JumpTarget::Top;
                    }
                }
            }
        }

        let count = jump_targets
            .iter()
            .filter(|(_, t)| matches!(t, JumpTarget::Const(_) | JumpTarget::Invalid))
            .count();

        (jump_targets, count)
    }

    /// Run a worklist-based fixpoint to compute abstract block states.
    ///
    /// Returns the discovered dynamic-jump target edges per block.
    /// Stack snapshots are recorded into `snapshots` during each block interpretation.
    fn run_fixpoint(
        &self,
        cfg: &Cfg,
        block_states: &mut [BlockState],
        snapshots: &mut [StackSnapshot],
    ) -> Vec<SmallVec<[Block; 2]>> {
        let num_blocks = cfg.blocks.len();
        let mut worklist = Worklist::new(num_blocks);
        worklist.push(Block::new(0));

        // Persistent set of discovered dynamic-jump target edges per block.
        // Once a dynamic jump in block `bid` resolves to a target block, that
        // edge is kept for all subsequent visits so updated states propagate.
        let mut discovered_jump_edges: Vec<SmallVec<[Block; 2]>> =
            vec![SmallVec::new(); num_blocks];

        let max_iterations = num_blocks * 8;
        let mut iterations = 0;

        while let Some(bid) = worklist.pop() {
            iterations += 1;
            if iterations > max_iterations {
                debug!("block_analysis: iteration limit reached");
                break;
            }

            let bi = bid.index();
            let input = match &block_states[bi] {
                BlockState::Known(s) => s.clone(),
                BlockState::Bottom => continue,
                BlockState::Conflict => {
                    // Propagate Conflict to all successors so they know their
                    // state may be incomplete.
                    for &succ in cfg.blocks[bi].succs.iter().chain(&discovered_jump_edges[bi]) {
                        if block_states[succ.index()].conflict() {
                            worklist.push(succ);
                        }
                    }
                    continue;
                }
            };

            let block = &cfg.blocks[bi];
            if block.dead {
                continue;
            }

            let Some(output) = self.interpret_block(block.insts.clone(), &input, Some(snapshots))
            else {
                continue;
            };

            // For dynamic jumps, discover target edges to propagate state through.
            let term = &self.insts[block.insts.end - 1];
            if term.is_legacy_jump() && !term.flags.contains(InstFlags::STATIC_JUMP) {
                if let Some(AbsValue::Const(val)) = self.jump_operand(block, &input)
                    && let Ok(target_pc) = usize::try_from(val)
                    && self.is_valid_jump(target_pc)
                {
                    let ti = self.pc_to_inst(target_pc);
                    if let Some(tb) = cfg.inst_to_block[ti] {
                        if !discovered_jump_edges[bi].contains(&tb) {
                            discovered_jump_edges[bi].push(tb);
                        }
                    }
                }
            }

            // Propagate to static CFG successors and discovered dynamic-jump targets.
            for &succ in block.succs.iter().chain(&discovered_jump_edges[bi]) {
                if block_states[succ.index()].join(&output) {
                    worklist.push(succ);
                }
            }
        }

        discovered_jump_edges
    }

    /// Interpret a sequence of instructions on the abstract stack.
    /// Returns the abstract stack state after the last instruction, or `None` on conflict.
    ///
    /// If `snapshots` is provided, records the abstract stack state *before* each instruction
    /// into `snapshots[inst_index]`.
    fn interpret_block(
        &self,
        range: Range<usize>,
        input: &[AbsValue],
        mut snapshots: Option<&mut [StackSnapshot]>,
    ) -> Option<Vec<AbsValue>> {
        let mut stack = input.to_vec();

        for i in range {
            let inst = &self.insts[i];
            if inst.is_dead_code() {
                continue;
            }

            // Record pre-instruction snapshot if requested.
            if let Some(snapshots) = &mut snapshots {
                snapshots[i] = StackSnapshot::Known(stack.iter().map(|v| v.as_const()).collect());
            }

            // Instructions marked SKIP_LOGIC (the PUSH in a PUSH+JUMP pair) are no-ops
            // for abstract interpretation — the value is consumed by the already-resolved jump.
            if inst.flags.contains(InstFlags::SKIP_LOGIC) {
                continue;
            }

            match inst.opcode {
                op::PUSH0 => {
                    stack.push(AbsValue::Const(U256::ZERO));
                }
                op::PUSH1..=op::PUSH32 => {
                    let val = self
                        .get_imm(inst)
                        .map_or(AbsValue::Top, |imm| AbsValue::Const(U256::from_be_slice(imm)));
                    stack.push(val);
                }
                op::POP => {
                    stack.pop()?;
                }
                op::DUP1..=op::DUP16 => {
                    let depth = (inst.opcode - op::DUP1 + 1) as usize;
                    if stack.len() < depth {
                        return None;
                    }
                    stack.push(stack[stack.len() - depth]);
                }
                op::SWAP1..=op::SWAP16 => {
                    let depth = (inst.opcode - op::SWAP1 + 1) as usize;
                    let len = stack.len();
                    if len < depth + 1 {
                        return None;
                    }
                    stack.swap(len - 1, len - 1 - depth);
                }
                op::JUMPDEST => {
                    // No stack effect.
                }
                _ => {
                    // For static jumps that were resolved by the simple pass, the jump
                    // already had its input count reduced — use `stack_io()` which accounts
                    // for that.
                    let (inp, out) = inst.stack_io();
                    let inp = inp as usize;
                    let out = out as usize;

                    if stack.len() < inp {
                        return None;
                    }

                    // Try constant folding for common arithmetic.
                    let result = self.try_const_fold(inst.opcode, &stack[stack.len() - inp..]);

                    // Pop inputs.
                    stack.truncate(stack.len() - inp);

                    // Push outputs.
                    if let Some(folded) = result {
                        debug_assert_eq!(out, 1);
                        stack.push(folded);
                    } else {
                        stack.resize(stack.len() + out, AbsValue::Top);
                    }
                }
            }
        }

        Some(stack)
    }

    /// Returns the abstract value of the jump operand (TOS before the terminator) for a block.
    fn jump_operand(&self, block: &BasicBlock, input: &[AbsValue]) -> Option<AbsValue> {
        if block.insts.len() == 1 {
            input.last().copied()
        } else {
            self.interpret_block(block.insts.start..block.insts.end - 1, input, None)?
                .last()
                .copied()
        }
    }

    /// Try to constant-fold a binary/unary operation.
    fn try_const_fold(&self, opcode: u8, inputs: &[AbsValue]) -> Option<AbsValue> {
        match opcode {
            // 1 -> 1
            op::NOT | op::ISZERO => {
                let &[AbsValue::Const(a)] = inputs else {
                    return None;
                };
                let result = match opcode {
                    op::NOT => !a,
                    op::ISZERO => U256::from(a.is_zero()),
                    _ => unreachable!(),
                };
                Some(AbsValue::Const(result))
            }

            // 2 -> 1
            op::ADD
            | op::SUB
            | op::MUL
            | op::AND
            | op::OR
            | op::XOR
            | op::SHL
            | op::SHR
            | op::EQ
            | op::LT
            | op::GT => {
                // b = second popped, a = TOS (first popped).
                let &[AbsValue::Const(b), AbsValue::Const(a)] = inputs else {
                    return None;
                };
                let result = match opcode {
                    op::ADD => a.wrapping_add(b),
                    op::SUB => a.wrapping_sub(b),
                    op::MUL => a.wrapping_mul(b),
                    op::AND => a & b,
                    op::OR => a | b,
                    op::XOR => a ^ b,
                    op::EQ => U256::from(a == b),
                    op::LT => U256::from(a < b),
                    op::GT => U256::from(a > b),
                    op::SHL => {
                        if a >= U256::from(256) {
                            U256::ZERO
                        } else {
                            b.wrapping_shl(a.as_limbs()[0] as usize)
                        }
                    }
                    op::SHR => {
                        if a >= U256::from(256) {
                            U256::ZERO
                        } else {
                            b.wrapping_shr(a.as_limbs()[0] as usize)
                        }
                    }
                    _ => unreachable!(),
                };
                Some(AbsValue::Const(result))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm_primitives::hardfork::SpecId;

    fn analyze_bytecode(hex: &str) -> Bytecode<'static> {
        let code = revm_primitives::hex::decode(hex.trim()).unwrap();
        let code = Box::leak(code.into_boxed_slice());
        let mut bytecode = Bytecode::new(code, SpecId::CANCUN);
        bytecode.analyze().unwrap();
        bytecode
    }

    #[test]
    fn revert_sub_call_storage_oog() {
        let bytecode = analyze_bytecode(
            "60606040526000357c01000000000000000000000000000000000000000000000000000000009004\
             63ffffffff168063b28175c4146046578063c0406226146052575b6000565b34600057605060765\
             65b005b34600057605c6081565b604051808215151515815260200191505060405180910390f35b\
             600c6000819055505b565b600060896076565b600d600181905550600e600281905550600190505\
             b905600a165627a7a723058202a8a75d7d795b5bcb9042fb18b283daa90b999a11ddec892f54873\
             22",
        );
        eprintln!("{bytecode}");
    }

    #[test]
    fn revert_remote_sub_call_storage_oog() {
        let bytecode = analyze_bytecode(
            "608060405234801561001057600080fd5b506004361061002b5760003560e01c806373027f6d14\
             610030575b600080fd5b61004a600480360381019061004591906101a9565b61004c565b005b60\
             00808273ffffffffffffffffffffffffffffffffffffffff1660405160240160405160208183030\
             38152906040527fb28175c4000000000000000000000000000000000000000000000000000000007\
             bffffffffffffffffffffffffffffffffffffffffffffffffffffffff19166020820180517bffff\
             ffffffffffffffffffffffffffffffffffffffffffffffffff838183161783525050505060405161\
             00f69190610247565b6000604051808303816000865af19150503d8060008114610133576040519150\
             601f19603f3d011682016040523d82523d6000602084013e610138565b606091505b50915091508160\
             0155505050565b600080fd5b600073ffffffffffffffffffffffffffffffffffffffff82169050919\
             050565b60006101768261014b565b9050919050565b6101868161016b565b811461019157600080fd5\
             b50565b6000813590506101a38161017d565b92915050565b6000602082840312156101bf576101be\
             610146565b5b60006101cd84828501610194565b91505092915050565b600081519050919050565b6\
             00081905092915050565b60005b8381101561020a5780820151818401526020810190506101ef565b6\
             0008484015250505050565b6000610221826101d6565b61022b81856101e1565b935061023b8185602\
             086016101ec565b80840191505092915050565b60006102538284610216565b91508190509291505056\
             fea2646970667358221220b4673c55c7b0268d7d118059e6509196d2185bb7fe040a7d3900f902c854\
             2ea464736f6c63430008180033",
        );
        // This contract has a single function with one call site. The analysis
        // should NOT incorrectly resolve any jumps (all dynamic jumps should
        // remain dynamic or be correctly resolved).
        eprintln!("{bytecode}");
    }

    #[test]
    fn trans_storage_ok() {
        // Test contracts from transStorageOK state test.
        // These use TLOAD-based return addresses.
        let contracts = [
            "366012575b600b5f6020565b5f5260205ff35b601c6160a75f6024565b6004565b5c90565b5d56",
            "60106001600a5f6012565b015f6016565b005b5c90565b5d56",
            "3033146033575b303303600e57005b601b5f35806001555f608d565b5f80808080305af1600255602e60016089565b600355005b603a5f6089565b8015608757604a600182035f608d565b5f80808080305af1156083576001606191035f608d565b5f80808080305af115608357607f60016078816089565b016001608d565b6006565b5f80fd5b005b5c90565b5d56",
            "60065f601d565b5f5560106001601d565b6001555f80808080335af1005b5c9056",
        ];
        for hex in &contracts {
            let bytecode = analyze_bytecode(hex);
            eprintln!("{bytecode}");
        }
    }

    #[test]
    fn const_operand_basic() {
        // PUSH1 0x42  PUSH1 0x01  ADD  PUSH1 0x00  MSTORE  STOP
        // inst 0: PUSH1 0x42 -> stack: [0x42]
        // inst 1: PUSH1 0x01 -> stack: [0x42, 0x01]
        // inst 2: ADD        -> stack: [0x43]  (const-folded)
        // inst 3: PUSH1 0x00 -> stack: [0x43, 0x00]
        // inst 4: MSTORE     -> pops 2
        // inst 5: STOP
        let bytecode = analyze_bytecode("60426001016000525b00");
        // At inst 2 (ADD), operand 0 (TOS) = 0x01, operand 1 = 0x42.
        assert_eq!(bytecode.const_operand(2, 0), Some(U256::from(0x01)));
        assert_eq!(bytecode.const_operand(2, 1), Some(U256::from(0x42)));
        // At inst 4 (MSTORE), operand 0 (TOS) = 0x00, operand 1 = 0x43 (folded ADD result).
        assert_eq!(bytecode.const_operand(4, 0), Some(U256::from(0x00)));
        assert_eq!(bytecode.const_operand(4, 1), Some(U256::from(0x43)));
    }

    #[test]
    fn const_operand_dynamic() {
        // CALLDATALOAD pushes an unknown value -> const_operand should return None.
        // PUSH1 0x00  CALLDATALOAD  PUSH1 0x00  MSTORE  STOP
        let bytecode = analyze_bytecode("5f355f5200");
        // At inst 2 (PUSH1 0x00 second), operand 0 doesn't exist yet (it's a push).
        // At inst 3 (MSTORE), operand 0 (TOS) = 0x00, operand 1 = unknown (CALLDATALOAD result).
        assert_eq!(bytecode.const_operand(3, 0), Some(U256::ZERO));
        assert_eq!(bytecode.const_operand(3, 1), None);
    }

    #[test]
    fn hash_10k() {
        let code =
            revm_primitives::hex::decode(include_str!("../../../../data/hash_10k.rt.hex").trim())
                .unwrap();
        let code = Box::leak(code.into_boxed_slice());
        let mut bytecode = Bytecode::new(code, SpecId::CANCUN);
        bytecode.analyze().unwrap();
        eprintln!("{bytecode}");
        assert!(!bytecode.has_dynamic_jumps, "expected all jumps to be resolved");
    }
}

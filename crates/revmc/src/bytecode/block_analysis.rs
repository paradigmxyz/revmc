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
use revm_bytecode::opcode as op;
use revm_primitives::U256;
use smallvec::SmallVec;
use std::collections::VecDeque;

/// Abstract value on the stack.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// Lattice join: two values merge to their least upper bound.
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) if a == b => Self::Const(*a),
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
                for (slot, inc) in existing.iter_mut().zip(incoming.iter()) {
                    let joined = slot.join(inc);
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

type BlockId = usize;

/// FIFO worklist with deduplication.
struct Worklist {
    queue: VecDeque<BlockId>,
    in_queue: Vec<bool>,
}

impl Worklist {
    fn new(size: usize) -> Self {
        Self { queue: VecDeque::new(), in_queue: vec![false; size] }
    }

    fn with(size: usize, initial: BlockId) -> Self {
        let mut w = Self::new(size);
        w.push(initial);
        w
    }

    fn pop(&mut self) -> Option<BlockId> {
        let id = self.queue.pop_front()?;
        self.in_queue[id] = false;
        Some(id)
    }

    fn push(&mut self, id: BlockId) {
        if !self.in_queue[id] {
            self.in_queue[id] = true;
            self.queue.push_back(id);
        }
    }
}

/// A basic block in the CFG.
struct BasicBlock {
    /// First instruction index (inclusive).
    start: usize,
    /// Last instruction index (inclusive, the terminator).
    end: usize,
    /// Predecessor block IDs.
    preds: Vec<BlockId>,
    /// Successor block IDs.
    succs: Vec<BlockId>,
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
    /// Maps instruction index to block ID.
    inst_to_block: Vec<BlockId>,
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
        let mut is_leader = vec![false; n];
        is_leader[0] = true;

        for (i, inst) in self.insts.iter().enumerate() {
            if inst.is_dead_code() {
                continue;
            }
            if inst.is_reachable_jumpdest(self.has_dynamic_jumps) {
                is_leader[i] = true;
            }
            if inst.is_branching() || inst.is_diverging() {
                // Next instruction is a leader (if it exists).
                if i + 1 < n {
                    is_leader[i + 1] = true;
                }
            }
        }

        // Build blocks.
        let mut blocks = Vec::new();
        let mut inst_to_block = vec![0usize; n];
        let mut current_start = None;

        for i in 0..n {
            if self.insts[i].is_dead_code() {
                inst_to_block[i] = usize::MAX;
                continue;
            }

            if is_leader[i] || current_start.is_none() {
                if let Some(start) = current_start {
                    // End the previous block at i-1.
                    let end = i - 1;
                    let block_id = blocks.len();
                    for (j, ib) in inst_to_block.iter_mut().enumerate().take(end + 1).skip(start) {
                        if !self.insts[j].is_dead_code() {
                            *ib = block_id;
                        }
                    }
                    blocks.push(BasicBlock { start, end, preds: Vec::new(), succs: Vec::new() });
                }
                current_start = Some(i);
            }
        }

        // Close the last block.
        if let Some(start) = current_start {
            let end = n - 1;
            let block_id = blocks.len();
            for (j, ib) in inst_to_block.iter_mut().enumerate().take(end + 1).skip(start) {
                if !self.insts[j].is_dead_code() {
                    *ib = block_id;
                }
            }
            blocks.push(BasicBlock { start, end, preds: Vec::new(), succs: Vec::new() });
        }

        // Build edges based on known control flow.
        let num_blocks = blocks.len();
        for bid in 0..num_blocks {
            let end = blocks[bid].end;
            let term = &self.insts[end];

            // Fallthrough edge: if the terminator doesn't unconditionally branch/diverge.
            let has_fallthrough = !term.is_diverging() && (term.opcode != op::JUMP);
            if has_fallthrough && end + 1 < n {
                let next_block = inst_to_block[end + 1];
                if next_block != usize::MAX && next_block < num_blocks {
                    blocks[bid].succs.push(next_block);
                    blocks[next_block].preds.push(bid);
                }
            }

            // Static jump edges.
            if term.is_legacy_static_jump() && !term.flags.contains(InstFlags::INVALID_JUMP) {
                let target_inst = term.data as usize;
                let target_block = inst_to_block[target_inst];
                if target_block != usize::MAX && target_block < num_blocks {
                    blocks[bid].succs.push(target_block);
                    blocks[target_block].preds.push(bid);
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
        let discovered_edges =
            self.run_fixpoint(cfg, &mut block_states, &vec![false; num_blocks], snapshots);

        if jump_insts.is_empty() {
            return (Vec::new(), 0);
        }

        // Build reverse discovered-edge map: for each target block, which source blocks have
        // discovered edges pointing to it.
        let mut disc_preds: Vec<Vec<BlockId>> = vec![Vec::new(); num_blocks];
        for (src, targets) in discovered_edges.iter().enumerate() {
            for &tgt in targets {
                disc_preds[tgt].push(src);
            }
        }

        // After convergence, resolve each dynamic jump using the final block states.
        // This avoids the problem of the fixpoint accumulating stale partial results.
        let mut jump_targets: Vec<(usize, JumpTarget)> = Vec::new();
        let mut has_top_jump = false;
        for &jump_inst in &jump_insts {
            let bid = cfg.inst_to_block[jump_inst];
            let target = if bid >= num_blocks || bid == usize::MAX {
                JumpTarget::Bottom
            } else {
                match &block_states[bid] {
                    BlockState::Bottom => JumpTarget::Bottom,
                    BlockState::Conflict => JumpTarget::Top,
                    BlockState::Known(input) => {
                        let block = &cfg.blocks[bid];
                        // Interpret up to (but not including) the terminator to get the
                        // pre-jump stack state.
                        let pre_jump_stack = if block.start == block.end {
                            input.clone()
                        } else {
                            self.interpret_block(block.start, block.end - 1, input, None)
                                .unwrap_or_default()
                        };
                        let target_val = pre_jump_stack.last().cloned();
                        match target_val {
                            Some(AbsValue::Const(val)) => match usize::try_from(val) {
                                Ok(target_pc) if self.is_valid_jump(target_pc) => {
                                    JumpTarget::Const(self.pc_to_inst(target_pc))
                                }
                                _ => JumpTarget::Invalid,
                            },
                            Some(AbsValue::Top) => JumpTarget::Top,
                            None => JumpTarget::Top,
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
            // Precompute: for each block, whether it has a suspect (Bottom + JUMPDEST) predecessor.
            let mut suspect = vec![false; num_blocks];
            for bid in 0..num_blocks {
                // Check static predecessors.
                let has_suspect =
                    cfg.blocks[bid].preds.iter().chain(disc_preds[bid].iter()).any(|&pred| {
                        if !matches!(block_states[pred], BlockState::Bottom) {
                            return false;
                        }
                        let pred_start = cfg.blocks[pred].start;
                        self.insts[pred_start].is_jumpdest()
                    });
                if has_suspect {
                    suspect[bid] = true;
                }
            }

            // Propagate suspect flag forward through the CFG: if a block is suspect,
            // all its successors (static + discovered) are also suspect.
            let mut propagate = Worklist::new(num_blocks);
            for (bid, is_suspect) in suspect.iter().enumerate() {
                if *is_suspect {
                    propagate.push(bid);
                }
            }
            while let Some(bid) = propagate.pop() {
                for &succ in cfg.blocks[bid].succs.iter().chain(discovered_edges[bid].iter()) {
                    if !suspect[succ] {
                        suspect[succ] = true;
                        propagate.push(succ);
                    }
                }
            }

            for (inst, target) in jump_targets.iter_mut() {
                if !matches!(target, JumpTarget::Const(_) | JumpTarget::Invalid) {
                    continue;
                }
                let bid = cfg.inst_to_block[*inst];
                if bid >= num_blocks || bid == usize::MAX {
                    continue;
                }
                if suspect[bid] {
                    *target = JumpTarget::Top;
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
        widen: &[bool],
        snapshots: &mut [StackSnapshot],
    ) -> Vec<Vec<BlockId>> {
        let num_blocks = cfg.blocks.len();
        let mut worklist = Worklist::with(num_blocks, 0);

        // Persistent set of discovered dynamic-jump target edges per block.
        // Once a dynamic jump in block `bid` resolves to a target block, that
        // edge is kept for all subsequent visits so updated states propagate.
        let mut discovered_jump_edges: Vec<Vec<BlockId>> = vec![Vec::new(); num_blocks];

        let max_iterations = num_blocks * 8;
        let mut iterations = 0;

        while let Some(bid) = worklist.pop() {
            iterations += 1;
            if iterations > max_iterations {
                debug!("block_analysis: iteration limit reached");
                break;
            }

            let input = match &block_states[bid] {
                BlockState::Known(s) => s.clone(),
                BlockState::Bottom => continue,
                BlockState::Conflict => {
                    // Propagate Conflict to all successors so they know their
                    // state may be incomplete.
                    let block = &cfg.blocks[bid];
                    for &succ in &block.succs {
                        if block_states[succ].conflict() {
                            worklist.push(succ);
                        }
                    }
                    for &succ in &discovered_jump_edges[bid] {
                        if block_states[succ].conflict() {
                            worklist.push(succ);
                        }
                    }
                    continue;
                }
            };

            let block = &cfg.blocks[bid];

            // For JUMPDEST blocks that need widening (potential dynamic jump
            // table targets), widen incoming values to Top. This preserves
            // stack height but makes all inherited values unknown.
            let input =
                if widen[bid] { input.iter().map(|_| AbsValue::Top).collect() } else { input };

            let output = self.interpret_block(block.start, block.end, &input, Some(snapshots));
            let output = match output {
                Some(s) => s,
                None => continue,
            };

            // For dynamic jumps, discover target edges to propagate state through.
            let term = &self.insts[block.end];
            if term.is_legacy_jump() && !term.flags.contains(InstFlags::STATIC_JUMP) {
                let pre_jump_stack = if block.start == block.end {
                    &input
                } else {
                    &self
                        .interpret_block(block.start, block.end - 1, &input, None)
                        .unwrap_or_default()
                };

                if let Some(AbsValue::Const(val)) = pre_jump_stack.last()
                    && let Ok(target_pc) = usize::try_from(*val)
                    && self.is_valid_jump(target_pc)
                {
                    let ti = self.pc_to_inst(target_pc);
                    let tb = cfg.inst_to_block[ti];
                    if tb != usize::MAX
                        && tb < num_blocks
                        && !discovered_jump_edges[bid].contains(&tb)
                    {
                        discovered_jump_edges[bid].push(tb);
                    }
                }
            }

            // Propagate to static CFG successors and discovered dynamic-jump targets.
            for &succ in block.succs.iter().chain(&discovered_jump_edges[bid]) {
                if block_states[succ].join(&output) {
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
        start: usize,
        end: usize,
        input: &[AbsValue],
        mut snapshots: Option<&mut [StackSnapshot]>,
    ) -> Option<Vec<AbsValue>> {
        let mut stack = input.to_vec();

        for i in start..=end {
            let inst = &self.insts[i];
            if inst.is_dead_code() {
                continue;
            }

            // Record pre-instruction snapshot if requested.
            if let Some(ref mut snapshots) = snapshots {
                snapshots[i] = StackSnapshot::Known(
                    stack
                        .iter()
                        .map(|v| match v {
                            AbsValue::Const(c) => Some(*c),
                            AbsValue::Top => None,
                        })
                        .collect(),
                );
            }

            // Instructions marked SKIP_LOGIC (the PUSH in a PUSH+JUMP pair) are no-ops
            // for abstract interpretation — the value is consumed by the already-resolved jump.
            if inst.flags.contains(InstFlags::SKIP_LOGIC) {
                continue;
            }

            let opcode = inst.opcode;

            match opcode {
                op::PUSH0 => {
                    stack.push(AbsValue::Const(U256::ZERO));
                }
                op::PUSH1..=op::PUSH32 => {
                    let val = self.get_imm(inst).map_or(AbsValue::Top, |imm| {
                        let mut buf = [0u8; 32];
                        buf[32 - imm.len()..].copy_from_slice(imm);
                        AbsValue::Const(U256::from_be_bytes(buf))
                    });
                    stack.push(val);
                }
                op::POP => {
                    stack.pop()?;
                }
                op::DUP1..=op::DUP16 => {
                    let depth = (opcode - op::DUP1 + 1) as usize;
                    if stack.len() < depth {
                        return None;
                    }
                    let val = stack[stack.len() - depth].clone();
                    stack.push(val);
                }
                op::SWAP1..=op::SWAP16 => {
                    let depth = (opcode - op::SWAP1 + 1) as usize;
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
                    let result = self.try_const_fold(opcode, &stack[stack.len() - inp..]);

                    // Pop inputs.
                    stack.truncate(stack.len() - inp);

                    // Push outputs.
                    if let Some(folded) = result {
                        debug_assert_eq!(out, 1);
                        stack.push(folded);
                    } else {
                        for _ in 0..out {
                            stack.push(AbsValue::Top);
                        }
                    }
                }
            }
        }

        Some(stack)
    }

    /// Try to constant-fold a binary/unary operation.
    fn try_const_fold(&self, opcode: u8, inputs: &[AbsValue]) -> Option<AbsValue> {
        // Only fold if all inputs are constants.
        match opcode {
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
                if inputs.len() < 2 {
                    return None;
                }
                let (AbsValue::Const(a), AbsValue::Const(b)) = (&inputs[0], &inputs[1]) else {
                    return None;
                };
                // EVM stack: a is deeper, b is on top.
                // Actually the top of stack is the last element in our vec representation.
                // inputs are the last N elements: inputs[0] is deepest, inputs[len-1] is top.
                // But for EVM ops, first popped = top of stack.
                // In our slice: inputs[inputs.len()-1] is TOS, inputs[inputs.len()-2] is second.
                // For a 2-input op: a = inputs[0] (second popped, deeper), b = inputs[1] (first
                // popped, TOS). EVM spec: ADD pops a (top), b (second), pushes a+b.
                // So: inputs[1] = a (TOS, first popped), inputs[0] = b (second popped).
                let (a, b) = (b, a); // a = TOS, b = second
                let result = match opcode {
                    op::ADD => a.wrapping_add(*b),
                    op::SUB => a.wrapping_sub(*b),
                    op::MUL => a.wrapping_mul(*b),
                    op::AND => *a & *b,
                    op::OR => *a | *b,
                    op::XOR => *a ^ *b,
                    op::EQ => {
                        if a == b {
                            U256::from(1)
                        } else {
                            U256::ZERO
                        }
                    }
                    op::LT => {
                        if a < b {
                            U256::from(1)
                        } else {
                            U256::ZERO
                        }
                    }
                    op::GT => {
                        if a > b {
                            U256::from(1)
                        } else {
                            U256::ZERO
                        }
                    }
                    op::SHL => {
                        if *a >= U256::from(256) {
                            U256::ZERO
                        } else {
                            b.wrapping_shl(a.as_limbs()[0] as usize)
                        }
                    }
                    op::SHR => {
                        if *a >= U256::from(256) {
                            U256::ZERO
                        } else {
                            b.wrapping_shr(a.as_limbs()[0] as usize)
                        }
                    }
                    _ => unreachable!(),
                };
                Some(AbsValue::Const(result))
            }
            op::NOT | op::ISZERO => {
                if inputs.is_empty() {
                    return None;
                }
                let AbsValue::Const(a) = &inputs[inputs.len() - 1] else {
                    return None;
                };
                let result = match opcode {
                    op::NOT => !*a,
                    op::ISZERO => {
                        if a.is_zero() {
                            U256::from(1)
                        } else {
                            U256::ZERO
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
    use crate::bytecode::Bytecode;
    use revm_primitives::{U256, hardfork::SpecId};

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
        // inst 2: ADD         -> stack: [0x43]  (const-folded)
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

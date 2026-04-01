//! Abstract stack interpretation for resolving dynamic jump targets and constant propagation.
//!
//! This pass builds a basic-block CFG and runs worklist-based abstract interpretation to a
//! fixpoint, propagating abstract stack states across blocks. It resolves jump targets that the
//! block-local pass `block_analysis_local` (which interprets each block independently) cannot
//! handle — including internal function return patterns where multiple callers push different
//! return addresses.
//!
//! After the fixpoint converges, the known-constant operands at each instruction are persisted
//! as [`OperandSnapshot`]s so that later passes (e.g. code generation) can query the
//! known-constant value of stack operands.
//!
//! ## Abstract domain
//!
//! Per-value lattice (ascending order):
//! - `Const(U256Idx)` — a single known constant.
//! - `ConstSet(ConstSetIdx)` — multiple known constants (interned, sorted, deduplicated).
//! - `Top` — reachable but unknown.
//!
//! Per-block lattice:
//! - `Bottom` — block not yet reached.
//! - `Known(Vec<AbsValue>)` — reached with a known stack state (top-aligned; when predecessors have
//!   different stack heights, the shorter stack is bottom-padded with `Top`).
//!
//! ## Soundness
//!
//! When unresolved `Top` jumps remain after the fixpoint, a transitive predecessor analysis
//! invalidates suspect resolutions that may be reachable from those unresolved jumps, ensuring
//! that only sound jump targets are reported as resolved.

use super::StackSection;
use crate::bytecode::{Bytecode, Inst, InstFlags, Interner, U256Idx};
use bitvec::vec::BitVec;
use either::Either;
use oxc_index::IndexVec;
use revm_bytecode::opcode as op;
use revm_primitives::U256;
use smallvec::SmallVec;
use std::{cmp::Ordering, collections::VecDeque, ops::Range};

oxc_index::define_nonmax_u32_index_type! {
    /// Index into the interned constant-set pool.
    pub(crate) struct ConstSetIdx;
}
crate::bytecode::impl_index_display!(ConstSetIdx, "{}");

/// Per-instruction snapshot of abstract operand values.
///
/// Stored in stack order: index 0 is the deepest operand, last element is TOS.
/// Only the instruction's inputs are stored, not the entire stack.
pub(crate) type OperandSnapshot = SmallVec<[AbsValue; 4]>;

/// Bundles input and output snapshots for recording during abstract interpretation.
#[derive(Default)]
pub(crate) struct Snapshots {
    /// Pre-instruction input operand snapshots.
    pub(crate) inputs: IndexVec<Inst, OperandSnapshot>,
    /// Post-instruction output snapshot (single value per instruction).
    pub(crate) outputs: IndexVec<Inst, Option<AbsValue>>,
}

/// Abstract value on the stack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum AbsValue {
    Const(U256Idx),
    /// Multiple known constant values (interned, sorted, deduplicated).
    ConstSet(ConstSetIdx),
    /// Top value (unknown concrete value).
    #[default]
    Top,
}

impl AbsValue {
    /// Returns the single constant if this is `Const`, or `None` otherwise.
    pub(crate) fn as_const(self) -> Option<U256Idx> {
        match self {
            Self::Const(v) => Some(v),
            _ => None,
        }
    }
}

/// Constant-set interner used by the abstract interpreter.
struct ConstSetInterner {
    interner: Interner<ConstSetIdx, Box<[U256Idx]>>,
}

impl ConstSetInterner {
    fn new() -> Self {
        Self { interner: Interner::new() }
    }

    /// Returns the constants in the given set.
    fn get(&self, idx: ConstSetIdx) -> &[U256Idx] {
        self.interner.get(idx)
    }

    /// Returns the set of known constants for an abstract value, or `None` if `Top`.
    fn abs_const_set<'a>(&'a self, val: &'a AbsValue) -> Option<&'a [U256Idx]> {
        match val {
            AbsValue::Top => None,
            AbsValue::Const(v) => Some(std::slice::from_ref(v)),
            AbsValue::ConstSet(idx) => Some(self.get(*idx)),
        }
    }

    /// Interns a sorted, deduplicated set and returns the corresponding `AbsValue`.
    fn intern_set(&mut self, set: &[U256Idx]) -> AbsValue {
        match set.len() {
            0 => unreachable!("empty const set"),
            1 => AbsValue::Const(set[0]),
            _ => AbsValue::ConstSet(self.interner.intern(set.into())),
        }
    }

    /// Lattice join: two values merge to their least upper bound.
    fn join(&mut self, a: AbsValue, b: AbsValue) -> AbsValue {
        match (a, b) {
            (AbsValue::Top, _) | (_, AbsValue::Top) => AbsValue::Top,
            (AbsValue::Const(x), AbsValue::Const(y)) if x == y => AbsValue::Const(x),
            _ => {
                let a = self.abs_const_set(&a).unwrap();
                let b = self.abs_const_set(&b).unwrap();
                // Sorted merge with dedup.
                let mut merged = SmallVec::<[U256Idx; 8]>::new();
                let (mut i, mut j) = (0, 0);
                while i < a.len() && j < b.len() {
                    match a[i].index().cmp(&b[j].index()) {
                        Ordering::Less => {
                            merged.push(a[i]);
                            i += 1;
                        }
                        Ordering::Greater => {
                            merged.push(b[j]);
                            j += 1;
                        }
                        Ordering::Equal => {
                            merged.push(a[i]);
                            i += 1;
                            j += 1;
                        }
                    }
                }
                merged.extend_from_slice(&a[i..]);
                merged.extend_from_slice(&b[j..]);
                self.intern_set(&merged)
            }
        }
    }
}

/// Maximum abstract stack depth tracked by the analysis. Top-aligned joins across paths with
/// differing stack heights pad the shorter stack with `Top` at the bottom; on CFG cycles this
/// padding can grow without bound. Clamping discards only those bottom `Top` entries, so no
/// precision is lost.
const MAX_ABS_STACK_DEPTH: usize = 64;

/// Abstract state at the entry of a block.
#[derive(Clone, Debug)]
enum BlockState {
    /// Block has not been reached yet.
    Bottom,
    /// Block has been reached with a known stack state (top-aligned).
    Known(Vec<AbsValue>),
}

impl BlockState {
    /// Join another incoming state into this one. Returns `true` if the state changed.
    ///
    /// When stack heights differ, the stacks are top-aligned and the shorter one is
    /// bottom-padded with `Top`. This handles the common Solidity dispatch pattern where
    /// the "no selector match" fallthrough leaves an extra item on the stack that the
    /// fallback ignores.
    fn join(&mut self, incoming: &[AbsValue], sets: &mut ConstSetInterner) -> bool {
        match self {
            Self::Bottom => {
                let start = incoming.len().saturating_sub(MAX_ABS_STACK_DEPTH);
                *self = Self::Known(incoming[start..].to_vec());
                true
            }
            Self::Known(existing) => {
                let new_len = existing.len().max(incoming.len());
                let mut changed = false;

                // Clamp to MAX_ABS_STACK_DEPTH by only joining the top portion;
                // elements below that are unreachable by any EVM instruction (max
                // depth is DUP16/SWAP16 = 16) and discarding them preserves soundness.
                let join_len = new_len.min(MAX_ABS_STACK_DEPTH);

                // Resize existing to join_len: pad at bottom with Top or truncate.
                if existing.len() < join_len {
                    let pad = join_len - existing.len();
                    existing.splice(0..0, std::iter::repeat_n(AbsValue::Top, pad));
                    changed = true;
                } else if existing.len() > join_len {
                    existing.drain(..existing.len() - join_len);
                    changed = true;
                }

                // Join element-wise, top-aligned. Both stacks have their top at the end.
                // `incoming` may be longer than join_len — we only look at its top portion.
                let incoming_start = incoming.len().saturating_sub(join_len);
                let incoming_top = &incoming[incoming_start..];
                // incoming_top.len() <= join_len. If shorter, bottom positions get Top.
                let pad = join_len - incoming_top.len();
                for i in 0..join_len {
                    let inc = if i < pad { AbsValue::Top } else { incoming_top[i - pad] };
                    let joined = sets.join(existing[i], inc);
                    if joined != existing[i] {
                        existing[i] = joined;
                        changed = true;
                    }
                }

                changed
            }
        }
    }
}

oxc_index::define_nonmax_u32_index_type! {
    /// A block index in the CFG.
    pub(crate) struct Block;
}
crate::bytecode::impl_index_display!(Block, "bb{}");

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
pub(crate) struct BlockData {
    /// Instruction index range (exclusive end). All instructions in this range are live.
    pub(crate) insts: Range<Inst>,
    /// Predecessor block IDs.
    pub(crate) preds: SmallVec<[Block; 4]>,
    /// Successor block IDs.
    pub(crate) succs: SmallVec<[Block; 4]>,
}

impl BlockData {
    #[inline]
    pub(crate) fn terminator(&self) -> Inst {
        self.insts.end - 1
    }

    /// Returns the instruction range as `Range<usize>` for indexing into raw arrays.
    #[inline]
    pub(crate) fn insts(&self) -> impl ExactSizeIterator<Item = Inst> + use<> {
        (self.insts.start.index()..self.insts.end.index()).map(Inst::from_usize)
    }
}

/// Resolved jump target after fixpoint.
#[derive(Clone, Debug)]
enum JumpTarget {
    /// Not yet observed.
    Bottom,
    /// Known constant target instruction index.
    Const(Inst),
    /// Multiple known constant target instruction indices.
    Multi(SmallVec<[Inst; 4]>),
    /// Known constant but invalid target.
    Invalid,
    /// Unknown target.
    Top,
}

/// CFG for abstract interpretation.
#[derive(Default)]
pub(crate) struct Cfg {
    pub(crate) blocks: IndexVec<Block, BlockData>,
    /// Maps instruction index to block ID. `None` for dead-code instructions.
    pub(crate) inst_to_block: IndexVec<Inst, Option<Block>>,
}

impl Bytecode<'_> {
    /// Ensures `self.snapshots` is sized for the current instruction count.
    fn init_snapshots(&mut self) {
        let n = self.insts.len();
        self.snapshots.inputs.resize(n, SmallVec::new());
        self.snapshots.outputs.resize(n, None);
    }

    /// Block-local jump resolution: interpret each block independently to discover
    /// jumps resolvable from block-local computation alone.
    ///
    /// Also initializes `self.snapshots`.
    #[instrument(name = "local_jumps", level = "debug", skip_all)]
    pub(crate) fn block_analysis_local(&mut self) {
        if self.cfg.blocks.is_empty() {
            return;
        }

        self.init_snapshots();

        let mut newly_resolved = 0u32;
        let mut stack = Vec::new();

        for bid in self.cfg.blocks.indices() {
            let block = &self.cfg.blocks[bid];

            // Compute required entry depth for this block using stack section analysis.
            let section =
                StackSection::from_stack_io(block.insts().map(|i| self.insts[i].stack_io_raw()));

            // Interpret the block with `Top` as inputs.
            stack.clear();
            stack.resize(section.inputs as usize, AbsValue::Top);
            if !self.interpret_block(block.insts(), &mut stack) {
                continue;
            }

            // Check if the block's terminator is an unresolved jump.
            // We interpret every block to initialize `snapshots`, so we check this after.
            let block = &self.cfg.blocks[bid];
            let term_inst = block.terminator();
            let term = &self.insts[term_inst];
            if !term.is_jump() || term.flags.contains(InstFlags::STATIC_JUMP) {
                continue;
            }

            // Check if the jump target resolved to a single constant.
            let Some(val) = self.const_operand(term_inst, 0) else {
                continue;
            };
            let Ok(target_pc) = usize::try_from(val) else {
                // Target too large — invalid jump.
                self.insts[term_inst].flags |= InstFlags::STATIC_JUMP | InstFlags::INVALID_JUMP;
                newly_resolved += 1;
                continue;
            };

            if !self.is_valid_jump(target_pc) {
                self.insts[term_inst].flags |= InstFlags::STATIC_JUMP | InstFlags::INVALID_JUMP;
                newly_resolved += 1;
                trace!(%term_inst, target_pc, "local: invalid jump target");
                continue;
            }

            let target = self.pc_to_inst(target_pc);

            // Check if this is a simple adjacent PUSH+JUMP pattern.
            // If so, use the SKIP_LOGIC optimization for backward compatibility with codegen.
            let is_adjacent_push_jump = term_inst.index() > 0 && {
                let push_inst = Inst::from_usize(term_inst.index() - 1);
                let push = &self.insts[push_inst];
                push.is_push() && !push.is_dead_code() && block.insts.contains(&push_inst)
            };

            if is_adjacent_push_jump {
                let push_inst = Inst::from_usize(term_inst.index() - 1);
                self.insts[push_inst].flags |= InstFlags::SKIP_LOGIC;
                self.insts[term_inst].flags |= InstFlags::STATIC_JUMP;
            } else {
                self.insts[term_inst].flags |=
                    InstFlags::STATIC_JUMP | InstFlags::BLOCK_RESOLVED_JUMP;
            }

            // Set target and mark JUMPDEST as reachable.
            self.insts[term_inst].data = target.index() as u32;
            self.insts[target].data = 1;
            newly_resolved += 1;
            trace!(%term_inst, %target, "local: resolved jump");
        }

        debug!(newly_resolved, "local jump resolution");
        self.recompute_has_dynamic_jumps();
    }

    /// Runs abstract stack interpretation to resolve additional jump targets.
    ///
    /// Also computes and stores per-instruction stack snapshots for constant propagation.
    #[instrument(name = "ba", level = "debug", skip_all)]
    pub(crate) fn block_analysis(&mut self) {
        if self.cfg.blocks.is_empty() {
            return;
        }

        self.init_snapshots();
        let (resolved, count) = self.run_abstract_interp();

        if count > 0 {
            let newly_resolved = self.commit_resolved_jumps(&resolved);
            debug!(newly_resolved, "resolved jumps");
        }

        self.recompute_has_dynamic_jumps();
    }

    /// Recomputes the `has_dynamic_jumps` flag based on the current instruction set.
    pub(crate) fn recompute_has_dynamic_jumps(&mut self) {
        let mut unresolved = self.insts.iter().filter(|inst| {
            inst.is_jump() && !inst.flags.contains(InstFlags::STATIC_JUMP) && !inst.is_dead_code()
        });
        self.has_dynamic_jumps = unresolved.next().is_some();
        if self.has_dynamic_jumps {
            debug!(n = 1 + unresolved.count(), "unresolved dynamic jumps remain");
        }
    }

    /// Commits resolved jump targets by setting flags and data on the corresponding instructions.
    ///
    /// Returns the number of newly resolved jumps.
    fn commit_resolved_jumps(&mut self, resolved: &[(Inst, JumpTarget)]) -> u32 {
        let has_top_jump = resolved.iter().any(|(_, t)| matches!(t, JumpTarget::Top));

        let mut newly_resolved = 0u32;
        for &(jump_inst, ref target) in resolved {
            // Skip if already resolved by block_analysis_local.
            if self.insts[jump_inst].flags.contains(InstFlags::STATIC_JUMP) {
                continue;
            }

            match *target {
                JumpTarget::Const(target_inst) => {
                    debug_assert_eq!(
                        self.insts[target_inst].opcode,
                        op::JUMPDEST,
                        "block_analysis resolved to non-JUMPDEST"
                    );
                    self.insts[jump_inst].flags |=
                        InstFlags::STATIC_JUMP | InstFlags::BLOCK_RESOLVED_JUMP;
                    self.insts[jump_inst].data = target_inst.index() as u32;
                    // Mark JUMPDEST as reachable.
                    self.insts[target_inst].data = 1;
                    newly_resolved += 1;
                    trace!(%jump_inst, %target_inst, "resolved jump");
                }
                JumpTarget::Multi(ref targets) => {
                    for &target_inst in targets {
                        debug_assert_eq!(
                            self.insts[target_inst].opcode,
                            op::JUMPDEST,
                            "block_analysis multi-resolved to non-JUMPDEST"
                        );
                        self.insts[target_inst].data = 1;
                    }
                    self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP
                        | InstFlags::BLOCK_RESOLVED_JUMP
                        | InstFlags::MULTI_JUMP;
                    self.multi_jump_targets.insert(jump_inst, targets.clone());
                    newly_resolved += 1;
                    trace!(%jump_inst, n_targets = targets.len(), "resolved multi-target jump");
                }
                JumpTarget::Invalid => {
                    self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP
                        | InstFlags::INVALID_JUMP
                        | InstFlags::BLOCK_RESOLVED_JUMP;
                    newly_resolved += 1;
                    trace!(%jump_inst, "resolved invalid jump");
                }
                JumpTarget::Bottom if !has_top_jump => {
                    // Truly unreachable: no unresolved jumps remain, so this
                    // code cannot be reached at runtime. Mark as invalid.
                    self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP | InstFlags::INVALID_JUMP;
                    trace!(%jump_inst, "unreachable jump");
                }
                JumpTarget::Bottom => {
                    // Unreachable according to the analysis, but there are
                    // unresolved (Top) jumps that might reach this code at
                    // runtime. Leave as-is.
                    trace!(%jump_inst, "unreachable jump (not marking, has_top_jump)");
                }
                JumpTarget::Top => {
                    trace!(%jump_inst, "unresolved jump (Top)");
                }
            }
        }
        newly_resolved
    }

    /// Rebuild the basic-block CFG from the current instruction state.
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn rebuild_cfg(&mut self) {
        let finish_block = |cfg: &mut Cfg, start: usize, end: usize| {
            debug_assert!(start < end, "empty block range: {start}..{end}");
            let bid = cfg.blocks.push(BlockData {
                insts: Inst::from_usize(start)..Inst::from_usize(end),
                preds: SmallVec::new(),
                succs: SmallVec::new(),
            });
            for i in start..end {
                cfg.inst_to_block[Inst::from_usize(i)] = Some(bid);
            }
        };

        let n = self.insts.len();

        let cfg = &mut self.cfg;
        cfg.blocks.clear();
        cfg.inst_to_block.clear();
        if n == 0 {
            return;
        }

        // Identify block leaders.
        let mut is_leader: BitVec = BitVec::repeat(false, n);
        is_leader.set(0, true);

        for (i, inst) in self.insts.raw.iter().enumerate() {
            if inst.is_dead_code() {
                continue;
            }
            if inst.is_reachable_jumpdest(self.has_dynamic_jumps) {
                is_leader.set(i, true);
            }
            if (inst.is_branching() || inst.is_diverging()) && i + 1 < n {
                is_leader.set(i + 1, true);
            }
        }

        // Build blocks. Dead instructions are skipped; `current_end` tracks the
        // exclusive end of live instructions so block ranges never include dead code.
        cfg.inst_to_block.resize(n, None);
        let mut current_start = None;
        let mut current_end = 0;

        for i in 0..n {
            if self.insts.raw[i].is_dead_code() {
                continue;
            }

            if is_leader[i] || current_start.is_none() {
                if let Some(start) = current_start {
                    finish_block(cfg, start, current_end);
                }
                current_start = Some(i);
            }
            current_end = i + 1;
        }

        // Close the last block.
        if let Some(start) = current_start {
            finish_block(cfg, start, current_end);
        }

        // Build edges based on known control flow.
        for bid in cfg.blocks.indices() {
            let term = &self.insts[cfg.blocks[bid].terminator()];

            // Resolve a target instruction through redirects to a CFG block.
            let resolve = |target: Inst| -> Option<Block> {
                let target = self.redirects.get(&target).copied().unwrap_or(target);
                cfg.inst_to_block.get(target).copied().flatten()
            };

            // Fallthrough edge.
            if term.can_fall_through()
                && let Some(target_block) = resolve(cfg.blocks[bid].terminator() + 1)
            {
                cfg.blocks[target_block].preds.push(bid);
                cfg.blocks[bid].succs.push(target_block);
            }

            // Jump edges: static single-target, or multi-jump.
            let term_inst = cfg.blocks[bid].terminator();
            if term.flags.contains(InstFlags::MULTI_JUMP)
                && let Some(targets) = self.multi_jump_targets.get(&term_inst)
            {
                for &t in targets {
                    if let Some(target_block) = resolve(t)
                        && !cfg.blocks[bid].succs.contains(&target_block)
                    {
                        cfg.blocks[target_block].preds.push(bid);
                        cfg.blocks[bid].succs.push(target_block);
                    }
                }
            } else if term.is_static_jump()
                && !term.flags.contains(InstFlags::INVALID_JUMP)
                && let Some(target_block) = resolve(Inst::from_usize(term.data as usize))
            {
                cfg.blocks[target_block].preds.push(bid);
                cfg.blocks[bid].succs.push(target_block);
            }
        }
    }

    /// Run worklist-based abstract interpretation over the CFG.
    ///
    /// Returns a list of (jump_inst, resolved_target) pairs and the count of resolvable jumps.
    /// Stack snapshots are recorded into `self.snapshots` during the fixpoint.
    fn run_abstract_interp(&mut self) -> (Vec<(Inst, JumpTarget)>, usize) {
        let num_blocks = self.cfg.blocks.len();

        // Initialize block states. Entry block starts with an empty stack.
        let mut block_states: IndexVec<Block, BlockState> =
            IndexVec::from_vec(vec![BlockState::Bottom; num_blocks]);
        block_states[Block::from_usize(0)] = BlockState::Known(Vec::new());

        // Collect unresolved jumps.
        let mut jump_insts: Vec<Inst> = Vec::new();
        for (i, inst) in self.insts.iter_enumerated() {
            if inst.is_jump() && !inst.flags.contains(InstFlags::STATIC_JUMP) {
                jump_insts.push(i);
            }
        }

        let mut const_sets = ConstSetInterner::new();
        let discovered_edges = self.run_fixpoint(&mut block_states, &mut const_sets);

        if jump_insts.is_empty() {
            return (Vec::new(), 0);
        }

        // After convergence, resolve each dynamic jump from its snapshot operand.
        let mut jump_targets: Vec<(Inst, JumpTarget)> = Vec::new();
        let mut has_top_jump = false;
        for &jump_inst in &jump_insts {
            let target = match self.snapshots.inputs[jump_inst].last() {
                Some(&operand) => self.resolve_jump_operand(operand, &const_sets),
                None => {
                    // No snapshot means the block was never interpreted (unreachable).
                    trace!(%jump_inst, pc = self.insts[jump_inst].pc, "jump in unreached block");
                    JumpTarget::Bottom
                }
            };
            if matches!(target, JumpTarget::Top) {
                has_top_jump = true;
            }
            jump_targets.push((jump_inst, target));
        }

        // Invalidate resolutions that may be unsound due to incomplete analysis.
        if has_top_jump {
            self.invalidate_suspect_jumps(&mut jump_targets, &block_states, &discovered_edges);
        }

        let count = jump_targets
            .iter()
            .filter(|(_, t)| {
                matches!(t, JumpTarget::Const(_) | JumpTarget::Multi(_) | JumpTarget::Invalid)
            })
            .count();

        (jump_targets, count)
    }

    /// Resolves a jump target from the snapshot operand recorded during the fixpoint.
    fn resolve_jump_operand(&self, operand: AbsValue, const_sets: &ConstSetInterner) -> JumpTarget {
        match operand {
            AbsValue::Const(idx) => {
                let val = *self.u256_interner.borrow().get(idx);
                match usize::try_from(val) {
                    Ok(target_pc) if self.is_valid_jump(target_pc) => {
                        JumpTarget::Const(self.pc_to_inst(target_pc))
                    }
                    _ => JumpTarget::Invalid,
                }
            }
            AbsValue::ConstSet(set_idx) => {
                let consts = const_sets.get(set_idx);
                let interner = self.u256_interner.borrow();
                let mut targets = SmallVec::new();
                for &idx in consts {
                    let val = *interner.get(idx);
                    match usize::try_from(val) {
                        Ok(pc) if self.is_valid_jump(pc) => {
                            targets.push(self.pc_to_inst(pc));
                        }
                        _ => {
                            // Mixed valid + invalid: can't resolve since at runtime
                            // the value might be any member of the set.
                            return JumpTarget::Top;
                        }
                    }
                }
                if !targets.is_empty() { JumpTarget::Multi(targets) } else { JumpTarget::Invalid }
            }
            AbsValue::Top => JumpTarget::Top,
        }
    }

    /// Adds discovered dynamic-jump target edges for a block.
    fn discover_jump_edges(
        &self,
        operand: AbsValue,
        bid: Block,
        const_sets: &ConstSetInterner,
        discovered: &mut IndexVec<Block, SmallVec<[Block; 4]>>,
        disc_preds: &mut IndexVec<Block, SmallVec<[Block; 4]>>,
    ) {
        let consts = match operand {
            AbsValue::Const(idx) => Either::Left(std::iter::once(idx)),
            AbsValue::ConstSet(set_idx) => Either::Right(const_sets.get(set_idx).iter().copied()),
            AbsValue::Top => return,
        };
        let interner = self.u256_interner.borrow();
        for idx in consts {
            let Ok(target_pc) = usize::try_from(*interner.get(idx)) else { continue };
            if !self.is_valid_jump(target_pc) {
                continue;
            }
            let ti = self.pc_to_inst(target_pc);
            if let Some(tb) = self.cfg.inst_to_block[ti]
                && !discovered[bid].contains(&tb)
            {
                discovered[bid].push(tb);
                disc_preds[tb].push(bid);
            }
        }
    }

    /// Invalidates jump resolutions that may be unsound due to unresolved `Top` jumps.
    ///
    /// When unresolved dynamic jumps remain, any reachable `JUMPDEST` block could
    /// potentially be reached by those jumps with an arbitrary stack state. This means
    /// resolutions derived from the fixpoint may be based on incomplete information.
    ///
    /// The conservative rule: seed every reachable `JUMPDEST` block as suspect,
    /// propagate forward through the CFG and discovered edges, and invalidate all
    /// resolved jumps in suspect blocks.
    fn invalidate_suspect_jumps(
        &self,
        jump_targets: &mut [(Inst, JumpTarget)],
        block_states: &IndexVec<Block, BlockState>,
        discovered_edges: &IndexVec<Block, SmallVec<[Block; 4]>>,
    ) {
        let num_blocks = self.cfg.blocks.len();

        // Seed: every reachable JUMPDEST block is suspect when Top jumps exist.
        let mut suspect: BitVec = BitVec::repeat(false, num_blocks);
        for bid in self.cfg.blocks.indices() {
            if !matches!(block_states[bid], BlockState::Bottom)
                && self.insts[self.cfg.blocks[bid].insts.start].is_jumpdest()
            {
                suspect.set(bid.index(), true);
            }
        }

        // Propagate suspect flag forward through the CFG.
        let mut propagate = Worklist::new(num_blocks);
        for bid in self.cfg.blocks.indices() {
            if suspect[bid.index()] {
                propagate.push(bid);
            }
        }
        while let Some(bid) = propagate.pop() {
            for &succ in self.cfg.blocks[bid].succs.iter().chain(discovered_edges[bid].iter()) {
                let si = succ.index();
                if !suspect[si] {
                    suspect.set(si, true);
                    propagate.push(succ);
                }
            }
        }

        for (inst, target) in jump_targets.iter_mut() {
            if !matches!(target, JumpTarget::Const(_) | JumpTarget::Multi(_) | JumpTarget::Invalid)
            {
                continue;
            }
            if let Some(bid) = self.cfg.inst_to_block[*inst]
                && suspect[bid.index()]
            {
                *target = JumpTarget::Top;
            }
        }
    }

    /// Run a worklist-based fixpoint to compute abstract block states.
    ///
    /// Returns the discovered dynamic-jump target edges per block.
    fn run_fixpoint(
        &mut self,
        block_states: &mut IndexVec<Block, BlockState>,
        const_sets: &mut ConstSetInterner,
    ) -> IndexVec<Block, SmallVec<[Block; 4]>> {
        let num_blocks = self.cfg.blocks.len();
        let mut worklist = Worklist::new(num_blocks);
        worklist.push(Block::from_usize(0));

        // Discovered dynamic-jump target edges per block.
        let mut discovered: IndexVec<Block, SmallVec<[Block; 4]>> =
            IndexVec::from_vec(vec![SmallVec::new(); num_blocks]);
        // Reverse map: discovered predecessors per block.
        let mut disc_preds: IndexVec<Block, SmallVec<[Block; 4]>> =
            IndexVec::from_vec(vec![SmallVec::new(); num_blocks]);

        let max_iterations = num_blocks * 8;
        let mut iterations = 0;
        let mut converged = true;

        // Reusable buffer to avoid per-iteration allocations.
        let mut stack_buf: Vec<AbsValue> = Vec::with_capacity(MAX_ABS_STACK_DEPTH);

        while let Some(bid) = worklist.pop() {
            iterations += 1;
            if iterations > max_iterations {
                converged = false;
                break;
            }

            // Copy input state into reusable buffer.
            stack_buf.clear();
            match &block_states[bid] {
                BlockState::Known(s) => stack_buf.extend_from_slice(s),
                BlockState::Bottom => continue,
            };

            let block = &self.cfg.blocks[bid];
            let block_insts = block.insts.clone();
            let block_succs = block.succs.clone();
            let insts_iter =
                (block_insts.start.index()..block_insts.end.index()).map(Inst::from_usize);
            if !self.interpret_block(insts_iter, &mut stack_buf) {
                continue;
            }

            // Discover dynamic-jump target edges from the snapshot recorded above.
            let term_inst = block_insts.end - 1;
            let term = &self.insts[term_inst];
            if term.is_jump()
                && !term.flags.contains(InstFlags::STATIC_JUMP)
                && let Some(&operand) = self.snapshots.inputs[term_inst].last()
            {
                self.discover_jump_edges(
                    operand,
                    bid,
                    const_sets,
                    &mut discovered,
                    &mut disc_preds,
                );
            }

            // Propagate to static CFG successors and discovered dynamic-jump targets.
            for &succ in block_succs.iter().chain(&discovered[bid]) {
                if block_states[succ].join(&stack_buf, const_sets) {
                    worklist.push(succ);
                }
            }
        }

        debug!(
            "{msg} after {iterations} iterations (max={max_iterations})",
            msg = if converged { "converged" } else { "did not converge" },
        );

        discovered
    }

    /// Interpret a sequence of instructions on the abstract stack.
    /// Returns `false` on stack underflow (conflict).
    ///
    /// The caller must pre-fill `stack` with the input state; on return it contains the output.
    /// Records per-instruction operand snapshots into `self.snapshots`.
    fn interpret_block(
        &mut self,
        insts: impl IntoIterator<Item = Inst>,
        stack: &mut Vec<AbsValue>,
    ) -> bool {
        for i in insts {
            let inst = &self.insts[i];
            if inst.is_dead_code() {
                continue;
            }

            // Instructions marked SKIP_LOGIC (the PUSH in a PUSH+JUMP pair) are no-ops
            // for abstract interpretation — the value is consumed by the already-resolved jump.
            if inst.flags.contains(InstFlags::SKIP_LOGIC) {
                continue;
            }

            let (inp, out) = inst.stack_io();
            let inp = inp as usize;
            let out = out as usize;

            // Record pre-instruction input operand snapshot (in stack order, TOS last).
            if inp > 0 {
                let start = stack.len().saturating_sub(inp);
                let snap = &mut self.snapshots.inputs[i];
                snap.clear();
                snap.extend_from_slice(&stack[start..]);
            }

            match inst.opcode {
                op::PUSH0 => {
                    stack.push(AbsValue::Const(self.intern_u256(U256::ZERO)));
                }
                op::PUSH1..=op::PUSH32 => {
                    let val = self.get_imm(inst).map_or(AbsValue::Top, |imm| {
                        AbsValue::Const(self.intern_u256(U256::from_be_slice(imm)))
                    });
                    stack.push(val);
                }
                op::POP => {
                    if stack.pop().is_none() {
                        return false;
                    }
                }
                op::DUP1..=op::DUP16 => {
                    let depth = (inst.opcode - op::DUP1 + 1) as usize;
                    if stack.len() < depth {
                        return false;
                    }
                    stack.push(stack[stack.len() - depth]);
                }
                op::SWAP1..=op::SWAP16 => {
                    let depth = (inst.opcode - op::SWAP1 + 1) as usize;
                    let len = stack.len();
                    if len < depth + 1 {
                        return false;
                    }
                    stack.swap(len - 1, len - 1 - depth);
                }
                _ => {
                    if stack.len() < inp {
                        return false;
                    }

                    // Try constant folding for common arithmetic.
                    let result = if out > 0 {
                        super::const_fold::try_const_fold(
                            inst,
                            &stack[stack.len() - inp..],
                            &mut self.u256_interner.borrow_mut(),
                            self.code.len(),
                        )
                    } else {
                        None
                    };

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

            // Record post-instruction output snapshot.
            if out > 0 {
                self.snapshots.outputs[i] = stack.last().copied();
            }
        }

        true
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) use crate::bytecode::Inst;
    use revm_primitives::{hardfork::SpecId, hex};

    fn analyze_hex(hex: &str) -> Bytecode<'static> {
        let code = hex::decode(hex.trim()).unwrap();
        analyze_code(code)
    }

    pub(crate) fn analyze_code(code: Vec<u8>) -> Bytecode<'static> {
        analyze_code_with(code, Default::default())
    }

    pub(crate) fn analyze_code_with(
        code: Vec<u8>,
        config: crate::bytecode::AnalysisConfig,
    ) -> Bytecode<'static> {
        crate::tests::init_tracing();

        eprintln!("{}", hex::encode(&code));
        let mut bytecode = Bytecode::new(code, SpecId::CANCUN);
        bytecode.config = config;
        bytecode.analyze().unwrap();
        bytecode
    }

    #[test]
    fn revert_sub_call_storage_oog() {
        let bytecode = analyze_hex(
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
        let bytecode = analyze_hex(
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
            let bytecode = analyze_hex(hex);
            eprintln!("{bytecode}");
        }
    }

    #[test]
    fn const_operand_basic() {
        // inst 0: PUSH1 0x42 -> stack: [0x42]
        // inst 1: PUSH1 0x01 -> stack: [0x42, 0x01]
        // inst 2: ADD        -> stack: [0x43]  (const-folded)
        // inst 3: PUSH1 0x00 -> stack: [0x43, 0x00]
        // inst 4: MSTORE     -> pops 2
        // inst 5: JUMPDEST
        // inst 6: STOP
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH1, 0x42,
            op::PUSH1, 0x01,
            op::ADD,
            op::PUSH1, 0x00,
            op::MSTORE,
            op::JUMPDEST,
            op::STOP,
        ]);
        // At inst 2 (ADD), operand 0 (TOS) = 0x01, operand 1 = 0x42.
        assert_eq!(bytecode.const_operand(Inst::from_usize(2), 0), Some(U256::from(0x01)));
        assert_eq!(bytecode.const_operand(Inst::from_usize(2), 1), Some(U256::from(0x42)));
        // At inst 4 (MSTORE), operand 0 (TOS) = 0x00, operand 1 = 0x43 (folded ADD result).
        assert_eq!(bytecode.const_operand(Inst::from_usize(4), 0), Some(U256::from(0x00)));
        assert_eq!(bytecode.const_operand(Inst::from_usize(4), 1), Some(U256::from(0x43)));
    }

    #[test]
    fn const_operand_dynamic() {
        // CALLDATALOAD pushes an unknown value -> const_operand should return None.
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH0,
            op::CALLDATALOAD,
            op::PUSH0,
            op::MSTORE,
            op::STOP,
        ]);
        // At inst 3 (MSTORE), operand 0 (TOS) = 0x00, operand 1 = unknown (CALLDATALOAD result).
        assert_eq!(bytecode.const_operand(Inst::from_usize(3), 0), Some(U256::ZERO));
        assert_eq!(bytecode.const_operand(Inst::from_usize(3), 1), None);
    }

    #[test]
    fn const_output_basic() {
        // PUSH1 0x42 -> output[0] = 0x42
        // PUSH1 0x01 -> output[0] = 0x01
        // ADD        -> output[0] = 0x43 (const-folded)
        // PUSH0      -> output[0] = 0x00
        // MSTORE     -> no outputs
        // STOP
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH1, 0x42,
            op::PUSH1, 0x01,
            op::ADD,
            op::PUSH0,
            op::MSTORE,
            op::STOP,
        ]);
        assert_eq!(bytecode.const_output(Inst::from_usize(0)), Some(U256::from(0x42)));
        assert_eq!(bytecode.const_output(Inst::from_usize(1)), Some(U256::from(0x01)));
        assert_eq!(bytecode.const_output(Inst::from_usize(2)), Some(U256::from(0x43)));
        assert_eq!(bytecode.const_output(Inst::from_usize(3)), Some(U256::ZERO));
        // MSTORE has no outputs.
        assert_eq!(bytecode.const_output(Inst::from_usize(4)), None);
    }

    #[test]
    fn const_output_dynamic() {
        // CALLDATALOAD pushes an unknown -> output should be None.
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH0,
            op::CALLDATALOAD,
            op::STOP,
        ]);
        assert_eq!(bytecode.const_output(Inst::from_usize(0)), Some(U256::ZERO));
        assert_eq!(bytecode.const_output(Inst::from_usize(1)), None);
    }

    #[test]
    fn snapshots_all_blocks() {
        // Snapshots must be populated for every block, not just those with
        // unresolved jumps. Here block 0 has a static PUSH+JUMP so the old
        // code would skip it; we verify const_operand still works.
        //
        // pc 0: PUSH1 0x42   (inst 0) -> stack: [0x42]
        // pc 2: PUSH1 0x05  (inst 1) -> stack: [0x42, 0x05]
        // pc 4: JUMP        (inst 2) -> static jump to pc 5
        // pc 5: JUMPDEST    (inst 3)
        // pc 6: PUSH1 0x01  (inst 4) -> stack: [0x42, 0x01]
        // pc 8: ADD         (inst 5) -> stack: [0x43]
        // pc 9: PUSH1 0x00  (inst 6) -> stack: [0x43, 0x00]
        // pc 11: MSTORE     (inst 7)
        // pc 12: STOP       (inst 8)
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH1, 0x42,
            op::PUSH1, 0x05,
            op::JUMP,
            op::JUMPDEST,
            op::PUSH1, 0x01,
            op::ADD,
            op::PUSH1, 0x00,
            op::MSTORE,
            op::STOP,
        ]);
        // Block 0 (PUSH+PUSH+JUMP): the PUSH instructions should have snapshots.
        assert_eq!(bytecode.const_output(Inst::from_usize(0)), Some(U256::from(0x42)));
        assert_eq!(bytecode.const_output(Inst::from_usize(1)), Some(U256::from(0x05)));
        // Block 1 (JUMPDEST..STOP): ADD operands should be known.
        assert_eq!(bytecode.const_operand(Inst::from_usize(5), 0), Some(U256::from(0x01)));
        assert_eq!(bytecode.const_operand(Inst::from_usize(5), 1), Some(U256::from(0x42)));
        assert_eq!(bytecode.const_output(Inst::from_usize(5)), Some(U256::from(0x43)));
    }

    #[test]
    fn multi_target_jump() {
        // Internal function called from two sites with different return addresses.
        // The return JUMP at the end should resolve to Multi([ret1, ret2]).
        let ret1: u8 = 5;
        let ret2: u8 = 12;
        let func: u8 = 15;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            // Call site 1.
            op::PUSH1, ret1,    // 0: push return address
            op::PUSH1, func,    // 2: push function entry
            op::JUMP,           // 4: call function (static: PUSH+JUMP)
            op::JUMPDEST,       // 5: ret1
            op::POP,            // 6: consume function result
            // Call site 2.
            op::PUSH1, ret2,    // 7: push return address
            op::PUSH1, func,    // 9: push function entry
            op::JUMP,           // 11: call function (static: PUSH+JUMP)
            op::JUMPDEST,       // 12: ret2
            op::POP,            // 13: consume function result
            op::STOP,           // 14
            // Internal function.
            op::JUMPDEST,       // 15: func entry
            op::PUSH1, 0x42,    // 16: push a result
            op::SWAP1,          // 18: swap result and return address
            op::JUMP,           // 19: return (dynamic)
        ]);
        eprintln!("{bytecode}");

        // The return JUMP (last inst before STOP's block) should be multi-target.
        let return_jump = bytecode
            .iter_insts()
            .find(|(_, d)| d.is_jump() && d.flags.contains(InstFlags::MULTI_JUMP));
        assert!(return_jump.is_some(), "expected a multi-target jump");
        let (rj_inst, _) = return_jump.unwrap();
        let targets = bytecode.multi_jump_targets(rj_inst).unwrap();
        assert_eq!(targets.len(), 2, "expected 2 targets, got {}", targets.len());
        // Verify both targets are JUMPDESTs.
        for &t in targets {
            assert_eq!(bytecode.inst(t).opcode, op::JUMPDEST);
        }
        // No dynamic jumps should remain.
        assert!(!bytecode.has_dynamic_jumps, "expected no dynamic jumps");
    }

    /// Demonstrates a known unsoundness in `invalidate_suspect_jumps`.
    ///
    /// The invalidation pass only seeds suspicion from `Bottom + JUMPDEST` predecessors,
    /// but an unresolved dynamic jump can also reach a `Known` JUMPDEST block at runtime
    /// with a different stack than the fixpoint computed. This test constructs a case
    /// where the analysis incorrectly resolves a return JUMP to `Multi` even though an
    /// opaque (MLOAD-based) dynamic jump could reach the same function entry with an
    /// arbitrary return address.
    ///
    /// Layout:
    ///   Call site 1: PUSH ret1, PUSH fn, JUMP → fn returns to ret1
    ///   Call site 2: PUSH ret2, PUSH fn, JUMP → fn returns to ret2
    ///   Opaque jump: PUSH0, MLOAD, JUMP      → Top target, could be fn at runtime
    ///   fn:          JUMPDEST, PUSH 0x42, SWAP1, JUMP → return (should be Top)
    ///
    /// The fn JUMPDEST block is `Known([ConstSet({ret1, ret2})])` from the two static
    /// callers, so the return JUMP resolves to `Multi([ret1, ret2])`. But the opaque
    /// jump could call fn with any return address, making that resolution unsound.
    /// `invalidate_suspect_jumps` now seeds every reachable JUMPDEST block as suspect,
    /// so the return jump is correctly invalidated.
    #[test]
    fn known_jumpdest_unsound_resolution() {
        let ret1: u8 = 5;
        let ret2: u8 = 12;
        let fn_entry: u8 = 20;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            // Call site 1.
            op::PUSH1, ret1,        // pc=0
            op::PUSH1, fn_entry,    // pc=2
            op::JUMP,               // pc=4: → fn_entry
            op::JUMPDEST,           // pc=5: ret1
            op::POP,                // pc=6: drop result
            // Call site 2.
            op::PUSH1, ret2,        // pc=7
            op::PUSH1, fn_entry,    // pc=9
            op::JUMP,               // pc=11: → fn_entry
            op::JUMPDEST,           // pc=12: ret2
            op::POP,                // pc=13: drop result
            // Opaque jump: loads target from memory (Top).
            op::PUSH0,              // pc=14
            op::MLOAD,              // pc=15: → Top
            op::JUMP,               // pc=16: dynamic, target = Top
            op::INVALID,            // pc=17
            op::INVALID,            // pc=18
            op::INVALID,            // pc=19
            // Internal function: pushes result, swaps with return addr, jumps back.
            op::JUMPDEST,           // pc=20: fn_entry
            op::PUSH1, 0x42,        // pc=21: result
            op::SWAP1,              // pc=23: swap result and return addr
            op::JUMP,               // pc=24: return (UNSOUND: resolved to Multi)
        ]);
        eprintln!("{bytecode}");

        // The return JUMP at pc=24 should NOT be resolved — the opaque jump at
        // pc=16 can reach fn_entry with any return address. But the current
        // invalidation misses this because fn_entry's block is Known, not Bottom.
        let return_jump = bytecode.iter_insts().find(|(_, d)| d.pc == 24 && d.is_jump());
        let (_, rj) = return_jump.unwrap();
        assert!(
            !rj.flags.contains(InstFlags::STATIC_JUMP),
            "unsound: return jump should not be resolved"
        );
    }

    #[test]
    fn nested_call_return() {
        // Two-level call chain: outer calls wrapper, wrapper calls inner function.
        // Direct caller also calls the inner function directly.
        //
        //   call site 1 (direct):   PUSH ret1, PUSH inner, JUMP
        //   call site 2 (wrapper):  PUSH ret2, PUSH wrapper, JUMP
        //     wrapper:              PUSH ret_w, PUSH inner, JUMP
        //     ret_w:                SWAP1 POP JUMP  (returns to ret2)
        //
        // The inner function is called with different stack depths from the two
        // sites. Its return JUMP resolves to Multi([ret1, ret_w]). But the
        // wrapper's final JUMP (at ret_w) must recover the outer return address
        // from deep in the stack — which the analysis currently cannot resolve
        // because the top-aligned join pads it to Top.
        let ret1: u8 = 5; // return from direct call
        let ret2: u8 = 14; // return from wrapper call
        let wrapper: u8 = 19; // wrapper entry
        let ret_w: u8 = 27; // inner returns here (inside wrapper)
        let inner: u8 = 32; // inner function entry
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            // Call site 1: direct call to inner.
            op::PUSH1, ret1,        // pc=0
            op::PUSH1, inner,       // pc=2
            op::JUMP,               // pc=4: -> inner
            op::JUMPDEST,           // pc=5: ret1
            op::POP,                // pc=6: drop result
            // Call site 2: call through wrapper.
            op::PUSH1, ret2,        // pc=7
            op::PUSH1, 0x42,        // pc=9: an argument
            op::PUSH1, wrapper,     // pc=11
            op::JUMP,               // pc=13: -> wrapper
            op::JUMPDEST,           // pc=14: ret2
            op::POP,                // pc=15: drop result
            op::POP,                // pc=16: drop arg
            op::STOP,               // pc=17
            op::INVALID,            // pc=18
            // Wrapper function: calls inner, then returns to caller.
            // Entry stack: [outer_ret, arg]
            op::JUMPDEST,           // pc=19: wrapper entry
            op::PUSH1, ret_w,       // pc=20: push return addr for inner
            op::PUSH1, inner,       // pc=22: push inner entry
            op::JUMP,               // pc=24: -> inner
            op::INVALID,            // pc=25
            op::INVALID,            // pc=26
            op::JUMPDEST,           // pc=27: ret_w — inner returned here
            // Stack: [outer_ret, arg, inner_result]
            op::POP,                // pc=28: drop inner_result
            op::POP,                // pc=29: drop arg
            op::JUMP,               // pc=30: return to caller via outer_ret (dynamic)
            op::INVALID,            // pc=31
            // Inner function: pushes a result and returns.
            // Entry stack: [..., ret_addr]
            op::JUMPDEST,           // pc=32: inner entry
            op::PUSH1, 0x42,        // pc=33: push result → [..., ret_addr, 0x42]
            op::SWAP1,              // pc=35: → [..., 0x42, ret_addr]
            op::JUMP,               // pc=36: jump to ret_addr, leaves [..., 0x42]
        ]);
        eprintln!("{bytecode}");

        // The wrapper return JUMP (pc=30) remains dynamic because the outer
        // return address is lost to Top during the top-aligned join.
        // Because an unresolved Top jump exists, the conservative invalidation
        // also invalidates the inner return JUMP — any reachable JUMPDEST
        // (including inner's entry) is suspect.
        assert!(bytecode.has_dynamic_jumps, "expected dynamic jumps to remain");
    }

    #[test]
    fn hash_10k() {
        let code = revm_primitives::hex::decode(
            include_str!("../../../../../data/hash_10k.rt.hex").trim(),
        )
        .unwrap();
        let mut bytecode = Bytecode::new(code, SpecId::CANCUN);
        bytecode.analyze().unwrap();
        eprintln!("{bytecode}");
        assert!(!bytecode.has_dynamic_jumps, "expected all jumps to be resolved");
    }
}

#[cfg(test)]
mod tests_edge_cases {
    use super::{tests::*, *};

    /// Three callers to the same internal function. The return JUMP should
    /// resolve to Multi with three targets.
    #[test]
    fn multi_target_jump_three_callers() {
        let ret1: u8 = 5;
        let ret2: u8 = 12;
        let ret3: u8 = 19;
        let func: u8 = 22;
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
            // Call site 3.
            op::PUSH1, ret3,
            op::PUSH1, func,
            op::JUMP,
            op::JUMPDEST,       // ret3
            op::POP,
            op::STOP,
            // Internal function.
            op::JUMPDEST,       // func
            op::PUSH1, 0x42,
            op::SWAP1,
            op::JUMP,           // return (dynamic)
        ]);
        eprintln!("{bytecode}");

        let return_jump = bytecode
            .iter_insts()
            .find(|(_, d)| d.is_jump() && d.flags.contains(InstFlags::MULTI_JUMP));
        assert!(return_jump.is_some(), "expected a multi-target jump");
        let (rj_inst, _) = return_jump.unwrap();
        let targets = bytecode.multi_jump_targets(rj_inst).unwrap();
        assert_eq!(targets.len(), 3, "expected 3 targets, got {}", targets.len());
        assert!(!bytecode.has_dynamic_jumps, "expected no dynamic jumps");
    }

    /// DUP and SWAP should preserve constant values through the stack.
    #[test]
    fn dup_swap_const_propagation() {
        // PUSH1 0xAA -> [0xAA]
        // PUSH1 0xBB -> [0xAA, 0xBB]
        // DUP2       -> [0xAA, 0xBB, 0xAA]
        // SWAP1      -> [0xAA, 0xAA, 0xBB]
        // PUSH0      -> [0xAA, 0xAA, 0xBB, 0x00]
        // MSTORE     -> pops (0x00, 0xBB), leaves [0xAA, 0xAA]
        // STOP
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH1, 0xAA,   // inst 0
            op::PUSH1, 0xBB,   // inst 1
            op::DUP2,          // inst 2
            op::SWAP1,         // inst 3
            op::PUSH0,         // inst 4
            op::MSTORE,        // inst 5
            op::STOP,          // inst 6
        ]);
        // DUP2 output should be 0xAA.
        assert_eq!(bytecode.const_output(Inst::from_usize(2)), Some(U256::from(0xAA)));
        // At MSTORE (inst 5): TOS = 0x00, second = 0xBB (swapped back).
        assert_eq!(bytecode.const_operand(Inst::from_usize(5), 0), Some(U256::ZERO));
        assert_eq!(bytecode.const_operand(Inst::from_usize(5), 1), Some(U256::from(0xBB)));
    }

    /// JUMPI with a constant condition should still record both operands.
    #[test]
    fn jumpi_const_condition_snapshot() {
        let target: u8 = 8;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH1, 0x01,       // inst 0: condition = 1 (always taken)
            op::PUSH1, target,     // inst 1: target
            op::JUMPI,             // inst 2: static JUMPI
            op::STOP,              // inst 3: fallthrough
            op::JUMPDEST,          // inst 4: target
            op::STOP,              // inst 5
        ]);
        eprintln!("{bytecode}");
        // The JUMPI should be resolved as static.
        assert!(!bytecode.has_dynamic_jumps);
    }

    /// A loop where the stack grows each iteration until clamped.
    /// Verifies the analysis converges without losing precision at the top.
    #[test]
    fn loop_stack_growth_clamping() {
        // bb0: PUSH1 0x42, then fall through to bb1.
        // bb1: JUMPDEST, PUSH1 0x01, PUSH1 bb1_pc, JUMP  (loop back, growing stack).
        //
        // The stack grows by 1 each iteration (net +1 from PUSH before the loop jump).
        // The abstract stack should clamp to MAX_ABS_STACK_DEPTH without conflict.
        let loop_pc: u8 = 3;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH1, 0x42,       // pc=0: initial value
            op::PUSH1, 0x01,       // pc=2: another
            op::JUMPDEST,          // pc=3: loop header (bb1)
            op::PUSH1, 0x01,       // pc=4: push each iteration
            op::PUSH1, loop_pc,    // pc=6: loop target
            op::JUMP,              // pc=8: back-edge
        ]);
        eprintln!("{bytecode}");
        // Should converge without panicking.
        // The JUMP should be static (resolved by adjacent PUSH+JUMP).
        assert!(!bytecode.has_dynamic_jumps);
    }

    /// A single-instruction block (just JUMPDEST as the terminator) used as a
    /// jump target. This tests the edge case where block.len() == 1.
    #[test]
    fn single_inst_block_jump_target() {
        let target: u8 = 5;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH1, target,     // pc=0
            op::JUMP,              // pc=2: static jump
            op::INVALID,           // pc=3
            op::INVALID,           // pc=4
            op::JUMPDEST,          // pc=5: single-inst block
            op::STOP,              // pc=6
        ]);
        eprintln!("{bytecode}");
        assert!(!bytecode.has_dynamic_jumps);
    }

    /// A JUMP with an invalid target (not a JUMPDEST) should be marked invalid.
    #[test]
    fn invalid_jump_target() {
        // The dynamic jump target points to a non-JUMPDEST instruction.
        // The function pushes a constant that isn't a valid JUMPDEST pc.
        let func_pc: u8 = 5;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH1, 0xFF,       // pc=0: push invalid target
            op::PUSH1, func_pc,    // pc=2: push function entry
            op::JUMP,              // pc=4: -> func
            // Function: swap and jump to the caller-provided target.
            op::JUMPDEST,          // pc=5: func
            op::JUMP,              // pc=6: jump to 0xFF (invalid)
        ]);
        eprintln!("{bytecode}");
        // The dynamic JUMP at pc=6 should be resolved as invalid.
        let jump_inst = bytecode
            .iter_insts()
            .find(|(_, d)| d.is_jump() && d.flags.contains(InstFlags::INVALID_JUMP));
        assert!(jump_inst.is_some(), "expected an invalid jump");
    }

    /// Constant propagation through a diamond CFG (if-then-else merge).
    /// Both branches push the same constant, so the merge should preserve it.
    #[test]
    fn diamond_cfg_same_const() {
        let then_pc: u8 = 9;
        let merge_pc: u8 = 13;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH1, 0x42,           // pc=0: push same const on both paths
            op::PUSH0,                 // pc=2: condition (always false)
            op::PUSH1, then_pc,        // pc=3
            op::JUMPI,                 // pc=5: branch
            // Else: push same const.
            op::PUSH1, merge_pc,       // pc=6
            op::JUMP,                  // pc=8: -> merge
            // Then:
            op::JUMPDEST,              // pc=9
            op::PUSH1, merge_pc,       // pc=10
            op::JUMP,                  // pc=12: -> merge
            // Merge:
            op::JUMPDEST,              // pc=13
            op::PUSH0,                 // pc=14
            op::MSTORE,               // pc=15: MSTORE(0, 0x42)
            op::STOP,                  // pc=16
        ]);
        eprintln!("{bytecode}");
        // At the merge MSTORE, the value (operand 1) should still be 0x42
        // since both branches had the same constant on the stack.
        let mstore = bytecode.iter_insts().find(|(_, d)| d.opcode == op::MSTORE).unwrap().0;
        assert_eq!(bytecode.const_operand(mstore, 1), Some(U256::from(0x42)));
    }

    /// Diamond CFG where branches push different constants — the merge should
    /// yield None (Top) for const_operand.
    #[test]
    fn diamond_cfg_different_const() {
        let then_pc: u8 = 10;
        let merge_pc: u8 = 16;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::PUSH0,                 // pc=0: condition
            op::PUSH1, then_pc,        // pc=1
            op::JUMPI,                 // pc=3: branch
            // Else: push 0xAA.
            op::PUSH1, 0xAA,          // pc=4
            op::PUSH1, merge_pc,       // pc=6
            op::JUMP,                  // pc=8: -> merge
            op::INVALID,               // pc=9
            // Then: push 0xBB.
            op::JUMPDEST,              // pc=10
            op::PUSH1, 0xBB,          // pc=11
            op::PUSH1, merge_pc,       // pc=13
            op::JUMP,                  // pc=15: -> merge
            // Merge:
            op::JUMPDEST,              // pc=16: merge
            op::PUSH0,                 // pc=17
            op::MSTORE,               // pc=18: MSTORE(0, ???)
            op::STOP,                  // pc=19
        ]);
        eprintln!("{bytecode}");
        // At the merge MSTORE, the value (operand 1) should be None
        // since the two branches push different constants.
        let mstore = bytecode.iter_insts().find(|(_, d)| d.opcode == op::MSTORE).unwrap().0;
        assert_eq!(bytecode.const_operand(mstore, 1), None);
    }

    /// A private-return trampoline (`JUMPDEST; JUMP`) reached by an opaque jump
    /// can forward arbitrary stacks into a real function entry. The trampoline's
    /// Top jump must not be filtered out by the `is_private_return` check.
    #[test]
    fn trampoline_private_return_unsound() {
        let ret1: u8 = 10;
        let ret2: u8 = 17;
        let opaque_path: u8 = 20;
        let tramp: u8 = 26;
        let fn_entry: u8 = 28;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            // Entry: branch to opaque path or call site 1.
            op::PUSH0,                  // pc=0
            op::CALLDATALOAD,           // pc=1
            op::PUSH1, opaque_path,     // pc=2
            op::JUMPI,                  // pc=4
            // Fallthrough: call site 1.
            op::PUSH1, ret1,            // pc=5
            op::PUSH1, fn_entry,        // pc=7
            op::JUMP,                   // pc=9
            // ret1.
            op::JUMPDEST,               // pc=10
            op::POP,                    // pc=11
            // Call site 2.
            op::PUSH1, ret2,            // pc=12
            op::PUSH1, fn_entry,        // pc=14
            op::JUMP,                   // pc=16
            // ret2.
            op::JUMPDEST,               // pc=17
            op::POP,                    // pc=18
            op::STOP,                   // pc=19
            // Opaque path (reachable from JUMPI at pc=4).
            op::JUMPDEST,               // pc=20
            op::PUSH0,                  // pc=21
            op::MLOAD,                  // pc=22: any_ret (Top)
            op::PUSH1, tramp,           // pc=23
            op::JUMP,                   // pc=25
            // Trampoline: bare JUMPDEST + JUMP (is_private_return = true).
            op::JUMPDEST,               // pc=26
            op::JUMP,                   // pc=27: target = any_ret (Top)
            // Internal function.
            op::JUMPDEST,               // pc=28: fn_entry
            op::PUSH1, 0x42,            // pc=29
            op::SWAP1,                  // pc=31
            op::JUMP,                   // pc=32: return
        ]);
        eprintln!("{bytecode}");

        // The fn return JUMP at pc=32 must NOT be resolved — the trampoline
        // can reach fn_entry with any return address.
        let return_jump = bytecode.iter_insts().find(|(_, d)| d.pc == 32 && d.is_jump());
        let (_, rj) = return_jump.unwrap();
        assert!(
            !rj.flags.contains(InstFlags::STATIC_JUMP),
            "unsound: return jump should not be resolved when trampoline exists"
        );
    }

    /// A JUMPI-based return where the target comes from the entry stack should
    /// be invalidated when a Top jump exists, just like JUMP-based returns.
    #[test]
    fn jumpi_return_unsound() {
        let ret1: u8 = 5;
        let ret2: u8 = 12;
        let fn_entry: u8 = 20;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            // Call site 1.
            op::PUSH1, ret1,        // pc=0
            op::PUSH1, fn_entry,    // pc=2
            op::JUMP,               // pc=4
            op::JUMPDEST,           // pc=5: ret1
            op::POP,                // pc=6
            // Call site 2.
            op::PUSH1, ret2,        // pc=7
            op::PUSH1, fn_entry,    // pc=9
            op::JUMP,               // pc=11
            op::JUMPDEST,           // pc=12: ret2
            op::POP,                // pc=13
            // Opaque jump.
            op::PUSH0,              // pc=14
            op::MLOAD,              // pc=15
            op::JUMP,               // pc=16: Top
            op::INVALID,            // pc=17
            op::INVALID,            // pc=18
            op::INVALID,            // pc=19
            // Internal function using JUMPI to return.
            op::JUMPDEST,           // pc=20: fn_entry
            op::PUSH1, 0x42,        // pc=21: result
            op::SWAP1,              // pc=23: [result, ret_addr]
            op::PUSH1, 0x01,        // pc=24: condition (always true)
            op::SWAP1,              // pc=26: [result, 1, ret_addr]
            op::JUMPI,              // pc=27: conditional return (always taken)
            op::STOP,               // pc=28: fallthrough (dead)
        ]);
        eprintln!("{bytecode}");

        // The JUMPI at pc=27 should NOT be resolved — opaque jump can reach
        // fn_entry with any return address.
        let return_jumpi = bytecode.iter_insts().find(|(_, d)| d.pc == 27 && d.is_jump());
        let (_, rj) = return_jumpi.unwrap();
        assert!(
            !rj.flags.contains(InstFlags::STATIC_JUMP),
            "unsound: JUMPI return should not be resolved"
        );
    }

    /// A third caller reaches the function entry with a known callee (static
    /// PUSH+JUMP) but an opaque return address (from MLOAD). The function's
    /// return jump must not be resolved because the return address is Top.
    #[test]
    fn opaque_return_addr_caller_unsound() {
        let ret1: u8 = 10;
        let ret2: u8 = 17;
        let opaque_caller: u8 = 20;
        let fn_entry: u8 = 26;
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            // Entry: branch to opaque caller or fallthrough.
            op::PUSH0,              // pc=0
            op::CALLDATALOAD,       // pc=1
            op::PUSH1, opaque_caller, // pc=2
            op::JUMPI,              // pc=4
            // Fallthrough: call site 1.
            op::PUSH1, ret1,        // pc=5
            op::PUSH1, fn_entry,    // pc=7
            op::JUMP,               // pc=9
            op::JUMPDEST,           // pc=10: ret1
            op::POP,                // pc=11
            // Call site 2.
            op::PUSH1, ret2,        // pc=12
            op::PUSH1, fn_entry,    // pc=14
            op::JUMP,               // pc=16
            op::JUMPDEST,           // pc=17: ret2
            op::POP,                // pc=18
            op::STOP,               // pc=19
            // Opaque third caller: loads return addr from memory (reachable via JUMPI).
            op::JUMPDEST,           // pc=20
            op::PUSH0,              // pc=21
            op::MLOAD,              // pc=22: any_ret
            op::PUSH1, fn_entry,    // pc=23
            op::JUMP,               // pc=25: → fn_entry with arbitrary return addr
            // Internal function.
            op::JUMPDEST,           // pc=26: fn_entry
            op::PUSH1, 0x42,        // pc=27
            op::SWAP1,              // pc=29
            op::JUMP,               // pc=30: return
        ]);
        eprintln!("{bytecode}");

        // The return JUMP at pc=30 must NOT be resolved — the opaque caller
        // at pc=25 reaches fn_entry with an arbitrary return address.
        let return_jump = bytecode.iter_insts().find(|(_, d)| d.pc == 30 && d.is_jump());
        let (_, rj) = return_jump.unwrap();
        assert!(
            !rj.flags.contains(InstFlags::STATIC_JUMP),
            "unsound: return jump should not be resolved with opaque caller"
        );
    }

    /// Bytecode that is just STOP — minimal edge case.
    #[test]
    fn empty_bytecode() {
        let bytecode = analyze_code(vec![op::STOP]);
        assert!(!bytecode.has_dynamic_jumps);
    }

    /// A dynamic jump in dead code should not cause `has_dynamic_jumps` to be true.
    #[test]
    fn dead_code_dynamic_jump() {
        #[rustfmt::skip]
        let bytecode = analyze_code(vec![
            op::STOP,
            op::PUSH0,
            op::CALLDATALOAD,
            op::JUMP,
        ]);
        eprintln!("{bytecode}");
        assert!(!bytecode.has_dynamic_jumps);
    }
}

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
        self.init_snapshots();

        let empty_sets = ConstSetInterner::new();
        let mut resolved = Vec::new();
        let mut stack = Vec::new();

        for bid in self.cfg.blocks.indices() {
            let block = &self.cfg.blocks[bid];

            // Compute required entry depth for this block using stack section analysis.
            let section =
                StackSection::from_stack_io(block.insts().map(|i| self.insts[i].stack_io()));

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

            let Some(&operand) = self.snapshots.inputs[term_inst].last() else { continue };
            debug_assert!(!matches!(operand, AbsValue::ConstSet(_)));
            let target = self.resolve_jump_operand(operand, &empty_sets);
            let JumpTarget::Const(target_inst) = target else { continue };

            // Log non-adjacent resolutions (not simple PUSH+JUMP).
            if enabled!(tracing::Level::TRACE)
                && let is_adjacent = (term_inst > 0 && {
                    let prev_inst = term_inst - 1;
                    let prev = &self.insts[prev_inst];
                    matches!(prev.opcode, op::PUSH0..=op::PUSH32)
                        && !prev.is_dead_code()
                        && block.insts.contains(&prev_inst)
                })
                && !is_adjacent
            {
                trace!(%term_inst, %target_inst, pc = self.insts[term_inst].pc, "resolved non-adjacent jump");
            }

            resolved.push((term_inst, target));
        }

        let newly_resolved = self.commit_resolved_jumps(&resolved);
        debug!(newly_resolved, "finished");
        self.recompute_has_dynamic_jumps();
    }

    /// Runs abstract stack interpretation to resolve additional jump targets.
    ///
    /// Also computes and stores per-instruction stack snapshots for constant propagation.
    #[instrument(name = "ba", level = "debug", skip_all)]
    pub(crate) fn block_analysis(&mut self) {
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
                    self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP;
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
                    self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP | InstFlags::MULTI_JUMP;
                    self.multi_jump_targets.insert(jump_inst, targets.clone());
                    newly_resolved += 1;
                    trace!(%jump_inst, n_targets = targets.len(), "resolved multi-target jump");
                }
                JumpTarget::Invalid => {
                    self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP | InstFlags::INVALID_JUMP;
                    newly_resolved += 1;
                    trace!(%jump_inst, "resolved invalid jump");
                }
                JumpTarget::Bottom if !has_top_jump => {
                    // Truly unreachable: no unresolved jumps remain, so this
                    // code cannot be reached at runtime. Mark as invalid.
                    self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP | InstFlags::INVALID_JUMP;
                    newly_resolved += 1;
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
        assert_ne!(n, 0, "insts should never be empty");

        let cfg = &mut self.cfg;
        cfg.blocks.clear();
        cfg.inst_to_block.clear();

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

        assert_ne!(cfg.blocks.len(), 0, "should always build at least one block");
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
            if !self.interpret_block(block.insts(), &mut stack_buf) {
                continue;
            }
            let block = &self.cfg.blocks[bid];

            // Discover dynamic-jump target edges from the snapshot recorded above.
            let term_inst = block.terminator();
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
            for &succ in block.succs.iter().chain(&discovered[bid]) {
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
        eprintln!("{bytecode}");
        bytecode
    }

    pub(crate) fn analyze_asm(src: &str) -> Bytecode<'static> {
        analyze_code(crate::parse_asm(src).unwrap())
    }

    pub(crate) fn analyze_asm_with(
        src: &str,
        config: crate::bytecode::AnalysisConfig,
    ) -> Bytecode<'static> {
        analyze_code_with(crate::parse_asm(src).unwrap(), config)
    }

    #[test]
    fn revert_sub_call_storage_oog() {
        analyze_hex(
            "60606040526000357c01000000000000000000000000000000000000000000000000000000009004\
             63ffffffff168063b28175c4146046578063c0406226146052575b6000565b34600057605060765\
             65b005b34600057605c6081565b604051808215151515815260200191505060405180910390f35b\
             600c6000819055505b565b600060896076565b600d600181905550600e600281905550600190505\
             b905600a165627a7a723058202a8a75d7d795b5bcb9042fb18b283daa90b999a11ddec892f54873\
             22",
        );
    }

    #[test]
    fn revert_remote_sub_call_storage_oog() {
        analyze_hex(
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
            analyze_hex(hex);
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
        let bytecode = analyze_asm("
            PUSH1 0x42
            PUSH1 0x01
            ADD
            PUSH1 0x00
            MSTORE
            JUMPDEST
            STOP
        ");
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
        let bytecode = analyze_asm("
            PUSH0
            CALLDATALOAD
            PUSH0
            MSTORE
            STOP
        ");
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
        let bytecode = analyze_asm("
            PUSH1 0x42
            PUSH1 0x01
            ADD
            PUSH0
            MSTORE
            STOP
        ");
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
        let bytecode = analyze_asm("
            PUSH0
            CALLDATALOAD
            STOP
        ");
        assert_eq!(bytecode.const_output(Inst::from_usize(0)), Some(U256::ZERO));
        assert_eq!(bytecode.const_output(Inst::from_usize(1)), None);
    }

    #[test]
    fn snapshots_all_blocks() {
        // Blocks with unresolvable dynamic jumps must still get snapshots from
        // block_analysis_local. Here CALLDATALOAD produces an opaque jump target,
        // so the JUMP stays dynamic, but the block's other instructions should
        // still have snapshots populated.
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            PUSH1 0x42          ; inst 0
            PUSH1 0x01          ; inst 1
            ADD                 ; inst 2: 0x42 + 0x01 = 0x43
            PUSH0               ; inst 3
            CALLDATALOAD        ; inst 4: opaque value
            JUMP                ; inst 5: dynamic, unresolvable
        target:
            JUMPDEST            ; inst 6
            PUSH1 0x01          ; inst 7
            ADD                 ; inst 8
            STOP                ; inst 9
        ");
        // The JUMP at inst 5 should remain dynamic.
        assert!(bytecode.has_dynamic_jumps, "expected unresolved dynamic jump");
        let jump = bytecode.inst(Inst::from_usize(5));
        assert!(jump.is_jump());
        assert!(!jump.flags.contains(InstFlags::STATIC_JUMP));
        // JUMP's TOS operand is Top (CALLDATALOAD result).
        assert_eq!(bytecode.const_operand(Inst::from_usize(5), 0), None);
        // block_analysis_local should still have populated snapshots for ADD at inst 2.
        assert_eq!(bytecode.const_operand(Inst::from_usize(2), 0), Some(U256::from(0x01)));
        assert_eq!(bytecode.const_operand(Inst::from_usize(2), 1), Some(U256::from(0x42)));
        assert_eq!(bytecode.const_output(Inst::from_usize(2)), Some(U256::from(0x43)));
        // The target block also gets snapshots (ADD at inst 8).
        assert_eq!(bytecode.const_operand(Inst::from_usize(8), 0), Some(U256::from(0x01)));
    }

    #[test]
    fn multi_target_jump() {
        // Internal function called from two sites with different return addresses.
        // The return JUMP at the end should resolve to Multi([ret1, ret2]).
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            ; Call site 1.
            PUSH %ret1          ; push return address
            PUSH %func          ; push function entry
            JUMP                ; call function (static: PUSH+JUMP)
        ret1:
            JUMPDEST
            POP                 ; consume function result
            ; Call site 2.
            PUSH %ret2          ; push return address
            PUSH %func          ; push function entry
            JUMP                ; call function (static: PUSH+JUMP)
        ret2:
            JUMPDEST
            POP                 ; consume function result
            STOP
            ; Internal function.
        func:
            JUMPDEST
            PUSH1 0x42          ; push a result
            SWAP1               ; swap result and return address
            JUMP                ; return (dynamic)
        ");

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
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            ; Call site 1.
            PUSH %ret1              ; pc=0
            PUSH %fn_entry          ; pc=2
            JUMP                    ; pc=4: -> fn_entry
        ret1:
            JUMPDEST                ; pc=5
            POP                     ; pc=6: drop result
            ; Call site 2.
            PUSH %ret2              ; pc=7
            PUSH %fn_entry          ; pc=9
            JUMP                    ; pc=11: -> fn_entry
        ret2:
            JUMPDEST                ; pc=12
            POP                     ; pc=13: drop result
            ; Opaque jump: loads target from memory (Top).
            PUSH0                   ; pc=14
            MLOAD                   ; pc=15: -> Top
            JUMP                    ; pc=16: dynamic, target = Top
            INVALID                 ; pc=17
            INVALID                 ; pc=18
            INVALID                 ; pc=19
            ; Internal function: pushes result, swaps with return addr, jumps back.
        fn_entry:
            JUMPDEST                ; pc=20
            PUSH1 0x42              ; pc=21: result
            SWAP1                   ; pc=23: swap result and return addr
            JUMP                    ; pc=24: return (UNSOUND: resolved to Multi)
        ");

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
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            ; Call site 1: direct call to inner.
            PUSH %ret1              ; pc=0
            PUSH %inner             ; pc=2
            JUMP                    ; pc=4: -> inner
        ret1:
            JUMPDEST                ; pc=5
            POP                     ; pc=6: drop result
            ; Call site 2: call through wrapper.
            PUSH %ret2              ; pc=7
            PUSH1 0x42              ; pc=9: an argument
            PUSH %wrapper           ; pc=11
            JUMP                    ; pc=13: -> wrapper
        ret2:
            JUMPDEST                ; pc=14
            POP                     ; pc=15: drop result
            POP                     ; pc=16: drop arg
            STOP                    ; pc=17
            INVALID                 ; pc=18
            ; Wrapper function: calls inner, then returns to caller.
            ; Entry stack: [outer_ret, arg]
        wrapper:
            JUMPDEST                ; pc=19
            PUSH %ret_w             ; pc=20: push return addr for inner
            PUSH %inner             ; pc=22: push inner entry
            JUMP                    ; pc=24: -> inner
            INVALID                 ; pc=25
            INVALID                 ; pc=26
        ret_w:
            JUMPDEST                ; pc=27: inner returned here
            ; Stack: [outer_ret, arg, inner_result]
            POP                     ; pc=28: drop inner_result
            POP                     ; pc=29: drop arg
            JUMP                    ; pc=30: return to caller via outer_ret (dynamic)
            INVALID                 ; pc=31
            ; Inner function: pushes a result and returns.
            ; Entry stack: [..., ret_addr]
        inner:
            JUMPDEST                ; pc=32
            PUSH1 0x42              ; pc=33: push result
            SWAP1                   ; pc=35
            JUMP                    ; pc=36: jump to ret_addr
        ");

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
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            ; Call site 1.
            PUSH %ret1
            PUSH %func
            JUMP
        ret1:
            JUMPDEST
            POP
            ; Call site 2.
            PUSH %ret2
            PUSH %func
            JUMP
        ret2:
            JUMPDEST
            POP
            ; Call site 3.
            PUSH %ret3
            PUSH %func
            JUMP
        ret3:
            JUMPDEST
            POP
            STOP
            ; Internal function.
        func:
            JUMPDEST
            PUSH1 0x42
            SWAP1
            JUMP
        ");

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
        let bytecode = analyze_asm("
            PUSH1 0xAA          ; inst 0
            PUSH1 0xBB          ; inst 1
            DUP2                ; inst 2
            SWAP1               ; inst 3
            PUSH0               ; inst 4
            MSTORE              ; inst 5
            STOP                ; inst 6
        ");
        // DUP2 output should be 0xAA.
        assert_eq!(bytecode.const_output(Inst::from_usize(2)), Some(U256::from(0xAA)));
        // At MSTORE (inst 5): TOS = 0x00, second = 0xBB (swapped back).
        assert_eq!(bytecode.const_operand(Inst::from_usize(5), 0), Some(U256::ZERO));
        assert_eq!(bytecode.const_operand(Inst::from_usize(5), 1), Some(U256::from(0xBB)));
    }

    /// JUMPI with a constant condition should still record both operands.
    #[test]
    fn jumpi_const_condition_snapshot() {
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            PUSH1 0x01              ; inst 0: condition = 1 (always taken)
            PUSH %target            ; inst 1: target
            JUMPI                   ; inst 2: static JUMPI
            STOP                    ; inst 3: fallthrough
        target:
            JUMPDEST                ; inst 4: target
            STOP                    ; inst 5
        ");
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
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            PUSH1 0x42              ; pc=0: initial value
            PUSH1 0x01              ; pc=2: another
        loop_header:
            JUMPDEST                ; loop header (bb1)
            PUSH1 0x01              ; push each iteration
            PUSH %loop_header       ; loop target
            JUMP                    ; back-edge
        ");
        // Should converge without panicking.
        // The JUMP should be static (resolved by adjacent PUSH+JUMP).
        assert!(!bytecode.has_dynamic_jumps);
    }

    /// A single-instruction block (just JUMPDEST as the terminator) used as a
    /// jump target. This tests the edge case where block.len() == 1.
    #[test]
    fn single_inst_block_jump_target() {
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            PUSH %target            ; pc=0
            JUMP                    ; static jump
            INVALID
            INVALID
        target:
            JUMPDEST                ; single-inst block
            STOP
        ");
        assert!(!bytecode.has_dynamic_jumps);
    }

    /// A JUMP with an invalid target (not a JUMPDEST) should be marked invalid.
    #[test]
    fn invalid_jump_target() {
        // The dynamic jump target points to a non-JUMPDEST instruction.
        // The function pushes a constant that isn't a valid JUMPDEST pc.
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            PUSH1 0xFF              ; pc=0: push invalid target
            PUSH %func              ; pc=2: push function entry
            JUMP                    ; pc=4: -> func
            ; Function: swap and jump to the caller-provided target.
        func:
            JUMPDEST
            JUMP                    ; jump to 0xFF (invalid)
        ");
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
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            PUSH1 0x42              ; push same const on both paths
            PUSH0                   ; condition (always false)
            PUSH %then_pc           ; then target
            JUMPI                   ; branch
            ; Else: push same const.
            PUSH %merge
            JUMP                    ; -> merge
            ; Then:
        then_pc:
            JUMPDEST
            PUSH %merge
            JUMP                    ; -> merge
            ; Merge:
        merge:
            JUMPDEST
            PUSH0
            MSTORE                  ; MSTORE(0, 0x42)
            STOP
        ");
        // At the merge MSTORE, the value (operand 1) should still be 0x42
        // since both branches had the same constant on the stack.
        let mstore = bytecode.iter_insts().find(|(_, d)| d.opcode == op::MSTORE).unwrap().0;
        assert_eq!(bytecode.const_operand(mstore, 1), Some(U256::from(0x42)));
    }

    /// Diamond CFG where branches push different constants — the merge should
    /// yield None (Top) for const_operand.
    #[test]
    fn diamond_cfg_different_const() {
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            PUSH0                   ; condition
            PUSH %then_pc
            JUMPI                   ; branch
            ; Else: push 0xAA.
            PUSH1 0xAA
            PUSH %merge
            JUMP                    ; -> merge
            INVALID
            ; Then: push 0xBB.
        then_pc:
            JUMPDEST
            PUSH1 0xBB
            PUSH %merge
            JUMP                    ; -> merge
            ; Merge:
        merge:
            JUMPDEST
            PUSH0
            MSTORE                  ; MSTORE(0, ???)
            STOP
        ");
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
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            ; Entry: branch to opaque path or call site 1.
            PUSH0                       ; pc=0
            CALLDATALOAD                ; pc=1
            PUSH %opaque_path           ; pc=2
            JUMPI                       ; pc=4
            ; Fallthrough: call site 1.
            PUSH %ret1                  ; pc=5
            PUSH %fn_entry              ; pc=7
            JUMP                        ; pc=9
        ret1:
            JUMPDEST                    ; pc=10
            POP                         ; pc=11
            ; Call site 2.
            PUSH %ret2                  ; pc=12
            PUSH %fn_entry              ; pc=14
            JUMP                        ; pc=16
        ret2:
            JUMPDEST                    ; pc=17
            POP                         ; pc=18
            STOP                        ; pc=19
            ; Opaque path (reachable from JUMPI).
        opaque_path:
            JUMPDEST                    ; pc=20
            PUSH0                       ; pc=21
            MLOAD                       ; pc=22: any_ret (Top)
            PUSH %tramp                 ; pc=23
            JUMP                        ; pc=25
            ; Trampoline: bare JUMPDEST + JUMP (is_private_return = true).
        tramp:
            JUMPDEST                    ; pc=26
            JUMP                        ; pc=27: target = any_ret (Top)
            ; Internal function.
        fn_entry:
            JUMPDEST                    ; pc=28
            PUSH1 0x42                  ; pc=29
            SWAP1                       ; pc=31
            JUMP                        ; pc=32: return
        ");

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
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            ; Call site 1.
            PUSH %ret1              ; pc=0
            PUSH %fn_entry          ; pc=2
            JUMP                    ; pc=4
        ret1:
            JUMPDEST                ; pc=5
            POP                     ; pc=6
            ; Call site 2.
            PUSH %ret2              ; pc=7
            PUSH %fn_entry          ; pc=9
            JUMP                    ; pc=11
        ret2:
            JUMPDEST                ; pc=12
            POP                     ; pc=13
            ; Opaque jump.
            PUSH0                   ; pc=14
            MLOAD                   ; pc=15
            JUMP                    ; pc=16: Top
            INVALID                 ; pc=17
            INVALID                 ; pc=18
            INVALID                 ; pc=19
            ; Internal function using JUMPI to return.
        fn_entry:
            JUMPDEST                ; pc=20
            PUSH1 0x42              ; pc=21: result
            SWAP1                   ; pc=23: [result, ret_addr]
            PUSH1 0x01              ; pc=24: condition (always true)
            SWAP1                   ; pc=26: [result, 1, ret_addr]
            JUMPI                   ; pc=27: conditional return (always taken)
            STOP                    ; pc=28: fallthrough (dead)
        ");

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
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            ; Entry: branch to opaque caller or fallthrough.
            PUSH0                       ; pc=0
            CALLDATALOAD                ; pc=1
            PUSH %opaque_caller         ; pc=2
            JUMPI                       ; pc=4
            ; Fallthrough: call site 1.
            PUSH %ret1                  ; pc=5
            PUSH %fn_entry              ; pc=7
            JUMP                        ; pc=9
        ret1:
            JUMPDEST                    ; pc=10
            POP                         ; pc=11
            ; Call site 2.
            PUSH %ret2                  ; pc=12
            PUSH %fn_entry              ; pc=14
            JUMP                        ; pc=16
        ret2:
            JUMPDEST                    ; pc=17
            POP                         ; pc=18
            STOP                        ; pc=19
            ; Opaque third caller: loads return addr from memory (reachable via JUMPI).
        opaque_caller:
            JUMPDEST                    ; pc=20
            PUSH0                       ; pc=21
            MLOAD                       ; pc=22: any_ret
            PUSH %fn_entry              ; pc=23
            JUMP                        ; pc=25: -> fn_entry with arbitrary return addr
            ; Internal function.
        fn_entry:
            JUMPDEST                    ; pc=26
            PUSH1 0x42                  ; pc=27
            SWAP1                       ; pc=29
            JUMP                        ; pc=30: return
        ");

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
        let bytecode = analyze_asm("STOP");
        assert!(!bytecode.has_dynamic_jumps);
    }

    /// A dynamic jump in dead code should not cause `has_dynamic_jumps` to be true.
    #[test]
    fn dead_code_dynamic_jump() {
        #[rustfmt::skip]
        let bytecode = analyze_asm("
            STOP
            PUSH0
            CALLDATALOAD
            JUMP
        ");
        assert!(!bytecode.has_dynamic_jumps);
    }
}

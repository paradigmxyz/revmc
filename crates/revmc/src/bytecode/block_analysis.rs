//! Abstract stack interpretation for resolving dynamic jump targets and constant propagation.
//!
//! This pass builds a basic-block CFG and runs worklist-based abstract interpretation to a
//! fixpoint, propagating abstract stack states across blocks. It resolves jump targets that the
//! simple `static_jump_analysis` (which only looks at adjacent `PUSH + JUMP/JUMPI`) cannot
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
//! - `Conflict` — reserved for future use; currently never produced.
//!
//! ## Soundness
//!
//! When unresolved `Top` jumps remain after the fixpoint, a transitive predecessor analysis
//! invalidates suspect resolutions that may be reachable from those unresolved jumps, ensuring
//! that only sound jump targets are reported as resolved.

use super::{Bytecode, Inst, InstFlags, Interner, U256Idx};
use crate::{FxHashMap, InstData};
use bitvec::vec::BitVec;
use oxc_index::IndexVec;
use revm_bytecode::opcode as op;
use revm_interpreter::instructions::i256::{i256_cmp, i256_div, i256_mod};
use revm_primitives::U256;
use smallvec::SmallVec;
use std::{borrow::Cow, cmp::Ordering, collections::VecDeque, ops::Range};

oxc_index::define_index_type! {
    /// Index into the interned constant-set pool.
    struct ConstSetIdx = u32;
    DISPLAY_FORMAT = "{}";
}

/// Per-instruction snapshot of known-constant operands.
///
/// Index 0 is TOS (first popped / depth 0), index 1 is second from top, etc.
/// Only the instruction's inputs are stored, not the entire stack.
pub(crate) type OperandSnapshot = SmallVec<[Option<U256Idx>; 2]>;

/// Abstract value on the stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AbsValue {
    Const(U256Idx),
    /// Multiple known constant values (interned, sorted, deduplicated).
    ConstSet(ConstSetIdx),
    /// Top value (unknown concrete value).
    Top,
}

impl AbsValue {
    /// Returns the single constant if this is `Const`, or `None` otherwise.
    fn as_const(self) -> Option<U256Idx> {
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

/// Abstract state at the entry of a block.
#[derive(Clone, Debug)]
enum BlockState {
    /// Block has not been reached yet.
    Bottom,
    /// Block has been reached with a known stack state (top-aligned).
    Known(Vec<AbsValue>),
    /// Analysis gives up on this block.
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
    ///
    /// When stack heights differ, the stacks are top-aligned and the shorter one is
    /// bottom-padded with `Top`. This handles the common Solidity dispatch pattern where
    /// the "no selector match" fallthrough leaves an extra item on the stack that the
    /// fallback ignores.
    fn join(&mut self, incoming: &[AbsValue], sets: &mut ConstSetInterner) -> bool {
        match self {
            Self::Bottom => {
                *self = Self::Known(incoming.to_vec());
                true
            }
            Self::Known(existing) => {
                let new_len = existing.len().max(incoming.len());
                let mut changed = false;

                // Cap the abstract stack to prevent unbounded growth in loops where each
                // iteration pushes net items and the top-aligned join keeps padding.
                // EVM stack max is 1024; we use a lower limit since legitimate dispatch
                // patterns only differ by 1–2 items.
                const MAX_STACK_DEPTH: usize = 64;
                if new_len > MAX_STACK_DEPTH {
                    *self = Self::Conflict;
                    return true;
                }

                // Pad existing stack at the bottom with Top if incoming is deeper.
                if existing.len() < new_len {
                    let pad = new_len - existing.len();
                    existing.splice(0..0, std::iter::repeat_n(AbsValue::Top, pad));
                    changed = true;
                }

                // Join element-wise, top-aligned.
                let incoming_pad = new_len - incoming.len();
                for i in 0..new_len {
                    let inc =
                        if i < incoming_pad { AbsValue::Top } else { incoming[i - incoming_pad] };
                    let joined = sets.join(existing[i], inc);
                    if joined != existing[i] {
                        existing[i] = joined;
                        changed = true;
                    }
                }

                changed
            }
            Self::Conflict => false,
        }
    }
}

oxc_index::define_index_type! {
    /// A block index in the CFG.
    struct Block = u32;
    DISPLAY_FORMAT = "{}";
}

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

/// Call-string context for k-CFA analysis (k=2).
///
/// Each context records the last `k` call-site instruction indices. This lets us
/// keep separate abstract states for the same block reached via different call chains,
/// preserving return addresses that would otherwise merge to Top.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct Context {
    /// Call-site instruction indices, oldest first. Length ≤ K.
    sites: SmallVec<[Inst; 2]>,
}

impl Context {
    const K: usize = 2;

    /// Push a call site and truncate to at most K entries.
    fn push(&self, call_inst: Inst) -> Self {
        let mut sites = self.sites.clone();
        sites.push(call_inst);
        if sites.len() > Self::K {
            sites.remove(0);
        }
        Self { sites }
    }

    /// Pop the most recent call site (return edge).
    fn pop(&self) -> Self {
        let mut sites = self.sites.clone();
        sites.pop();
        Self { sites }
    }

    /// Returns the most recent call site, if any.
    fn top(&self) -> Option<Inst> {
        self.sites.last().copied()
    }
}

impl std::fmt::Display for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, s) in self.sites.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{s}")?;
        }
        write!(f, "]")
    }
}

/// Information about a detected internal-function call site.
#[derive(Clone)]
struct CallSiteInfo {
    /// The block that the callee will jump to (the callee entry block).
    callee_entry: Block,
    /// The block at the return address (where the caller resumes).
    return_block: Block,
}

/// A basic block in the CFG.
struct BlockData {
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
    Const(Inst),
    /// Multiple known constant target instruction indices.
    Multi(SmallVec<[Inst; 4]>),
    /// Known constant but invalid target.
    Invalid,
    /// Unknown target.
    Top,
}

/// CFG for abstract interpretation.
struct Cfg {
    blocks: IndexVec<Block, BlockData>,
    /// Maps instruction index to block ID. `None` for dead-code instructions.
    inst_to_block: IndexVec<Inst, Option<Block>>,
}

impl Bytecode<'_> {
    /// Runs abstract stack interpretation to resolve additional jump targets.
    ///
    /// Also computes and stores per-instruction stack snapshots for constant propagation.
    #[instrument(name = "ba", level = "debug", skip_all)]
    pub(crate) fn block_analysis(&mut self) {
        let cfg = self.build_cfg();
        if cfg.blocks.is_empty() {
            return;
        }

        let mut snapshots: IndexVec<Inst, OperandSnapshot> =
            IndexVec::from_vec(vec![SmallVec::new(); self.insts.len()]);
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
            // Skip if already resolved by static_jump_analysis.
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

        debug!(newly_resolved, "block_analysis complete");

        // Recompute dynamic jumps flag.
        self.has_dynamic_jumps = self
            .insts
            .iter()
            .any(|inst| inst.is_jump() && !inst.flags.contains(InstFlags::STATIC_JUMP));
    }

    /// Build a basic-block CFG from the instruction list.
    fn build_cfg(&self) -> Cfg {
        let n = self.insts.len();
        if n == 0 {
            return Cfg { blocks: IndexVec::new(), inst_to_block: IndexVec::new() };
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

        // Build blocks.
        let mut blocks: IndexVec<Block, BlockData> = IndexVec::new();
        let mut inst_to_block: IndexVec<Inst, Option<Block>> = IndexVec::from_vec(vec![None; n]);
        let mut current_start = None;

        let mut finish_block = |start: usize, end: usize| {
            let range = start..end;
            let dead = range.clone().all(|j| self.insts.raw[j].is_dead_code());
            let bid = blocks.push(BlockData {
                insts: range.clone(),
                preds: SmallVec::new(),
                succs: SmallVec::new(),
                dead,
            });
            for j in range {
                if !self.insts.raw[j].is_dead_code() {
                    inst_to_block.raw[j] = Some(bid);
                }
            }
        };

        for i in 0..n {
            if self.insts.raw[i].is_dead_code() {
                continue;
            }

            if is_leader[i] || current_start.is_none() {
                if let Some(start) = current_start {
                    finish_block(start, i);
                }
                current_start = Some(i);
            }
        }

        // Close the last block.
        if let Some(start) = current_start {
            finish_block(start, n);
        }

        // Build edges based on known control flow.
        for bid in blocks.indices() {
            if blocks[bid].dead {
                continue;
            }
            let term_idx = blocks[bid].insts.end - 1;
            let term = &self.insts.raw[term_idx];

            // Fallthrough edge: if the terminator doesn't unconditionally branch/diverge.
            let has_fallthrough = !term.is_diverging() && (term.opcode != op::JUMP);
            if has_fallthrough
                && blocks[bid].insts.end < n
                && let Some(next_block) = inst_to_block.raw[blocks[bid].insts.end]
            {
                blocks[next_block].preds.push(bid);
                blocks[bid].succs.push(next_block);
            }

            // Static jump edges.
            if term.is_static_jump() && !term.flags.contains(InstFlags::INVALID_JUMP) {
                let target_inst = term.data as usize;
                if let Some(target_block) = inst_to_block.raw[target_inst] {
                    blocks[target_block].preds.push(bid);
                    blocks[bid].succs.push(target_block);
                }
            }
        }

        Cfg { blocks, inst_to_block }
    }

    /// Detect internal-function call patterns: blocks ending in a static JUMP where
    /// the preceding instructions are PUSH retaddr; PUSH func; JUMP. Both targets must
    /// be valid JUMPDESTs. Returns a per-block map of call-site info.
    fn precompute_call_sites(&self, cfg: &Cfg) -> IndexVec<Block, Option<CallSiteInfo>> {
        let num_blocks = cfg.blocks.len();
        let mut call_sites: IndexVec<Block, Option<CallSiteInfo>> =
            IndexVec::from_vec(vec![None; num_blocks]);

        for bid in cfg.blocks.indices() {
            let block = &cfg.blocks[bid];
            if block.dead {
                continue;
            }

            let term_idx = block.insts.end - 1;
            let term = &self.insts.raw[term_idx];

            // Must be an unconditional JUMP with STATIC_JUMP (from static_jump_analysis).
            if term.opcode != op::JUMP || !term.flags.contains(InstFlags::STATIC_JUMP) {
                continue;
            }
            if term.flags.contains(InstFlags::INVALID_JUMP) {
                continue;
            }

            // Walk backwards past dead code and SKIP_LOGIC to find the semantic instructions
            // before the JUMP. We need two PUSHes: PUSH retaddr, PUSH func, JUMP.
            // The PUSH func is marked SKIP_LOGIC by static_jump_analysis.
            let mut semantic_before: SmallVec<[usize; 4]> = SmallVec::new();
            for idx in (block.insts.start..term_idx).rev() {
                let inst = &self.insts.raw[idx];
                if inst.is_dead_code() {
                    continue;
                }
                semantic_before.push(idx);
                if semantic_before.len() >= 2 {
                    break;
                }
            }
            // semantic_before[0] = closest to JUMP, semantic_before[1] = the one before that.
            if semantic_before.len() < 2 {
                continue;
            }

            let push_func_idx = semantic_before[0];
            let push_ret_idx = semantic_before[1];

            let push_func = &self.insts.raw[push_func_idx];
            let push_ret = &self.insts.raw[push_ret_idx];

            // Both must be PUSH instructions.
            if !push_func.is_push() || !push_ret.is_push() {
                continue;
            }
            // PUSH func should be SKIP_LOGIC (consumed by the static JUMP).
            if !push_func.flags.contains(InstFlags::SKIP_LOGIC) {
                continue;
            }

            // Resolve the return address.
            let ret_imm = if push_ret.opcode == op::PUSH0 {
                Some(&[] as &[u8])
            } else {
                self.get_imm(push_ret)
            };
            let Some(ret_imm) = ret_imm else {
                continue;
            };
            let ret_pc = if ret_imm.is_empty() {
                0usize
            } else if ret_imm.len() <= std::mem::size_of::<usize>() {
                let mut padded = [0u8; std::mem::size_of::<usize>()];
                padded[std::mem::size_of::<usize>() - ret_imm.len()..].copy_from_slice(ret_imm);
                usize::from_be_bytes(padded)
            } else {
                continue;
            };
            if !self.is_valid_jump(ret_pc) {
                continue;
            }

            let ret_inst = self.pc_to_inst(ret_pc);
            let Some(return_block) = cfg.inst_to_block[ret_inst] else {
                continue;
            };

            // The callee entry block is the static jump target.
            let callee_inst = Inst::from_usize(term.data as usize);
            let Some(callee_entry) = cfg.inst_to_block[callee_inst] else {
                continue;
            };

            // Use the JUMP instruction as the call-site identifier.
            call_sites[bid] = Some(CallSiteInfo { callee_entry, return_block });
        }

        call_sites
    }

    /// Run worklist-based abstract interpretation over the CFG.
    ///
    /// Returns a list of (jump_inst, resolved_target) pairs and the count of resolvable jumps.
    /// Stack snapshots are recorded into `snapshots` during the fixpoint.
    fn run_abstract_interp(
        &self,
        cfg: &Cfg,
        snapshots: &mut IndexVec<Inst, OperandSnapshot>,
    ) -> (Vec<(Inst, JumpTarget)>, usize) {
        let num_blocks = cfg.blocks.len();

        // Collect unresolved jumps.
        let mut jump_insts: Vec<Inst> = Vec::new();
        for (i, inst) in self.insts.iter_enumerated() {
            if inst.is_jump() && !inst.flags.contains(InstFlags::STATIC_JUMP) {
                jump_insts.push(i);
            }
        }

        // Always run the fixpoint to propagate stack states through the CFG.
        // Snapshots are recorded during interpretation — last write wins (= converged state).
        let mut const_sets = ConstSetInterner::new();
        let call_sites = self.precompute_call_sites(cfg);
        let (ctx_states, discovered_edges) =
            self.run_fixpoint(cfg, &call_sites, snapshots, &mut const_sets);

        if jump_insts.is_empty() {
            return (Vec::new(), 0);
        }

        // Merge per-context states into a single state per block for jump resolution
        // and soundness checking.
        let mut block_states: IndexVec<Block, BlockState> =
            IndexVec::from_vec(vec![BlockState::Bottom; num_blocks]);
        for bid in cfg.blocks.indices() {
            if let Some(ctx_map) = ctx_states.get(bid) {
                for state in ctx_map.values() {
                    match state {
                        BlockState::Known(incoming) => {
                            block_states[bid].join(incoming, &mut const_sets);
                        }
                        BlockState::Conflict => {
                            block_states[bid].conflict();
                        }
                        BlockState::Bottom => {}
                    }
                }
            }
        }

        // Flatten per-context discovered edges into a per-block set for soundness analysis.
        let mut discovered_edges_flat: IndexVec<Block, SmallVec<[Block; 2]>> =
            IndexVec::from_vec(vec![SmallVec::new(); num_blocks]);
        for (&(bid, _), targets) in &discovered_edges {
            for &tgt in targets {
                if !discovered_edges_flat[bid].contains(&tgt) {
                    discovered_edges_flat[bid].push(tgt);
                }
            }
        }

        // After convergence, resolve each dynamic jump using per-context states.
        // If all contexts agree on the same target(s), resolve; if any disagrees, widen.
        let mut jump_targets: Vec<(Inst, JumpTarget)> = Vec::new();
        let mut has_top_jump = false;
        for &jump_inst in &jump_insts {
            let target = match cfg.inst_to_block[jump_inst] {
                None => JumpTarget::Bottom,
                Some(bid) => {
                    let ctx_map = &ctx_states[bid];
                    if ctx_map.is_empty() {
                        trace!(%jump_inst, pc = self.insts[jump_inst].pc, "jump in unreached block");
                        JumpTarget::Bottom
                    } else {
                        // Collect resolved targets across all contexts.
                        let mut all_target_insts: SmallVec<[Inst; 4]> = SmallVec::new();
                        let mut any_top = false;
                        let mut any_invalid = false;
                        let mut all_bottom = true;

                        for state in ctx_map.values() {
                            match state {
                                BlockState::Bottom => continue,
                                BlockState::Conflict => {
                                    any_top = true;
                                    all_bottom = false;
                                }
                                BlockState::Known(input) => {
                                    all_bottom = false;
                                    match self.jump_operand(&cfg.blocks[bid], input) {
                                        Some(AbsValue::Const(idx)) => {
                                            let val = *self.u256_interner.borrow().get(idx);
                                            match usize::try_from(val) {
                                                Ok(pc) if self.is_valid_jump(pc) => {
                                                    let ti = self.pc_to_inst(pc);
                                                    if !all_target_insts.contains(&ti) {
                                                        all_target_insts.push(ti);
                                                    }
                                                }
                                                _ => any_invalid = true,
                                            }
                                        }
                                        Some(AbsValue::ConstSet(set_idx)) => {
                                            let consts = const_sets.get(set_idx);
                                            let interner = self.u256_interner.borrow();
                                            for &idx in consts {
                                                let val = *interner.get(idx);
                                                match usize::try_from(val) {
                                                    Ok(pc) if self.is_valid_jump(pc) => {
                                                        let ti = self.pc_to_inst(pc);
                                                        if !all_target_insts.contains(&ti) {
                                                            all_target_insts.push(ti);
                                                        }
                                                    }
                                                    _ => any_invalid = true,
                                                }
                                            }
                                        }
                                        _ => any_top = true,
                                    }
                                }
                            }
                        }

                        if all_bottom {
                            JumpTarget::Bottom
                        } else if any_top {
                            trace!(
                                inst = %jump_inst,
                                pc = self.insts[jump_inst].pc,
                                "unresolved jump (Top in some context)",
                            );
                            JumpTarget::Top
                        } else if any_invalid && all_target_insts.is_empty() {
                            JumpTarget::Invalid
                        } else if any_invalid {
                            JumpTarget::Top
                        } else {
                            match all_target_insts.len() {
                                0 => JumpTarget::Bottom,
                                1 => JumpTarget::Const(all_target_insts[0]),
                                _ => JumpTarget::Multi(all_target_insts),
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
            let mut disc_preds: IndexVec<Block, SmallVec<[Block; 2]>> =
                IndexVec::from_vec(vec![SmallVec::new(); num_blocks]);
            for (src, targets) in discovered_edges_flat.iter_enumerated() {
                for &tgt in targets {
                    disc_preds[tgt].push(src);
                }
            }

            // Precompute: for each block, whether it has a suspect (Bottom + JUMPDEST) predecessor.
            let mut suspect: BitVec = BitVec::repeat(false, num_blocks);
            for bid in cfg.blocks.indices() {
                let bi = bid.index();
                let has_suspect =
                    cfg.blocks[bid].preds.iter().chain(disc_preds[bid].iter()).any(|&pred| {
                        if !matches!(block_states[pred], BlockState::Bottom) {
                            return false;
                        }
                        self.insts.raw[cfg.blocks[pred].insts.start].is_jumpdest()
                    });
                if has_suspect {
                    suspect.set(bi, true);
                }
            }

            // Propagate suspect flag forward through the CFG: if a block is suspect,
            // all its successors (static + discovered) are also suspect.
            let mut propagate = Worklist::new(num_blocks);
            for bid in cfg.blocks.indices() {
                if suspect[bid.index()] {
                    propagate.push(bid);
                }
            }
            while let Some(bid) = propagate.pop() {
                for &succ in cfg.blocks[bid].succs.iter().chain(discovered_edges_flat[bid].iter()) {
                    let si = succ.index();
                    if !suspect[si] {
                        suspect.set(si, true);
                        propagate.push(succ);
                    }
                }
            }

            for (inst, target) in jump_targets.iter_mut() {
                if !matches!(
                    target,
                    JumpTarget::Const(_) | JumpTarget::Multi(_) | JumpTarget::Invalid
                ) {
                    continue;
                }
                if let Some(bid) = cfg.inst_to_block[*inst]
                    && suspect[bid.index()]
                {
                    *target = JumpTarget::Top;
                }
            }
        }

        let count = jump_targets
            .iter()
            .filter(|(_, t)| {
                matches!(t, JumpTarget::Const(_) | JumpTarget::Multi(_) | JumpTarget::Invalid)
            })
            .count();

        (jump_targets, count)
    }

    /// Run a worklist-based fixpoint to compute per-context abstract block states.
    ///
    /// Uses k-CFA (k=2) call-string context sensitivity: each block is analyzed
    /// separately for each call-string context that reaches it. This prevents
    /// return addresses from different callers from merging into Top.
    ///
    /// Returns per-context block states and per-context discovered edges.
    /// Stack snapshots are recorded into `snapshots` during each block interpretation.
    #[allow(clippy::type_complexity)]
    fn run_fixpoint(
        &self,
        cfg: &Cfg,
        call_sites: &IndexVec<Block, Option<CallSiteInfo>>,
        snapshots: &mut IndexVec<Inst, OperandSnapshot>,
        const_sets: &mut ConstSetInterner,
    ) -> (
        IndexVec<Block, FxHashMap<Context, BlockState>>,
        FxHashMap<(Block, Context), SmallVec<[Block; 2]>>,
    ) {
        let num_blocks = cfg.blocks.len();

        // Build a reverse map: callee_entry -> list of CallSiteInfos that call it.
        let mut return_block_to_call_site: FxHashMap<Block, SmallVec<[Block; 2]>> =
            FxHashMap::default();
        for (bid, cs) in call_sites.iter_enumerated() {
            if let Some(info) = cs {
                return_block_to_call_site.entry(info.return_block).or_default().push(bid);
            }
        }

        // Per-context block states.
        let mut ctx_states: IndexVec<Block, FxHashMap<Context, BlockState>> =
            IndexVec::from_vec(vec![FxHashMap::default(); num_blocks]);

        // Per-context discovered edges.
        let mut discovered_edges: FxHashMap<(Block, Context), SmallVec<[Block; 2]>> =
            FxHashMap::default();

        const MAX_CONTEXTS_PER_BLOCK: usize = 16;

        // FIFO worklist of (Block, Context) pairs with dedup.
        let empty_ctx = Context::default();
        let mut wl_queue: VecDeque<(Block, Context)> = VecDeque::new();
        let mut wl_set: FxHashMap<Block, SmallVec<[Context; 2]>> = FxHashMap::default();

        // Initialize entry block with empty context and empty stack.
        ctx_states[Block::from_usize(0)].insert(empty_ctx.clone(), BlockState::Known(Vec::new()));
        wl_queue.push_back((Block::from_usize(0), empty_ctx.clone()));
        wl_set.entry(Block::from_usize(0)).or_default().push(empty_ctx);

        let max_iterations = num_blocks * MAX_CONTEXTS_PER_BLOCK * 8;
        let mut iterations = 0;
        let mut converged = true;

        while let Some((bid, ctx)) = wl_queue.pop_front() {
            // Remove from in-queue set.
            if let Some(ctxs) = wl_set.get_mut(&bid)
                && let Some(pos) = ctxs.iter().position(|c| c == &ctx)
            {
                ctxs.swap_remove(pos);
            }

            iterations += 1;
            if iterations > max_iterations {
                converged = false;
                break;
            }

            let input = match ctx_states[bid].get(&ctx) {
                Some(BlockState::Known(s)) => s.clone(),
                Some(BlockState::Bottom) | None => continue,
                Some(BlockState::Conflict) => {
                    // Propagate Conflict to all successors.
                    let disc_key = (bid, ctx.clone());
                    let disc = discovered_edges.get(&disc_key).cloned().unwrap_or_default();
                    for &succ in cfg.blocks[bid].succs.iter().chain(disc.iter()) {
                        let succ_ctx = self.transfer_context(
                            bid,
                            succ,
                            &ctx,
                            call_sites,
                            &return_block_to_call_site,
                            cfg,
                        );
                        let entry =
                            ctx_states[succ].entry(succ_ctx.clone()).or_insert(BlockState::Bottom);
                        if entry.conflict() {
                            trace!(
                                from = %bid, to = %succ,
                                to_pc = self.insts.raw[cfg.blocks[succ].insts.start].pc,
                                %ctx,
                                "propagating conflict",
                            );
                            self.wl_push(&mut wl_queue, &mut wl_set, succ, succ_ctx);
                        }
                    }
                    continue;
                }
            };

            let block = &cfg.blocks[bid];
            if block.dead {
                continue;
            }

            let Some(output) = self.interpret_block(block.insts.clone(), &input, Some(snapshots))
            else {
                continue;
            };

            // For dynamic jumps, discover target edges (context-sensitive).
            let term = &self.insts.raw[block.insts.end - 1];
            if term.is_jump()
                && !term.flags.contains(InstFlags::STATIC_JUMP)
                && let Some(operand) = self.jump_operand(block, &input)
            {
                let target_pcs: SmallVec<[usize; 4]> = match operand {
                    AbsValue::Const(idx) => {
                        let val = *self.u256_interner.borrow().get(idx);
                        usize::try_from(val).ok().into_iter().collect()
                    }
                    AbsValue::ConstSet(set_idx) => {
                        let interner = self.u256_interner.borrow();
                        const_sets
                            .get(set_idx)
                            .iter()
                            .filter_map(|&idx| usize::try_from(*interner.get(idx)).ok())
                            .collect()
                    }
                    AbsValue::Top => SmallVec::new(),
                };
                let disc_key = (bid, ctx.clone());
                for target_pc in target_pcs {
                    if !self.is_valid_jump(target_pc) {
                        continue;
                    }
                    let ti = self.pc_to_inst(target_pc);
                    if let Some(tb) = cfg.inst_to_block[ti] {
                        let edges = discovered_edges.entry(disc_key.clone()).or_default();
                        if !edges.contains(&tb) {
                            edges.push(tb);
                        }
                    }
                }
            }

            // Propagate to static CFG successors and discovered dynamic-jump targets.
            let disc_key = (bid, ctx.clone());
            let disc = discovered_edges.get(&disc_key).cloned().unwrap_or_default();
            for &succ in block.succs.iter().chain(disc.iter()) {
                let succ_ctx = self.transfer_context(
                    bid,
                    succ,
                    &ctx,
                    call_sites,
                    &return_block_to_call_site,
                    cfg,
                );

                // Cap contexts per block.
                let succ_map = &mut ctx_states[succ];
                if !succ_map.contains_key(&succ_ctx) && succ_map.len() >= MAX_CONTEXTS_PER_BLOCK {
                    // Merge into existing context with fewest sites (most general).
                    let merge_ctx =
                        succ_map.keys().min_by_key(|c| c.sites.len()).cloned().unwrap_or_default();
                    let entry = succ_map.entry(merge_ctx.clone()).or_insert(BlockState::Bottom);
                    if entry.join(&output, const_sets) {
                        self.wl_push(&mut wl_queue, &mut wl_set, succ, merge_ctx);
                    }
                } else {
                    let entry = succ_map.entry(succ_ctx.clone()).or_insert(BlockState::Bottom);
                    if entry.join(&output, const_sets) {
                        self.wl_push(&mut wl_queue, &mut wl_set, succ, succ_ctx);
                    }
                }
            }
        }

        debug!(
            "{msg} after {iterations} iterations (max={max_iterations})",
            msg = if converged { "converged" } else { "did not converge" },
        );

        (ctx_states, discovered_edges)
    }

    /// Compute the successor context when propagating from `src` to `dst`.
    fn transfer_context(
        &self,
        src: Block,
        dst: Block,
        ctx: &Context,
        call_sites: &IndexVec<Block, Option<CallSiteInfo>>,
        return_block_to_call_site: &FxHashMap<Block, SmallVec<[Block; 2]>>,
        cfg: &Cfg,
    ) -> Context {
        // Check if src is a call site and dst is the callee entry.
        if let Some(info) = &call_sites[src]
            && dst == info.callee_entry
        {
            let call_inst_idx = cfg.blocks[src].insts.end - 1;
            return ctx.push(Inst::from_usize(call_inst_idx));
        }

        // Check if this is a return edge: dst is the return_block of the top call site.
        if let Some(top_call) = ctx.top()
            && let Some(top_bid) = cfg.inst_to_block[top_call]
            && let Some(info) = &call_sites[top_bid]
            && dst == info.return_block
        {
            return ctx.pop();
        }

        // Also check: if dst is a known return block from *any* call site whose caller
        // matches the top of the context. This handles discovered dynamic edges.
        if let Some(callers) = return_block_to_call_site.get(&dst)
            && let Some(top_call) = ctx.top()
        {
            for &caller_bid in callers {
                let call_inst_idx = cfg.blocks[caller_bid].insts.end - 1;
                if Inst::from_usize(call_inst_idx) == top_call {
                    return ctx.pop();
                }
            }
        }

        // Normal edge: context unchanged.
        ctx.clone()
    }

    /// Push a (Block, Context) pair onto the worklist with dedup.
    fn wl_push(
        &self,
        queue: &mut VecDeque<(Block, Context)>,
        set: &mut FxHashMap<Block, SmallVec<[Context; 2]>>,
        bid: Block,
        ctx: Context,
    ) {
        let ctxs = set.entry(bid).or_default();
        if !ctxs.contains(&ctx) {
            ctxs.push(ctx.clone());
            queue.push_back((bid, ctx));
        }
    }

    /// Interpret a sequence of instructions on the abstract stack.
    /// Returns the abstract stack state after the last instruction, or `None` on conflict.
    ///
    /// If `snapshots` is provided, records the abstract stack state *before* each instruction
    /// into `snapshots[inst_index]`.
    fn interpret_block<'a>(
        &self,
        range: Range<usize>,
        input: &'a [AbsValue],
        mut snapshots: Option<&mut IndexVec<Inst, OperandSnapshot>>,
    ) -> Option<Cow<'a, [AbsValue]>> {
        let mut stack = Cow::Borrowed(input);

        for i in range {
            let inst = &self.insts.raw[i];
            if inst.is_dead_code() {
                continue;
            }

            // Instructions marked SKIP_LOGIC (the PUSH in a PUSH+JUMP pair) are no-ops
            // for abstract interpretation — the value is consumed by the already-resolved jump.
            if inst.flags.contains(InstFlags::SKIP_LOGIC) {
                continue;
            }

            let stack = stack.to_mut();

            // Record pre-instruction operand snapshot: only the top `inp` values.
            if let Some(snaps) = &mut snapshots {
                let (inp, _) = inst.stack_io();
                let inp = inp as usize;
                let len = stack.len();
                let start = len.saturating_sub(inp);
                snaps.raw[i] = stack[start..].iter().rev().map(|v| v.as_const()).collect();
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
                    let result = if out > 0 {
                        self.try_const_fold(inst, &stack[stack.len() - inp..])
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
        }

        Some(stack)
    }

    /// Returns the abstract value of the jump operand (TOS before the terminator) for a block.
    fn jump_operand(&self, block: &BlockData, input: &[AbsValue]) -> Option<AbsValue> {
        if block.insts.len() == 1 {
            input.last().copied()
        } else {
            self.interpret_block(block.insts.start..block.insts.end - 1, input, None)?
                .last()
                .copied()
        }
    }

    /// Try to constant-fold in instruction.
    fn try_const_fold(&self, inst: &InstData, inputs: &[AbsValue]) -> Option<AbsValue> {
        let opcode = inst.opcode;
        let mut interner = self.u256_interner.borrow_mut();
        let result = match opcode {
            // 0 -> 1
            op::CODESIZE => U256::from(self.code.len()),
            op::PC => U256::from(inst.pc),

            // 1 -> 1
            op::ISZERO | op::NOT | op::CLZ => {
                let &[AbsValue::Const(ai)] = inputs else {
                    return None;
                };
                let a = *interner.get(ai);
                match opcode {
                    op::ISZERO => U256::from(a.is_zero()),
                    op::NOT => !a,
                    op::CLZ => U256::from(a.leading_zeros()),
                    _ => unreachable!(),
                }
            }

            // 2 -> 1
            op::ADD
            | op::MUL
            | op::SUB
            | op::DIV
            | op::SDIV
            | op::MOD
            | op::SMOD
            | op::EXP
            | op::SIGNEXTEND
            | op::LT
            | op::GT
            | op::SLT
            | op::SGT
            | op::EQ
            | op::AND
            | op::OR
            | op::XOR
            | op::BYTE
            | op::SHL
            | op::SHR
            | op::SAR => {
                let &[AbsValue::Const(bi), AbsValue::Const(ai)] = inputs else {
                    return None;
                };
                let a = *interner.get(ai);
                let b = *interner.get(bi);
                match opcode {
                    op::ADD => a.wrapping_add(b),
                    op::MUL => a.wrapping_mul(b),
                    op::SUB => a.wrapping_sub(b),
                    op::DIV => {
                        if !b.is_zero() {
                            a.wrapping_div(b)
                        } else {
                            U256::ZERO
                        }
                    }
                    op::SDIV => i256_div(a, b),
                    op::MOD => {
                        if !b.is_zero() {
                            a.wrapping_rem(b)
                        } else {
                            U256::ZERO
                        }
                    }
                    op::SMOD => i256_mod(a, b),
                    op::EXP => a.pow(b),
                    op::SIGNEXTEND => {
                        if a < U256::from(31) {
                            let ext = a.as_limbs()[0];
                            let bit_index = (8 * ext + 7) as usize;
                            let bit = b.bit(bit_index);
                            let mask = (U256::from(1) << bit_index) - U256::from(1);
                            if bit { b | !mask } else { b & mask }
                        } else {
                            b
                        }
                    }
                    op::LT => U256::from(a < b),
                    op::GT => U256::from(a > b),
                    op::SLT => U256::from(i256_cmp(&a, &b) == Ordering::Less),
                    op::SGT => U256::from(i256_cmp(&a, &b) == Ordering::Greater),
                    op::EQ => U256::from(a == b),
                    op::AND => a & b,
                    op::OR => a | b,
                    op::XOR => a ^ b,
                    op::BYTE => {
                        let i = a.saturating_to::<usize>();
                        if i < 32 { U256::from(b.byte(31 - i)) } else { U256::ZERO }
                    }
                    op::SHL => {
                        let shift = a.saturating_to::<usize>();
                        if shift < 256 { b << shift } else { U256::ZERO }
                    }
                    op::SHR => {
                        let shift = a.saturating_to::<usize>();
                        if shift < 256 { b >> shift } else { U256::ZERO }
                    }
                    op::SAR => {
                        let shift = a.saturating_to::<usize>();
                        if shift < 256 {
                            b.arithmetic_shr(shift)
                        } else if b.bit(255) {
                            U256::MAX
                        } else {
                            U256::ZERO
                        }
                    }
                    _ => unreachable!(),
                }
            }

            // 3 -> 1
            op::ADDMOD | op::MULMOD => {
                let &[AbsValue::Const(ci), AbsValue::Const(bi), AbsValue::Const(ai)] = inputs
                else {
                    return None;
                };
                let a = *interner.get(ai);
                let b = *interner.get(bi);
                let n = *interner.get(ci);
                match opcode {
                    op::ADDMOD => a.add_mod(b, n),
                    op::MULMOD => a.mul_mod(b, n),
                    _ => unreachable!(),
                }
            }

            _ => return None,
        };
        Some(AbsValue::Const(interner.intern(result)))
    }
}

#[cfg(test)]
mod tests {
    use super::{super::Inst, *};
    use revm_primitives::{hardfork::SpecId, hex};

    fn analyze_hex(hex: &str) -> Bytecode<'static> {
        let code = hex::decode(hex.trim()).unwrap();
        analyze_code(code)
    }

    pub(super) fn analyze_code(code: Vec<u8>) -> Bytecode<'static> {
        let code = &*Box::leak(code.into_boxed_slice());
        eprintln!("{}", hex::encode(code));
        let mut bytecode = Bytecode::new(code, SpecId::CANCUN);
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

        // The inner return JUMP should resolve to Multi([ret1, ret_w]).
        let inner_return = bytecode
            .iter_insts()
            .find(|(_, d)| d.is_jump() && d.flags.contains(InstFlags::MULTI_JUMP));
        assert!(inner_return.is_some(), "expected inner return to be multi-target");

        // With k=2 call-string context sensitivity, the wrapper return JUMP (pc=30)
        // now resolves: the outer return address (ret2=14) is preserved because
        // the inner function is analyzed separately for each calling context.
        assert!(!bytecode.has_dynamic_jumps, "expected all jumps to be resolved with k-CFA");
    }

    #[test]
    fn debug_erc20_calls() {
        let code = revm_primitives::hex::decode(
            include_str!("../../../../data/erc20_transfer.rt.hex").trim(),
        )
        .unwrap();
        let code = Box::leak(code.into_boxed_slice());
        let mut bytecode = Bytecode::new(code, SpecId::CANCUN);
        bytecode.analyze().unwrap();

        eprintln!("\n=== Dynamic JUMPs ===");
        for (i, inst) in bytecode.iter_insts() {
            if inst.is_jump() && !inst.is_static_jump() {
                eprintln!("  inst={i} pc={} op={}", inst.pc, inst.to_op());
            }
        }

        eprintln!("\n=== Static unconditional JUMPs (candidates) ===");
        for (i, inst) in bytecode.iter_insts() {
            if inst.opcode == op::JUMP
                && inst.is_static_jump()
                && !inst.flags.contains(InstFlags::INVALID_JUMP)
            {
                let target = Inst::from_usize(inst.data as usize);
                let target_pc = bytecode.inst(target).pc;
                // Check if the instruction after this JUMP is a JUMPDEST (return point).
                let mut next_jd_pc = None;
                for j in (i.index() + 1)..bytecode.iter_all_insts().len() {
                    let d = bytecode.inst(Inst::from_usize(j));
                    if d.is_dead_code() {
                        continue;
                    }
                    if d.is_jumpdest() {
                        next_jd_pc = Some(d.pc);
                    }
                    break;
                }
                // Walk backwards in same block looking for a PUSH of next_jd_pc.
                let mut found_ret = false;
                let mut dist = 0u32;
                if let Some(ret_pc) = next_jd_pc {
                    for prev in (0..i.index()).rev() {
                        let d = bytecode.inst(Inst::from_usize(prev));
                        if d.is_dead_code() || d.flags.contains(InstFlags::SKIP_LOGIC) {
                            continue;
                        }
                        if d.is_branching() || d.is_jumpdest() {
                            break;
                        }
                        dist += 1;
                        if d.is_push() {
                            let val = if d.opcode == op::PUSH0 {
                                0u32
                            } else if let Some(b) = bytecode.get_imm(d) {
                                let mut p = [0u8; 4];
                                let l = b.len().min(4);
                                p[4 - l..].copy_from_slice(&b[b.len() - l..]);
                                u32::from_be_bytes(p)
                            } else {
                                continue;
                            };
                            if val == ret_pc {
                                found_ret = true;
                                break;
                            }
                        }
                    }
                }
                eprintln!(
                    "  inst={i} pc={} -> callee_pc={target_pc} \
                     next_jd={next_jd_pc:?} ret_push_found={found_ret} dist={dist}",
                    inst.pc
                );
            }
        }
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

#[cfg(test)]
mod tests_const_fold {
    use super::{tests::*, *};

    /// Builds bytecode that pushes operands, executes `opcode`, sinks the result
    /// with `PUSH0 MSTORE STOP`, analyzes it, and returns the folded result.
    ///
    /// Operands are in natural order matching the yellow paper specification:
    /// for `SUB(a, b)` = `a - b`, pass `&[a, b]` (first popped = first element).
    fn const_fold(opcode: u8, operands: &[U256]) -> Option<U256> {
        let mut code = Vec::new();
        let mut num_insts = 0usize;
        // Push in reverse so that operands[0] ends up on TOS.
        for v in operands.iter().rev() {
            let bytes = v.to_be_bytes::<32>();
            let lz = bytes.iter().position(|&b| b != 0).unwrap_or(32);
            if lz == 32 {
                code.push(op::PUSH0);
            } else {
                let len = 32 - lz;
                code.push(op::PUSH0 + len as u8);
                code.extend_from_slice(&bytes[lz..]);
            }
            num_insts += 1;
        }
        code.push(opcode);
        num_insts += 1;
        // Sink: PUSH0 + MSTORE + STOP to consume the result.
        let mstore_inst = num_insts + 1;
        code.extend_from_slice(&[op::PUSH0, op::MSTORE, op::STOP]);

        let bytecode = analyze_code(code);
        // The folded result is operand 1 (second from top) at MSTORE.
        bytecode.const_operand(Inst::from_usize(mstore_inst), 1)
    }

    #[test]
    fn const_fold_0_to_1() {
        // CODESIZE: code = CODESIZE(1) + PUSH0(1) + MSTORE(1) + STOP(1) = 4 bytes.
        assert_eq!(const_fold(op::CODESIZE, &[]), Some(U256::from(4)));

        // PC: at position 0.
        assert_eq!(const_fold(op::PC, &[]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_1_to_1() {
        // ISZERO.
        assert_eq!(const_fold(op::ISZERO, &[U256::ZERO]), Some(U256::from(1)));
        assert_eq!(const_fold(op::ISZERO, &[U256::from(42)]), Some(U256::ZERO));

        // NOT.
        assert_eq!(const_fold(op::NOT, &[U256::ZERO]), Some(U256::MAX));
        assert_eq!(const_fold(op::NOT, &[U256::MAX]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_add() {
        assert_eq!(const_fold(op::ADD, &[U256::from(3), U256::from(4)]), Some(U256::from(7)));
        assert_eq!(const_fold(op::ADD, &[U256::MAX, U256::from(1)]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_mul() {
        assert_eq!(const_fold(op::MUL, &[U256::from(3), U256::from(7)]), Some(U256::from(21)));
        assert_eq!(const_fold(op::MUL, &[U256::from(5), U256::ZERO]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_sub() {
        // SUB(10, 3) = 7.
        assert_eq!(const_fold(op::SUB, &[U256::from(10), U256::from(3)]), Some(U256::from(7)));
        assert_eq!(const_fold(op::SUB, &[U256::ZERO, U256::from(1)]), Some(U256::MAX));
    }

    #[test]
    fn const_fold_div() {
        // DIV(10, 3) = 3.
        assert_eq!(const_fold(op::DIV, &[U256::from(10), U256::from(3)]), Some(U256::from(3)));
        assert_eq!(const_fold(op::DIV, &[U256::from(7), U256::ZERO]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_sdiv() {
        assert_eq!(const_fold(op::SDIV, &[U256::from(10), U256::from(3)]), Some(U256::from(3)));
        let neg10 = U256::ZERO.wrapping_sub(U256::from(10));
        let neg3 = U256::ZERO.wrapping_sub(U256::from(3));
        // SDIV(-10, 3) = -3.
        assert_eq!(const_fold(op::SDIV, &[neg10, U256::from(3)]), Some(neg3));
        assert_eq!(const_fold(op::SDIV, &[U256::from(7), U256::ZERO]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_mod() {
        // MOD(10, 3) = 1.
        assert_eq!(const_fold(op::MOD, &[U256::from(10), U256::from(3)]), Some(U256::from(1)));
        assert_eq!(const_fold(op::MOD, &[U256::from(7), U256::ZERO]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_smod() {
        let neg8 = U256::ZERO.wrapping_sub(U256::from(8));
        let neg3 = U256::ZERO.wrapping_sub(U256::from(3));
        let neg2 = U256::ZERO.wrapping_sub(U256::from(2));
        // SMOD(-8, 3) = -2.
        assert_eq!(const_fold(op::SMOD, &[neg8, U256::from(3)]), Some(neg2));
        // SMOD(-8, -3) = -2.
        assert_eq!(const_fold(op::SMOD, &[neg8, neg3]), Some(neg2));
    }

    #[test]
    fn const_fold_addmod() {
        // ADDMOD(10, 10, 8) = 4.
        assert_eq!(
            const_fold(op::ADDMOD, &[U256::from(10), U256::from(10), U256::from(8)]),
            Some(U256::from(4)),
        );
        assert_eq!(
            const_fold(op::ADDMOD, &[U256::from(1), U256::from(2), U256::ZERO]),
            Some(U256::ZERO),
        );
    }

    #[test]
    fn const_fold_mulmod() {
        // MULMOD(10, 10, 8) = 4.
        assert_eq!(
            const_fold(op::MULMOD, &[U256::from(10), U256::from(10), U256::from(8)]),
            Some(U256::from(4)),
        );
    }

    #[test]
    fn const_fold_exp() {
        // EXP(2, 10) = 1024.
        assert_eq!(const_fold(op::EXP, &[U256::from(2), U256::from(10)]), Some(U256::from(1024)));
        // EXP(5, 0) = 1.
        assert_eq!(const_fold(op::EXP, &[U256::from(5), U256::ZERO]), Some(U256::from(1)));
    }

    #[test]
    fn const_fold_signextend() {
        // SIGNEXTEND(0, 0xFF) -> all ones.
        assert_eq!(const_fold(op::SIGNEXTEND, &[U256::ZERO, U256::from(0xFF)]), Some(U256::MAX));
        // SIGNEXTEND(0, 0x7F) -> 0x7F.
        assert_eq!(
            const_fold(op::SIGNEXTEND, &[U256::ZERO, U256::from(0x7F)]),
            Some(U256::from(0x7F)),
        );
        // ext >= 31 -> no-op.
        assert_eq!(
            const_fold(op::SIGNEXTEND, &[U256::from(31), U256::from(0xFF)]),
            Some(U256::from(0xFF)),
        );
    }

    #[test]
    fn const_fold_lt_gt_eq() {
        // LT(1, 2) = 1.
        assert_eq!(const_fold(op::LT, &[U256::from(1), U256::from(2)]), Some(U256::from(1)));
        assert_eq!(const_fold(op::LT, &[U256::from(2), U256::from(1)]), Some(U256::ZERO));
        // GT(2, 1) = 1.
        assert_eq!(const_fold(op::GT, &[U256::from(2), U256::from(1)]), Some(U256::from(1)));
        assert_eq!(const_fold(op::GT, &[U256::from(1), U256::from(2)]), Some(U256::ZERO));
        assert_eq!(const_fold(op::EQ, &[U256::from(5), U256::from(5)]), Some(U256::from(1)));
        assert_eq!(const_fold(op::EQ, &[U256::from(5), U256::from(6)]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_slt_sgt() {
        let neg1 = U256::MAX;
        // SLT(-1, 1) = 1.
        assert_eq!(const_fold(op::SLT, &[neg1, U256::from(1)]), Some(U256::from(1)));
        // SLT(1, -1) = 0.
        assert_eq!(const_fold(op::SLT, &[U256::from(1), neg1]), Some(U256::ZERO));
        // SGT(1, -1) = 1.
        assert_eq!(const_fold(op::SGT, &[U256::from(1), neg1]), Some(U256::from(1)));
    }

    #[test]
    fn const_fold_and_or_xor() {
        assert_eq!(
            const_fold(op::AND, &[U256::from(0xFF), U256::from(0x0F)]),
            Some(U256::from(0x0F)),
        );
        assert_eq!(
            const_fold(op::OR, &[U256::from(0xF0), U256::from(0x0F)]),
            Some(U256::from(0xFF)),
        );
        assert_eq!(const_fold(op::XOR, &[U256::from(0xFF), U256::from(0xFF)]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_byte() {
        // BYTE(31, 0xFF) = 0xFF (last byte).
        assert_eq!(
            const_fold(op::BYTE, &[U256::from(31), U256::from(0xFF)]),
            Some(U256::from(0xFF)),
        );
        // BYTE(0, 0xFF) = 0 (0xFF is in byte 31 only).
        assert_eq!(const_fold(op::BYTE, &[U256::ZERO, U256::from(0xFF)]), Some(U256::ZERO));
        // Out of bounds.
        assert_eq!(const_fold(op::BYTE, &[U256::from(32), U256::from(0xFF)]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_shl_shr() {
        // SHL(1, 0x80) = 0x100.
        assert_eq!(
            const_fold(op::SHL, &[U256::from(1), U256::from(0x80)]),
            Some(U256::from(0x100)),
        );
        assert_eq!(const_fold(op::SHL, &[U256::from(256), U256::from(1)]), Some(U256::ZERO));
        // SHR(1, 0x80) = 0x40.
        assert_eq!(const_fold(op::SHR, &[U256::from(1), U256::from(0x80)]), Some(U256::from(0x40)),);
        assert_eq!(const_fold(op::SHR, &[U256::from(256), U256::from(1)]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_sar() {
        // SAR(1, 2) = 1.
        assert_eq!(const_fold(op::SAR, &[U256::from(1), U256::from(2)]), Some(U256::from(1)));
        // SAR(4, -1) = -1.
        assert_eq!(const_fold(op::SAR, &[U256::from(4), U256::MAX]), Some(U256::MAX));
        // SAR(256, -1) = -1.
        assert_eq!(const_fold(op::SAR, &[U256::from(256), U256::MAX]), Some(U256::MAX));
        // SAR(256, 1) = 0.
        assert_eq!(const_fold(op::SAR, &[U256::from(256), U256::from(1)]), Some(U256::ZERO));
    }
}

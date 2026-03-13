//! Abstract stack interpretation for resolving dynamic jump targets.
//!
//! This pass builds a basic-block CFG and performs constant propagation over an abstract stack
//! to resolve jump targets that the simple `static_jump_analysis` (which only looks at adjacent
//! `PUSH + JUMP/JUMPI`) cannot handle.
//!
//! The abstract domain is:
//! - `Const(U256)` — a known constant value.
//! - `Top` — reachable but unknown.
//!
//! Block input states use `Bottom` (unreachable) at the block level.

use super::{Bytecode, InstFlags};
use revm_bytecode::opcode as op;
use revm_primitives::U256;
use std::collections::VecDeque;

/// Abstract value on the stack.
#[derive(Clone, Debug, PartialEq, Eq)]
enum AbsValue {
    Const(U256),
    Top,
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

impl JumpTarget {
    /// Join another observed target into this one.
    fn join(&mut self, other: &Self) {
        *self = match (&*self, other) {
            (Self::Bottom, x) => x.clone(),
            (x, Self::Bottom) => x.clone(),
            (Self::Const(a), Self::Const(b)) if a == b => Self::Const(*a),
            (Self::Invalid, Self::Invalid) => Self::Invalid,
            _ => Self::Top,
        };
    }
}

/// CFG for abstract interpretation.
struct Cfg {
    blocks: Vec<BasicBlock>,
    /// Maps instruction index to block ID.
    inst_to_block: Vec<BlockId>,
}

impl Bytecode<'_> {
    /// Runs abstract stack interpretation to resolve additional jump targets.
    #[instrument(name = "block_analysis", level = "debug", skip_all)]
    pub(crate) fn block_analysis(&mut self) {
        // Only run if there are dynamic jumps remaining.
        if !self.has_dynamic_jumps {
            return;
        }

        let cfg = self.build_cfg();
        if cfg.blocks.is_empty() {
            return;
        }

        let (resolved, count) = self.run_abstract_interp(&cfg);
        if count == 0 {
            return;
        }

        // Commit resolved targets.
        let mut newly_resolved = 0u32;
        for (jump_inst, target) in resolved {
            let jump = &self.insts[jump_inst];
            // Skip if already resolved by static_jump_analysis.
            if jump.flags.contains(InstFlags::STATIC_JUMP) {
                continue;
            }

            match target {
                JumpTarget::Const(target_inst) => {
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
                JumpTarget::Bottom => {
                    // Unreachable jump — mark as invalid so it doesn't keep
                    // `has_dynamic_jumps` set and force a dynamic jump table.
                    self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP | InstFlags::INVALID_JUMP;
                    trace!(jump_inst, "unreachable jump");
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
    fn run_abstract_interp(&self, cfg: &Cfg) -> (Vec<(usize, JumpTarget)>, usize) {
        let num_blocks = cfg.blocks.len();

        // Collect unresolved jumps and their indices.
        let mut jump_inst_to_idx = rustc_hash::FxHashMap::default();
        let mut jump_targets: Vec<(usize, JumpTarget)> = Vec::new();
        for (i, inst) in self.insts.iter().enumerate() {
            if inst.is_legacy_jump() && !inst.flags.contains(InstFlags::STATIC_JUMP) {
                let idx = jump_targets.len();
                jump_targets.push((i, JumpTarget::Bottom));
                jump_inst_to_idx.insert(i, idx);
            }
        }

        if jump_targets.is_empty() {
            return (jump_targets, 0);
        }

        // Phase 1: Run optimistically without widening to discover targets.
        let mut block_states: Vec<BlockState> = vec![BlockState::Bottom; num_blocks];
        block_states[0] = BlockState::Known(Vec::new());
        self.run_fixpoint(
            cfg,
            &mut block_states,
            &mut jump_targets,
            &jump_inst_to_idx,
            &vec![false; num_blocks],
        );

        // Invalidate resolutions that flowed through conflicted blocks.
        // A conflict means multiple predecessors had incompatible stack heights,
        // so any resolution that depends on flow through that block is unsound
        // (the optimistic pass may have seen only one predecessor's values).
        let mut tainted = vec![false; num_blocks];
        for (bid, state) in block_states.iter().enumerate() {
            if matches!(state, BlockState::Conflict) {
                tainted[bid] = true;
            }
        }
        // Propagate taint to successors.
        let mut changed = true;
        while changed {
            changed = false;
            for bid in 0..num_blocks {
                if !tainted[bid] {
                    continue;
                }
                for &succ in &cfg.blocks[bid].succs {
                    if !tainted[succ] {
                        tainted[succ] = true;
                        changed = true;
                    }
                }
            }
        }
        // Invalidate any resolution in a tainted block.
        for (inst, target) in jump_targets.iter_mut() {
            if matches!(target, JumpTarget::Const(_) | JumpTarget::Invalid) {
                let bid = cfg.inst_to_block[*inst];
                if bid < num_blocks && tainted[bid] {
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

    /// Run a single fixpoint pass of abstract interpretation.
    fn run_fixpoint(
        &self,
        cfg: &Cfg,
        block_states: &mut [BlockState],
        jump_targets: &mut [(usize, JumpTarget)],
        jump_inst_to_idx: &rustc_hash::FxHashMap<usize, usize>,
        widen: &[bool],
    ) {
        let num_blocks = cfg.blocks.len();
        let mut worklist = VecDeque::new();
        worklist.push_back(0usize);
        let mut in_worklist = vec![false; num_blocks];
        in_worklist[0] = true;

        let max_iterations = num_blocks * 8;
        let mut iterations = 0;

        while let Some(bid) = worklist.pop_front() {
            in_worklist[bid] = false;
            iterations += 1;
            if iterations > max_iterations {
                debug!("block_analysis: iteration limit reached");
                break;
            }

            let input = match &block_states[bid] {
                BlockState::Known(s) => s.clone(),
                BlockState::Bottom => continue,
                BlockState::Conflict => continue,
            };

            let block = &cfg.blocks[bid];

            // For JUMPDEST blocks that need widening (potential dynamic jump
            // table targets), widen incoming values to Top. This preserves
            // stack height but makes all inherited values unknown.
            let input =
                if widen[bid] { input.iter().map(|_| AbsValue::Top).collect() } else { input };

            let output = self.interpret_block(block.start, block.end, &input);
            let output = match output {
                Some(s) => s,
                None => continue,
            };

            let term = &self.insts[block.end];
            let mut extra_succs: Vec<BlockId> = Vec::new();

            if term.is_legacy_jump() && !term.flags.contains(InstFlags::STATIC_JUMP) {
                let pre_jump_stack = if block.start == block.end {
                    input.clone()
                } else {
                    self.interpret_block(block.start, block.end - 1, &input).unwrap_or_default()
                };

                let target_val = pre_jump_stack.last().cloned();

                let resolved = match target_val {
                    Some(AbsValue::Const(val)) => match usize::try_from(val) {
                        Ok(target_pc) if self.is_valid_jump(target_pc) => {
                            let target_inst = self.pc_to_inst(target_pc);
                            JumpTarget::Const(target_inst)
                        }
                        _ => JumpTarget::Invalid,
                    },
                    Some(AbsValue::Top) => JumpTarget::Top,
                    None => JumpTarget::Top,
                };

                if let JumpTarget::Const(target_inst) = &resolved {
                    let target_block = cfg.inst_to_block[*target_inst];
                    if target_block != usize::MAX && target_block < num_blocks {
                        extra_succs.push(target_block);
                    }
                }

                if term.opcode == op::JUMPI {
                    let cond_val = pre_jump_stack
                        .len()
                        .checked_sub(2)
                        .and_then(|i| pre_jump_stack.get(i).cloned());
                    match cond_val {
                        Some(AbsValue::Const(v)) if v.is_zero() => {
                            extra_succs.clear();
                        }
                        Some(AbsValue::Const(_)) => {}
                        _ => {}
                    }
                }

                if let Some(&idx) = jump_inst_to_idx.get(&block.end) {
                    jump_targets[idx].1.join(&resolved);
                }
            }

            for &succ in &block.succs {
                if block_states[succ].join(&output) && !in_worklist[succ] {
                    worklist.push_back(succ);
                    in_worklist[succ] = true;
                }
            }

            for succ in extra_succs {
                if block_states[succ].join(&output) && !in_worklist[succ] {
                    worklist.push_back(succ);
                    in_worklist[succ] = true;
                }
            }
        }
    }

    /// Interpret a sequence of instructions on the abstract stack.
    /// Returns the abstract stack state after the last instruction, or `None` on conflict.
    fn interpret_block(
        &self,
        start: usize,
        end: usize,
        input: &[AbsValue],
    ) -> Option<Vec<AbsValue>> {
        let mut stack = input.to_vec();

        for i in start..=end {
            let inst = &self.insts[i];
            if inst.is_dead_code() {
                continue;
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
    fn revert_remote_sub_call_storage_oog_caller() {
        let bytecode = analyze_bytecode(
            "608060405234801561001057600080fd5b50600436106100415760003560e01c806354d1405f14\
             610046578063b28175c414610050578063c04062261461005a575b600080fd5b61004e610064565b\
             005b610058610110565b005b610062610116565b005b6000604051610072906101b6565b60405180\
             9103906000f08015801561008e573d6000803e3d6000fd5b5090508073ffffffffffffffffffff\
             ffffffffffffffffffff166373027f6d306040518263ffffffff1660e01b81526004016100ca91\
             90610204565b600060405180830381600087803b1580156100e457600080fd5b505af11580156100\
             f8573d6000803e3d6000fd5b50505050600360025560038055622fffff6000205050565b60028055\
             565b6000604051610124906101b6565b604051809103906000f080158015610140573d6000803e3d\
             6000fd5b5090508073ffffffffffffffffffffffffffffffffffffffff166373027f6d3060405182\
             63ffffffff1660e01b815260040161017c9190610204565b600060405180830381600087803b1580\
             1561019657600080fd5b505af11580156101aa573d6000803e3d6000fd5b50505050600360025550\
             565b6102b48061022083390190565b600073ffffffffffffffffffffffffffffffffffffff\
             ff82169050919050565b60006101ee826101c3565b9050919050565b6101fe816101e3565b825250\
             50565b600060208201905061021960008301846101f5565b9291505056fe",
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

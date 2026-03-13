//! Abstract interpretation for constant propagation over the EVM control flow graph.
//!
//! Discovers constant operands of instructions by forward dataflow analysis, primarily to resolve
//! more JUMP/JUMPI targets statically.

use super::Bytecode;
use crate::{Inst, InstData, InstFlags};
use revm_bytecode::opcode as op;
use revm_primitives::U256;

/// Abstract value in the constant propagation lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AbsVal {
    /// Unknown value.
    Top,
    /// Known constant.
    Const(U256),
}

impl AbsVal {
    /// Joins two abstract values.
    #[inline]
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) if a == b => Self::Const(a),
            _ => Self::Top,
        }
    }

    /// Returns the constant value, if known.
    #[inline]
    #[allow(dead_code)]
    fn as_const(self) -> Option<U256> {
        match self {
            Self::Const(v) => Some(v),
            Self::Top => None,
        }
    }
}

/// Abstract EVM stack.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct AbsStack {
    values: Vec<AbsVal>,
}

impl AbsStack {
    fn push(&mut self, val: AbsVal) {
        self.values.push(val);
    }

    fn pop(&mut self) -> Option<AbsVal> {
        self.values.pop()
    }

    fn pop_or_top(&mut self) -> AbsVal {
        self.pop().unwrap_or(AbsVal::Top)
    }

    /// Pops `n` values, returning Top for any underflow.
    fn pop_n(&mut self, n: usize) -> Vec<AbsVal> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.pop_or_top());
        }
        out
    }

    #[allow(dead_code)]
    fn push_n(&mut self, vals: impl IntoIterator<Item = AbsVal>) {
        for v in vals {
            self.push(v);
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.values.len()
    }

    /// Peek at the nth element from the top (0 = top of stack).
    fn peek(&self, n: usize) -> AbsVal {
        if n < self.values.len() { self.values[self.values.len() - 1 - n] } else { AbsVal::Top }
    }

    /// Swap the top of stack with the element at depth `n` (1-indexed, like SWAP1..SWAP16).
    fn swap(&mut self, n: usize) {
        let len = self.values.len();
        if n < len {
            self.values.swap(len - 1, len - 1 - n);
        } else {
            // Swapping with unknown → both become Top.
            if let Some(top) = self.values.last_mut() {
                *top = AbsVal::Top;
            }
        }
    }

    /// Element-wise join. Returns `None` if stack heights differ.
    fn join(&self, other: &Self) -> Option<Self> {
        if self.values.len() != other.values.len() {
            return None;
        }
        let values = self.values.iter().zip(other.values.iter()).map(|(a, b)| a.join(*b)).collect();
        Some(Self { values })
    }
}

/// State at the entry of a basic block.
#[derive(Clone, Debug)]
enum BlockInState {
    /// Not yet reached.
    Unreachable,
    /// Reached with a known abstract stack.
    Known(AbsStack),
    /// Reached but stack height mismatch → stop propagating.
    Conflict,
}

/// A basic block in the analysis CFG.
#[derive(Clone, Debug)]
struct AnalysisBlock {
    /// First instruction index (inclusive).
    start: Inst,
    /// Last instruction index (inclusive).
    end: Inst,
    /// Successor block IDs.
    succs: Vec<u32>,
}

type BlockId = u32;

/// The analysis CFG built from known control flow.
struct AnalysisCfg {
    blocks: Vec<AnalysisBlock>,
    /// Maps instruction index → block ID. `u32::MAX` for dead/unmapped instructions.
    #[allow(dead_code)]
    inst_to_block: Vec<BlockId>,
}

impl AnalysisCfg {
    /// Build the CFG from the current bytecode state.
    fn build(bytecode: &Bytecode<'_>) -> Self {
        let n_insts = bytecode.insts.len();

        // Step 1: Identify block leaders.
        // A leader is: inst 0, any JUMPDEST, any instruction following a branching/diverging inst.
        let mut is_leader = vec![false; n_insts];
        if n_insts > 0 {
            is_leader[0] = true;
        }
        for (i, data) in bytecode.iter_all_insts() {
            if data.is_dead_code() {
                continue;
            }
            if data.is_jumpdest() {
                is_leader[i] = true;
            }
            if (data.is_branching() || data.is_diverging()) && i + 1 < n_insts {
                is_leader[i + 1] = true;
            }
        }

        // Step 2: Assign block IDs and build blocks.
        let mut inst_to_block = vec![u32::MAX; n_insts];
        let mut blocks = Vec::new();
        let mut current_start: Option<usize> = None;

        let finalize_block = |blocks: &mut Vec<AnalysisBlock>,
                              inst_to_block: &mut [BlockId],
                              start: usize,
                              end_exclusive: usize| {
            let block_id = blocks.len() as BlockId;
            inst_to_block[start..end_exclusive].fill(block_id);
            blocks.push(AnalysisBlock { start, end: end_exclusive - 1, succs: Vec::new() });
        };

        for (i, data) in bytecode.iter_all_insts() {
            if data.is_dead_code() {
                if let Some(start) = current_start.take() {
                    finalize_block(&mut blocks, &mut inst_to_block, start, i);
                }
                continue;
            }

            if is_leader[i] {
                if let Some(start) = current_start.take() {
                    finalize_block(&mut blocks, &mut inst_to_block, start, i);
                }
                current_start = Some(i);
            } else if current_start.is_none() {
                current_start = Some(i);
            }
        }

        // Finalize last block.
        if let Some(start) = current_start {
            finalize_block(&mut blocks, &mut inst_to_block, start, n_insts);
        }

        // Step 3: Add edges.
        for block in &mut blocks {
            let end = block.end;
            let term = &bytecode.insts[end];

            if term.is_dead_code() {
                continue;
            }

            if term.is_known_jump() {
                let target_inst = term.data as usize;
                if target_inst < n_insts && inst_to_block[target_inst] != u32::MAX {
                    block.succs.push(inst_to_block[target_inst]);
                }
                // JUMPI also falls through.
                if term.opcode == op::JUMPI {
                    let next = end + 1;
                    if next < n_insts && inst_to_block[next] != u32::MAX {
                        block.succs.push(inst_to_block[next]);
                    }
                }
            } else if term.is_diverging() {
                // No successors (STOP, RETURN, REVERT, INVALID, etc.).
            } else if term.is_jump() {
                // Dynamic jump — barrier, no known successors.
                // JUMPI with dynamic target: fallthrough edge only.
                if term.opcode == op::JUMPI {
                    let next = end + 1;
                    if next < n_insts && inst_to_block[next] != u32::MAX {
                        block.succs.push(inst_to_block[next]);
                    }
                }
            } else {
                // Fallthrough.
                let next = end + 1;
                if next < n_insts && inst_to_block[next] != u32::MAX {
                    block.succs.push(inst_to_block[next]);
                }
            }
        }

        Self { blocks, inst_to_block }
    }
}

/// Result of constant propagation analysis.
#[derive(Debug, Default)]
pub(crate) struct ConstPropResult {
    /// Discovered jump target constants: maps jump_inst → known constant target value.
    /// Only jumps where the target is still constant after convergence are included.
    pub(crate) resolved_jumps: Vec<(Inst, Inst)>,
}

/// Per-jump accumulator for constant target values across worklist iterations.
#[derive(Debug, Clone, Copy)]
enum JumpTarget {
    /// Constant target discovered so far.
    Const(U256),
    /// Multiple conflicting values seen → not resolvable.
    Conflict,
}

/// Run constant propagation on the bytecode.
///
/// This is the outer fixed-point loop that alternates between:
/// 1. Building the CFG from currently known control flow
/// 2. Running forward const-prop over the CFG
/// 3. Applying newly discovered jump targets
/// 4. Repeating until stable
pub(crate) fn run(bytecode: &mut Bytecode<'_>) {
    // Seed with the existing trivial PUSH+JUMP analysis.
    bytecode.static_jump_analysis();

    let mut iteration = 0;
    loop {
        iteration += 1;
        trace!(iteration, "const_prop iteration");

        let cfg = AnalysisCfg::build(bytecode);
        let result = propagate(&cfg, bytecode);

        if result.resolved_jumps.is_empty() {
            trace!(iteration, "const_prop converged");
            break;
        }

        debug!(
            iteration,
            n_resolved = result.resolved_jumps.len(),
            "const_prop resolved new jumps"
        );

        // Apply newly discovered jump targets.
        for &(jump_inst, target_inst) in &result.resolved_jumps {
            let jump = &mut bytecode.insts[jump_inst];
            jump.flags |= InstFlags::CONST_JUMP;
            jump.data = target_inst as u32;

            // Mark the JUMPDEST as reachable.
            bytecode.insts[target_inst].data = 1;
        }
    }

    // Recompute has_dynamic_jumps. Note: do NOT change has_dynamic_jumps to false here.
    // The const-prop analysis only considers reachable CFG paths, so it may miss dynamic jumps
    // that are reachable through paths it didn't analyze (e.g., from dynamic JUMPI fallthrough
    // edges into code containing more jumps). Widening from false→true is safe; narrowing is not.
    if !bytecode.has_dynamic_jumps {
        bytecode.has_dynamic_jumps =
            bytecode.insts.iter().any(|data| data.is_jump() && !data.is_known_jump());
    }
}

/// Run one pass of forward constant propagation over the CFG.
fn propagate(cfg: &AnalysisCfg, bytecode: &Bytecode<'_>) -> ConstPropResult {
    use rustc_hash::FxHashMap;

    let n_blocks = cfg.blocks.len();
    if n_blocks == 0 {
        return ConstPropResult::default();
    }

    let mut states: Vec<BlockInState> = vec![BlockInState::Unreachable; n_blocks];
    states[0] = BlockInState::Known(AbsStack::default());

    // If dynamic jumps exist, any JUMPDEST block could be entered with unknown stack.
    // Mark those blocks as Conflict to prevent unsound constant propagation.
    if bytecode.has_dynamic_jumps() {
        for (block_id, block) in cfg.blocks.iter().enumerate() {
            let first_inst = &bytecode.insts[block.start];
            if first_inst.is_reachable_jumpdest(true) {
                states[block_id] = BlockInState::Conflict;
            }
        }
    }

    let mut worklist: Vec<BlockId> = vec![0];
    let mut in_worklist = vec![false; n_blocks];
    in_worklist[0] = true;

    // Track jump target constants with join semantics: if a jump is reached with different
    // constant targets on different paths, it becomes Conflict and won't be resolved.
    let mut jump_targets: FxHashMap<Inst, JumpTarget> = FxHashMap::default();

    while let Some(block_id) = worklist.pop() {
        let block_id = block_id as usize;
        in_worklist[block_id] = false;

        let in_stack = match &states[block_id] {
            BlockInState::Known(s) => s.clone(),
            BlockInState::Unreachable | BlockInState::Conflict => continue,
        };

        let block = &cfg.blocks[block_id];
        let out_stack = eval_block(block, &in_stack, bytecode, &mut jump_targets);

        // Propagate to successors.
        for &succ_id in &block.succs {
            let succ = succ_id as usize;
            let changed = match &states[succ] {
                BlockInState::Unreachable => {
                    states[succ] = BlockInState::Known(out_stack.clone());
                    true
                }
                BlockInState::Known(existing) => {
                    if let Some(joined) = existing.join(&out_stack) {
                        if joined != *existing {
                            states[succ] = BlockInState::Known(joined);
                            true
                        } else {
                            false
                        }
                    } else {
                        // Stack height mismatch.
                        states[succ] = BlockInState::Conflict;
                        true
                    }
                }
                BlockInState::Conflict => false,
            };

            if changed && !in_worklist[succ] {
                worklist.push(succ_id);
                in_worklist[succ] = true;
            }
        }
    }

    // Collect resolved jumps from the converged jump_targets map.
    let mut result = ConstPropResult::default();
    for (jump_inst, target) in jump_targets {
        if let JumpTarget::Const(target_pc) = target {
            let target_pc_usize: usize = target_pc.try_into().unwrap_or(usize::MAX);
            if bytecode.is_valid_jump(target_pc_usize) {
                let target_inst = bytecode.pc_to_inst(target_pc_usize);
                result.resolved_jumps.push((jump_inst, target_inst));
                trace!(
                    jump_inst,
                    target_inst,
                    target_pc = target_pc_usize,
                    "resolved jump via const-prop"
                );
            }
        }
    }

    result
}

/// Evaluate a single basic block with the given input stack.
fn eval_block(
    block: &AnalysisBlock,
    in_stack: &AbsStack,
    bytecode: &Bytecode<'_>,
    jump_targets: &mut rustc_hash::FxHashMap<Inst, JumpTarget>,
) -> AbsStack {
    let mut stack = in_stack.clone();

    for inst in block.start..=block.end {
        let data = &bytecode.insts[inst];
        if data.is_dead_code() || data.flags.contains(InstFlags::SKIP_LOGIC) {
            continue;
        }

        eval_inst(inst, data, bytecode, &mut stack, jump_targets);
    }

    stack
}

/// Evaluate a single instruction's effect on the abstract stack.
fn eval_inst(
    inst: Inst,
    data: &InstData,
    bytecode: &Bytecode<'_>,
    stack: &mut AbsStack,
    jump_targets: &mut rustc_hash::FxHashMap<Inst, JumpTarget>,
) {
    let opcode = data.opcode;

    match opcode {
        // PUSH instructions.
        op::PUSH0 => {
            stack.push(AbsVal::Const(U256::ZERO));
        }
        op::PUSH1..=op::PUSH32 => {
            let val = if let Some(imm) = bytecode.get_imm(data) {
                let mut buf = [0u8; 32];
                buf[32 - imm.len()..].copy_from_slice(imm);
                AbsVal::Const(U256::from_be_bytes(buf))
            } else {
                AbsVal::Top
            };
            stack.push(val);
        }

        // DUP instructions.
        op::DUP1..=op::DUP16 => {
            let depth = (opcode - op::DUP1) as usize;
            let val = stack.peek(depth);
            stack.push(val);
        }

        // SWAP instructions.
        op::SWAP1..=op::SWAP16 => {
            let depth = (opcode - op::SWAP1 + 1) as usize;
            stack.swap(depth);
        }

        op::POP => {
            stack.pop_or_top();
        }

        // PC is a known constant.
        op::PC => {
            stack.push(AbsVal::Const(U256::from(data.pc)));
        }

        // CODESIZE is known at compile time.
        op::CODESIZE => {
            stack.push(AbsVal::Const(U256::from(bytecode.code.len())));
        }

        // JUMP/JUMPI: check if the target is a known constant.
        op::JUMP | op::JUMPI => {
            // If already resolved, handle the stack effect correctly.
            if data.is_known_jump() {
                if data.flags.contains(InstFlags::STATIC_JUMP) {
                    // PUSH was marked SKIP_LOGIC, target not on abstract stack.
                    // stack_io already accounts for the reduced inputs.
                    let (inp, _) = data.stack_io();
                    stack.pop_n(inp as usize);
                } else {
                    // CONST_JUMP: target is still on the stack.
                    stack.pop_or_top(); // target
                    if opcode == op::JUMPI {
                        stack.pop_or_top(); // condition
                    }
                }
                return;
            }

            let target_val = stack.pop_or_top();
            if opcode == op::JUMPI {
                stack.pop_or_top(); // condition
            }

            // Record the target value with join semantics.
            match target_val {
                AbsVal::Const(pc) => {
                    jump_targets
                        .entry(inst)
                        .and_modify(|existing| match *existing {
                            JumpTarget::Const(prev) if prev == pc => {}
                            _ => *existing = JumpTarget::Conflict,
                        })
                        .or_insert(JumpTarget::Const(pc));
                }
                AbsVal::Top => {
                    jump_targets.insert(inst, JumpTarget::Conflict);
                }
            }
        }

        // Pure arithmetic/bitwise ops.
        op::ADD => binary_op(stack, |a, b| a.wrapping_add(b)),
        op::SUB => binary_op(stack, |a, b| a.wrapping_sub(b)),
        op::MUL => binary_op(stack, |a, b| a.wrapping_mul(b)),
        op::DIV => binary_op(stack, |a, b| a.checked_div(b).unwrap_or(U256::ZERO)),
        op::SDIV => binary_op(stack, sdiv),
        op::MOD => binary_op(stack, |a, b| if b.is_zero() { U256::ZERO } else { a % b }),
        op::SMOD => binary_op(stack, smod),
        op::ADDMOD => {
            ternary_op(stack, |a, b, n| if n.is_zero() { U256::ZERO } else { a.add_mod(b, n) })
        }
        op::MULMOD => {
            ternary_op(stack, |a, b, n| if n.is_zero() { U256::ZERO } else { a.mul_mod(b, n) })
        }
        op::EXP => binary_op(stack, |a, b| a.pow(b)),
        op::SIGNEXTEND => binary_op(stack, signextend),

        op::LT => binary_op(stack, |a, b| U256::from(a < b)),
        op::GT => binary_op(stack, |a, b| U256::from(a > b)),
        op::SLT => binary_op(stack, |a, b| U256::from(i256_lt(a, b))),
        op::SGT => binary_op(stack, |a, b| U256::from(i256_lt(b, a))),
        op::EQ => binary_op(stack, |a, b| U256::from(a == b)),
        op::ISZERO => unary_op(stack, |a| U256::from(a.is_zero())),

        op::AND => binary_op(stack, |a, b| a & b),
        op::OR => binary_op(stack, |a, b| a | b),
        op::XOR => binary_op(stack, |a, b| a ^ b),
        op::NOT => unary_op(stack, |a| !a),
        op::BYTE => binary_op(stack, |i, x| {
            if i < U256::from(32) {
                let i: usize = i.to();
                U256::from(x.byte(31 - i))
            } else {
                U256::ZERO
            }
        }),
        op::SHL => binary_op(stack, |shift, value| {
            if shift < U256::from(256) {
                let shift: usize = shift.to();
                value << shift
            } else {
                U256::ZERO
            }
        }),
        op::SHR => binary_op(stack, |shift, value| {
            if shift < U256::from(256) {
                let shift: usize = shift.to();
                value >> shift
            } else {
                U256::ZERO
            }
        }),
        op::SAR => binary_op(stack, sar),

        // Everything else: generic stack effect.
        _ => {
            let (inp, out) = data.stack_io();
            stack.pop_n(inp as usize);
            for _ in 0..out {
                stack.push(AbsVal::Top);
            }
        }
    }
}

/// Apply a unary operation on the abstract stack.
fn unary_op(stack: &mut AbsStack, f: impl FnOnce(U256) -> U256) {
    let a = stack.pop_or_top();
    match a {
        AbsVal::Const(a) => stack.push(AbsVal::Const(f(a))),
        AbsVal::Top => stack.push(AbsVal::Top),
    }
}

/// Apply a binary operation on the abstract stack.
fn binary_op(stack: &mut AbsStack, f: impl FnOnce(U256, U256) -> U256) {
    let a = stack.pop_or_top();
    let b = stack.pop_or_top();
    match (a, b) {
        (AbsVal::Const(a), AbsVal::Const(b)) => stack.push(AbsVal::Const(f(a, b))),
        _ => stack.push(AbsVal::Top),
    }
}

/// Apply a ternary operation on the abstract stack.
fn ternary_op(stack: &mut AbsStack, f: impl FnOnce(U256, U256, U256) -> U256) {
    let a = stack.pop_or_top();
    let b = stack.pop_or_top();
    let c = stack.pop_or_top();
    match (a, b, c) {
        (AbsVal::Const(a), AbsVal::Const(b), AbsVal::Const(c)) => {
            stack.push(AbsVal::Const(f(a, b, c)))
        }
        _ => stack.push(AbsVal::Top),
    }
}

// --- Signed arithmetic helpers ---

fn i256_sign(val: U256) -> i32 {
    if val.is_zero() {
        0
    } else if val.bit(255) {
        -1
    } else {
        1
    }
}

fn i256_neg(val: U256) -> U256 {
    (!val).wrapping_add(U256::from(1))
}

fn i256_lt(a: U256, b: U256) -> bool {
    let a_sign = i256_sign(a);
    let b_sign = i256_sign(b);
    match a_sign.cmp(&b_sign) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => a < b,
    }
}

fn sdiv(a: U256, b: U256) -> U256 {
    if b.is_zero() {
        return U256::ZERO;
    }
    let a_sign = i256_sign(a);
    let b_sign = i256_sign(b);
    let a_abs = if a_sign < 0 { i256_neg(a) } else { a };
    let b_abs = if b_sign < 0 { i256_neg(b) } else { b };
    let result = a_abs / b_abs;
    if a_sign != b_sign { i256_neg(result) } else { result }
}

fn smod(a: U256, b: U256) -> U256 {
    if b.is_zero() {
        return U256::ZERO;
    }
    let a_sign = i256_sign(a);
    let b_sign = i256_sign(b);
    let a_abs = if a_sign < 0 { i256_neg(a) } else { a };
    let b_abs = if b_sign < 0 { i256_neg(b) } else { b };
    let result = a_abs % b_abs;
    if a_sign < 0 { i256_neg(result) } else { result }
}

fn signextend(b: U256, x: U256) -> U256 {
    if b < U256::from(31) {
        let b: usize = b.to();
        let bit = b * 8 + 7;
        let mask = (U256::from(1) << bit) - U256::from(1);
        if x.bit(bit) { x | !mask } else { x & mask }
    } else {
        x
    }
}

fn sar(shift: U256, value: U256) -> U256 {
    if shift >= U256::from(256) {
        if value.bit(255) { U256::MAX } else { U256::ZERO }
    } else {
        let shift: usize = shift.to();
        if value.bit(255) {
            // Arithmetic right shift: fill with 1s.
            let shifted = value >> shift;
            let mask = U256::MAX << (256 - shift);
            shifted | mask
        } else {
            value >> shift
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_val_join() {
        let a = AbsVal::Const(U256::from(42));
        let b = AbsVal::Const(U256::from(42));
        let c = AbsVal::Const(U256::from(99));
        assert_eq!(a.join(b), AbsVal::Const(U256::from(42)));
        assert_eq!(a.join(c), AbsVal::Top);
        assert_eq!(a.join(AbsVal::Top), AbsVal::Top);
        assert_eq!(AbsVal::Top.join(AbsVal::Top), AbsVal::Top);
    }

    #[test]
    fn abs_stack_swap() {
        let mut stack = AbsStack::default();
        stack.push(AbsVal::Const(U256::from(1)));
        stack.push(AbsVal::Const(U256::from(2)));
        stack.push(AbsVal::Const(U256::from(3)));
        stack.swap(1); // SWAP1: swap top with depth 1
        assert_eq!(stack.peek(0), AbsVal::Const(U256::from(2)));
        assert_eq!(stack.peek(1), AbsVal::Const(U256::from(3)));
        assert_eq!(stack.peek(2), AbsVal::Const(U256::from(1)));
    }

    #[test]
    fn abs_stack_dup() {
        let mut stack = AbsStack::default();
        stack.push(AbsVal::Const(U256::from(10)));
        stack.push(AbsVal::Const(U256::from(20)));
        let val = stack.peek(1); // DUP2
        stack.push(val);
        assert_eq!(stack.len(), 3);
        assert_eq!(stack.peek(0), AbsVal::Const(U256::from(10)));
    }

    #[test]
    fn abs_stack_join() {
        let mut a = AbsStack::default();
        a.push(AbsVal::Const(U256::from(1)));
        a.push(AbsVal::Const(U256::from(2)));

        let mut b = AbsStack::default();
        b.push(AbsVal::Const(U256::from(1)));
        b.push(AbsVal::Const(U256::from(3)));

        let joined = a.join(&b).unwrap();
        assert_eq!(joined.peek(0), AbsVal::Top); // 2 vs 3
        assert_eq!(joined.peek(1), AbsVal::Const(U256::from(1))); // same

        // Different heights → None.
        let mut c = AbsStack::default();
        c.push(AbsVal::Const(U256::from(1)));
        assert!(a.join(&c).is_none());
    }
}

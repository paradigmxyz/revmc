//! Internal EVM bytecode and opcode representation.

use crate::FxHashMap;
use bitvec::vec::BitVec;
use oxc_index::IndexVec;
use revm_bytecode::opcode as op;
use revm_primitives::{U256, hardfork::SpecId};
use revmc_backend::Result;
use std::cell::RefCell;

mod block_analysis;
pub(crate) use block_analysis::StackSnapshot;

mod fmt;
mod interner;
pub(crate) use interner::Interner;

mod sections;
use sections::{Section, SectionAnalysis};

mod info;
pub use info::*;

mod opcode;
pub use opcode::*;

/// Noop opcode used to test suspend-resume.
#[cfg(any(feature = "__fuzzing", test))]
pub(crate) const TEST_SUSPEND: u8 = 0x25;

oxc_index::define_index_type! {
    /// An EVM instruction index into [`Bytecode`] instructions.
    ///
    /// Also known as `ic`, or instruction counter; not to be confused with SSA `inst`s.
    pub(crate) struct Inst = u32;
}

oxc_index::define_index_type! {
    /// Index into the deduplicated U256 constant pool.
    pub(crate) struct U256Idx = u32;
}

/// Hasher used by the U256 interner.
type FxBuildHasher = alloy_primitives::map::FxBuildHasher;

/// EVM bytecode.
#[doc(hidden)] // Not public API.
pub struct Bytecode<'a> {
    /// The original bytecode slice.
    pub(crate) code: &'a [u8],
    /// The instructions.
    insts: IndexVec<Inst, InstData>,
    /// `JUMPDEST` opcode map. `jumpdests[pc]` is `true` if `code[pc] == op::JUMPDEST`.
    jumpdests: BitVec,
    /// The [`SpecId`].
    pub(crate) spec_id: SpecId,
    /// Whether the bytecode contains dynamic jumps.
    has_dynamic_jumps: bool,
    /// Whether the bytecode may suspend execution.
    may_suspend: bool,
    /// Per-instruction abstract stack snapshots computed by block analysis.
    /// Each entry is the abstract stack state *before* the instruction executes.
    stack_snapshots: IndexVec<Inst, StackSnapshot>,
    /// Deduplicated constant pool for U256 values.
    u256_interner: RefCell<Interner<U256Idx, U256, FxBuildHasher>>,
    /// Mapping from program counter to instruction.
    pc_to_inst: FxHashMap<u32, u32>,
    /// Instruction index to 1-based line number in the formatted dump, built during formatting.
    inst_lines: RefCell<IndexVec<Inst, u32>>,
}

impl<'a> Bytecode<'a> {
    #[instrument(name = "new_bytecode", level = "debug", skip_all)]
    pub(crate) fn new(code: &'a [u8], spec_id: SpecId) -> Self {
        let mut insts = IndexVec::with_capacity(code.len() + 8);
        let mut jumpdests = BitVec::repeat(false, code.len());
        let mut pc_to_inst = FxHashMap::with_capacity_and_hasher(code.len(), Default::default());
        let op_infos = op_info_map(spec_id);
        for (pc, Opcode { opcode, immediate: _ }) in OpcodesIter::new(code, spec_id).with_pc() {
            let inst: Inst = insts.next_idx();
            pc_to_inst.insert(pc as u32, inst.index() as u32);

            if opcode == op::JUMPDEST {
                jumpdests.set(pc, true)
            }

            let data = 0;

            let mut flags = InstFlags::empty();
            let info = op_infos[opcode as usize];
            if info.is_unknown() {
                flags |= InstFlags::UNKNOWN;
            }
            if info.is_disabled() {
                flags |= InstFlags::DISABLED;
            }
            let base_gas = info.base_gas();

            let section = Section::default();

            insts.push(InstData { opcode, flags, base_gas, data, pc: pc as u32, section });
        }

        let mut bytecode = Self {
            code,
            insts,
            jumpdests,
            spec_id,
            has_dynamic_jumps: false,
            may_suspend: false,
            stack_snapshots: IndexVec::new(),
            u256_interner: RefCell::new(Interner::new()),
            pc_to_inst,
            inst_lines: RefCell::new(IndexVec::new()),
        };

        // Pad code to ensure there is at least one diverging instruction.
        if bytecode.insts.last().is_none_or(|last| !last.is_diverging()) {
            bytecode.insts.push(InstData::new(op::STOP));
        }

        bytecode
    }

    /// Takes the instruction-to-line map built during formatting.
    ///
    /// Returns an empty `Vec` if the bytecode has not been formatted yet.
    pub(crate) fn take_inst_lines(&self) -> IndexVec<Inst, u32> {
        self.inst_lines.take()
    }

    /// Returns an iterator over the opcodes.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn opcodes(&self) -> OpcodesIter<'a> {
        OpcodesIter::new(self.code, self.spec_id)
    }

    /// Returns the instruction at the given instruction counter.
    #[inline]
    #[track_caller]
    pub(crate) fn inst(&self, inst: Inst) -> &InstData {
        &self.insts[inst]
    }

    /// Returns a mutable reference the instruction at the given instruction counter.
    #[inline]
    #[track_caller]
    #[allow(dead_code)]
    pub(crate) fn inst_mut(&mut self, inst: Inst) -> &mut InstData {
        &mut self.insts[inst]
    }

    /// Returns the opcode at the given instruction counter.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn opcode(&self, inst: Inst) -> Opcode<'_> {
        self.inst(inst).to_op_in(self)
    }

    /// Returns an iterator over the instructions.
    #[inline]
    pub(crate) fn iter_insts(
        &self,
    ) -> impl DoubleEndedIterator<Item = (Inst, &InstData)> + Clone + '_ {
        self.iter_all_insts().filter(|(_, data)| !data.is_dead_code())
    }

    /// Returns an iterator over the instructions.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn iter_mut_insts(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (Inst, &mut InstData)> + '_ {
        self.iter_mut_all_insts().filter(|(_, data)| !data.is_dead_code())
    }

    /// Returns an iterator over all the instructions, including dead code.
    #[inline]
    pub(crate) fn iter_all_insts(
        &self,
    ) -> impl DoubleEndedIterator<Item = (Inst, &InstData)> + ExactSizeIterator + Clone + '_ {
        self.insts.iter_enumerated()
    }

    /// Returns an iterator over all the instructions, including dead code.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn iter_mut_all_insts(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (Inst, &mut InstData)> + ExactSizeIterator + '_ {
        self.insts.iter_mut_enumerated()
    }

    /// Runs a list of analysis passes on the instructions.
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn analyze(&mut self) -> Result<()> {
        self.static_jump_analysis();
        self.block_analysis();
        // NOTE: `mark_dead_code` must run after jump analysis as it can mark
        // unreachable `JUMPDEST`s as dead code.
        self.mark_dead_code();

        self.calc_may_suspend();

        self.construct_sections();

        Ok(())
    }

    /// Mark `PUSH<N>` followed by `JUMP[I]` as `STATIC_JUMP` and resolve the target.
    #[instrument(name = "sj", level = "debug", skip_all)]
    fn static_jump_analysis(&mut self) {
        for jump_inst in self.insts.indices() {
            let jump = &self.insts[jump_inst];
            let Some(push_inst) = jump_inst.index().checked_sub(1).map(Inst::from_usize) else {
                if jump.is_legacy_jump() {
                    trace!(jump_inst = jump_inst.index(), target=?None::<()>, "found jump");
                    self.has_dynamic_jumps = true;
                }
                continue;
            };

            let push = &self.insts[push_inst];
            if !(push.is_push() && jump.is_legacy_jump()) {
                if jump.is_legacy_jump() {
                    trace!(jump_inst = jump_inst.index(), target=?None::<()>, "found jump");
                    self.has_dynamic_jumps = true;
                }
                continue;
            }

            let imm_opt = self.get_imm(push);
            if push.opcode != op::PUSH0 && imm_opt.is_none() {
                continue;
            }
            let imm = imm_opt.unwrap_or(&[]);
            self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP;

            const USIZE_SIZE: usize = std::mem::size_of::<usize>();
            if imm.len() > USIZE_SIZE {
                trace!(jump_inst = jump_inst.index(), "jump target too large");
                self.insts[jump_inst].flags |= InstFlags::INVALID_JUMP;
                continue;
            }

            let mut padded = [0; USIZE_SIZE];
            padded[USIZE_SIZE - imm.len()..].copy_from_slice(imm);
            let target_pc = usize::from_be_bytes(padded);
            if !self.is_valid_jump(target_pc) {
                trace!(jump_inst = jump_inst.index(), target_pc, "invalid jump target");
                self.insts[jump_inst].flags |= InstFlags::INVALID_JUMP;
                continue;
            }

            self.insts[push_inst].flags |= InstFlags::SKIP_LOGIC;
            let target = self.pc_to_inst(target_pc);

            // Mark the `JUMPDEST` as reachable.
            debug_assert_eq!(
                self.insts[target],
                op::JUMPDEST,
                "is_valid_jump returned true for non-JUMPDEST: \
                 jump_inst={jump_inst:?} target_pc={target_pc} target={target:?}",
            );
            self.insts[target].data = 1;

            // Set the target on the `JUMP` instruction.
            trace!(jump_inst = jump_inst.index(), target = target.index(), "found jump");
            self.insts[jump_inst].data = target.index() as u32;
        }
    }

    /// Mark unreachable instructions as `DEAD_CODE` to not generate any code for them.
    ///
    /// This pass is technically unnecessary as the backend will very likely optimize any
    /// unreachable code that we generate, but this is trivial for us to do and significantly speeds
    /// up code generation.
    ///
    /// We can simply mark all instructions that are between diverging instructions and
    /// `JUMPDEST`s.
    #[instrument(name = "dce", level = "debug", skip_all)]
    fn mark_dead_code(&mut self) {
        let mut iter = self.insts.iter_mut_enumerated();
        while let Some((i, data)) = iter.next() {
            if data.is_diverging() {
                let mut end = i;
                for (j, data) in &mut iter {
                    end = j;
                    if data.is_reachable_jumpdest(self.has_dynamic_jumps) {
                        break;
                    }
                    data.flags |= InstFlags::DEAD_CODE;
                }
                let start = i + 1;
                if end > start {
                    debug!("found dead code: {start:?}..{end:?}");
                }
            }
        }
    }

    /// Calculates whether the bytecode suspend suspend execution.
    ///
    /// This can only happen if the bytecode contains `*CALL*` or `*CREATE*` instructions.
    #[instrument(name = "suspend", level = "debug", skip_all)]
    fn calc_may_suspend(&mut self) {
        let may_suspend = self.iter_insts().any(|(_, data)| data.may_suspend());
        self.may_suspend = may_suspend;
    }

    /// Constructs the sections in the bytecode.
    #[instrument(name = "sections", level = "debug", skip_all)]
    fn construct_sections(&mut self) {
        let mut analysis = SectionAnalysis::default();
        for inst in self.insts.indices() {
            if !self.inst(inst).is_dead_code() {
                analysis.process(self, inst);
            }
        }
        analysis.finish(self);
    }

    /// Constructs the sections in the bytecode.
    #[instrument(name = "sections", level = "debug", skip_all)]
    #[cfg(any())]
    fn construct_sections_default(&mut self) {
        for inst in &mut self.insts {
            let (inp, out) = inst.stack_io();
            let stack_diff = out as i16 - inp as i16;
            inst.section =
                Section { gas_cost: inst.base_gas as _, inputs: inp as _, max_growth: stack_diff }
        }
    }

    /// Returns the immediate value of the given instruction data, if any.
    /// Returns `None` if out of bounds too.
    pub(crate) fn get_imm(&self, data: &InstData) -> Option<&'a [u8]> {
        let imm_len = data.imm_len() as usize;
        if imm_len == 0 {
            return None;
        }
        let start = data.pc as usize + 1;
        self.code.get(start..start + imm_len)
    }

    /// Returns `true` if the given program counter is a valid jump destination.
    fn is_valid_jump(&self, pc: usize) -> bool {
        self.jumpdests.get(pc).as_deref().copied() == Some(true)
    }

    /// Returns `true` if the bytecode has dynamic jumps.
    pub(crate) fn has_dynamic_jumps(&self) -> bool {
        self.has_dynamic_jumps
    }

    /// Returns `true` if the bytecode may suspend execution, to be resumed later.
    pub(crate) fn may_suspend(&self) -> bool {
        self.may_suspend
    }

    /// Returns `true` if the bytecode is small.
    ///
    /// This is arbitrarily chosen to speed up compilation for larger contracts.
    pub(crate) fn is_small(&self) -> bool {
        self.insts.len() < 2000
    }

    /// Returns `true` if the instruction is diverging.
    pub(crate) fn is_instr_diverging(&self, inst: Inst) -> bool {
        self.insts[inst].is_diverging()
    }

    /// Converts a program counter (`self.code[pc]`) to an instruction (`self.inst(inst)`).
    #[inline]
    pub(crate) fn pc_to_inst(&self, pc: usize) -> Inst {
        match self.pc_to_inst.get(&(pc as u32)) {
            Some(&inst) => Inst::from_usize(inst as usize),
            None => panic!("pc out of bounds: {pc}"),
        }
    }

    /// Interns a U256 constant, returning its deduplicated index.
    pub(crate) fn intern_u256(&self, value: U256) -> U256Idx {
        self.u256_interner.borrow_mut().intern(value)
    }

    /// Returns the known constant value of a stack operand at the given instruction.
    ///
    /// `depth` 0 is TOS (first popped by this instruction), 1 is second, etc.
    /// Returns `None` if the value is unknown or the analysis didn't cover this instruction.
    pub(crate) fn const_operand(&self, inst: Inst, depth: usize) -> Option<U256> {
        let idx = self.stack_snapshots.get(inst)?.operand(depth)?;
        Some(*self.u256_interner.borrow().get(idx))
    }

    /// Returns the name for a basic block.
    pub(crate) fn op_block_name(&self, inst: Option<Inst>, name: &str) -> String {
        use std::fmt::Write;

        let Some(inst) = inst else {
            return format!("entry.{name}");
        };
        let data = self.inst(inst);

        let mut s = String::new();
        let _ = write!(s, "OP{}.{}", inst.index(), data.to_op());
        if !name.is_empty() {
            let _ = write!(s, ".{name}");
        }
        s
    }
}

/// A single instruction in the bytecode.
#[derive(Clone, Default)]
pub(crate) struct InstData {
    /// The opcode byte.
    pub(crate) opcode: u8,
    /// Flags.
    pub(crate) flags: InstFlags,
    /// The base gas cost of the opcode.
    ///
    /// This may not be the final/full gas cost of the opcode as it may also have a dynamic cost.
    base_gas: u16,
    /// Instruction-specific data:
    /// - if the instruction has immediate data, this is a packed offset+length into the bytecode;
    /// - `JUMP{,I} && STATIC_JUMP in kind`: the jump target, `Instr`;
    /// - `JUMPDEST`: `1` if the jump destination is reachable, `0` otherwise;
    /// - otherwise: no meaning.
    pub(crate) data: u32,
    /// The program counter, meaning `code[pc]` is this instruction's opcode.
    pub(crate) pc: u32,
    /// The section this instruction belongs to.
    pub(crate) section: Section,
}

impl PartialEq<u8> for InstData {
    #[inline]
    fn eq(&self, other: &u8) -> bool {
        self.opcode == *other
    }
}

impl PartialEq<InstData> for u8 {
    #[inline]
    fn eq(&self, other: &InstData) -> bool {
        *self == other.opcode
    }
}

impl InstData {
    /// Creates a new instruction data with the given opcode byte.
    /// Note that this may not be a valid instruction.
    #[inline]
    fn new(opcode: u8) -> Self {
        Self { opcode, ..Default::default() }
    }

    /// Returns the length of the immediate data of this instruction.
    #[inline]
    pub(crate) const fn imm_len(&self) -> u8 {
        min_imm_len(self.opcode)
    }

    /// Returns the number of input and output stack elements of this instruction.
    #[inline]
    pub(crate) fn stack_io(&self) -> (u8, u8) {
        let (mut inp, out) = stack_io(self.opcode);
        // For adjacent PUSH+JUMP, the PUSH is marked SKIP_LOGIC so the target is never on the
        // stack. Reduce inputs accordingly. Block-resolved jumps still have the target on the
        // stack, so their input count is unchanged.
        if self.is_legacy_static_jump()
            && !self.flags.contains(InstFlags::BLOCK_RESOLVED_JUMP)
            && !(self.opcode == op::JUMPI && self.flags.contains(InstFlags::INVALID_JUMP))
        {
            inp -= 1;
        }
        (inp, out)
    }

    /// Converts this instruction to a raw opcode. Note that the immediate data is not resolved.
    #[inline]
    pub(crate) const fn to_op(&self) -> Opcode<'static> {
        Opcode { opcode: self.opcode, immediate: None }
    }

    /// Converts this instruction to a raw opcode in the given bytecode.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn to_op_in<'a>(&self, bytecode: &Bytecode<'a>) -> Opcode<'a> {
        Opcode { opcode: self.opcode, immediate: bytecode.get_imm(self) }
    }

    /// Returns `true` if this instruction is a push instruction.
    #[inline]
    pub(crate) fn is_push(&self) -> bool {
        matches!(self.opcode, op::PUSH0..=op::PUSH32)
    }

    /// Returns `true` if this instruction is a legacy jump instruction (`JUMP`/`JUMPI`).
    #[inline]
    pub(crate) fn is_legacy_jump(&self) -> bool {
        matches!(self.opcode, op::JUMP | op::JUMPI)
    }

    /// Returns `true` if this instruction is a legacy jump instruction (`JUMP`/`JUMPI`), and the
    /// target known statically.
    #[inline]
    pub(crate) fn is_legacy_static_jump(&self) -> bool {
        self.is_legacy_jump() && self.flags.contains(InstFlags::STATIC_JUMP)
    }

    /// Returns `true` if this instruction is a `JUMPDEST`.
    #[inline]
    pub(crate) const fn is_jumpdest(&self) -> bool {
        self.opcode == op::JUMPDEST
    }

    /// Returns `true` if this instruction is a reachable `JUMPDEST`.
    #[inline]
    pub(crate) const fn is_reachable_jumpdest(&self, has_dynamic_jumps: bool) -> bool {
        self.is_jumpdest() && (has_dynamic_jumps || self.data == 1)
    }

    /// Returns `true` if this instruction is dead code.
    pub(crate) fn is_dead_code(&self) -> bool {
        self.flags.contains(InstFlags::DEAD_CODE)
    }

    /// Returns `true` if this instruction requires to know `gasleft()`.
    /// Note that this does not include CALL and CREATE.
    #[inline]
    pub(crate) fn requires_gasleft(&self, spec_id: SpecId) -> bool {
        // For SSTORE, see `revm_interpreter::gas::sstore_cost`.
        self.opcode == op::GAS
            || (self.opcode == op::SSTORE && spec_id.is_enabled_in(SpecId::ISTANBUL))
    }

    /// Returns `true` if we know that this instruction will branch or stop execution.
    #[inline]
    pub(crate) fn is_branching(&self) -> bool {
        self.is_legacy_jump() || self.is_diverging()
    }

    /// Returns `true` if we know that this instruction will stop execution.
    #[inline]
    pub(crate) fn is_diverging(&self) -> bool {
        #[cfg(test)]
        if self.opcode == TEST_SUSPEND {
            return false;
        }

        (self.opcode == op::JUMP && self.flags.contains(InstFlags::INVALID_JUMP))
            || self.flags.contains(InstFlags::DISABLED)
            || self.flags.contains(InstFlags::UNKNOWN)
            || matches!(
                self.opcode,
                op::STOP | op::RETURN | op::REVERT | op::INVALID | op::SELFDESTRUCT
            )
    }

    /// Returns `true` if this instruction may suspend execution.
    #[inline]
    pub(crate) const fn may_suspend(&self) -> bool {
        #[cfg(test)]
        if self.opcode == TEST_SUSPEND {
            return true;
        }

        matches!(
            self.opcode,
            op::CALL | op::CALLCODE | op::DELEGATECALL | op::STATICCALL | op::CREATE | op::CREATE2
        )
    }
}

bitflags::bitflags! {
    /// [`InstrData`] flags.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) struct InstFlags: u8 {
        /// The `JUMP`/`JUMPI` target is known at compile time.
        const STATIC_JUMP = 1 << 0;
        /// The jump target is known to be invalid.
        /// Always returns [`InstructionResult::InvalidJump`] at runtime.
        const INVALID_JUMP = 1 << 1;

        /// The instruction is disabled in this EVM version.
        /// Always returns [`InstructionResult::NotActivated`] at runtime.
        const DISABLED = 1 << 3;
        /// The instruction is unknown.
        /// Always returns [`InstructionResult::NotFound`] at runtime.
        const UNKNOWN = 1 << 4;

        /// The jump target was resolved by block analysis (not adjacent PUSH+JUMP).
        /// The target value is still on the stack and must be popped at runtime.
        const BLOCK_RESOLVED_JUMP = 1 << 5;

        /// Skip generating instruction logic, but keep the gas calculation.
        const SKIP_LOGIC = 1 << 6;
        /// Don't generate any code.
        const DEAD_CODE = 1 << 7;
    }
}

fn bitvec_as_bytes<T: bitvec::store::BitStore, O: bitvec::order::BitOrder>(
    bitvec: &BitVec<T, O>,
) -> &[u8] {
    slice_as_bytes(bitvec.as_raw_slice())
}

fn slice_as_bytes<T>(a: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), std::mem::size_of_val(a)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm_bytecode::opcode::OPCODE_INFO;

    #[test]
    fn test_suspend_is_free() {
        assert_eq!(OPCODE_INFO[TEST_SUSPEND as usize], None);
    }
}

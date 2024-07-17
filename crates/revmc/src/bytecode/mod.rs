//! Internal EVM bytecode and opcode representation.

use bitvec::vec::BitVec;
use revm_interpreter::opcode as op;
use revm_primitives::{hex, Eof, SpecId, EOF_MAGIC_BYTES};
use revmc_backend::Result;
use rustc_hash::FxHashMap;
use std::{borrow::Cow, fmt};

mod sections;
use sections::{Section, SectionAnalysis};

mod info;
pub use info::*;

mod opcode;
pub use opcode::*;

/// Noop opcode used to test suspend-resume.
#[cfg(test)]
pub(crate) const TEST_SUSPEND: u8 = 0x25;

// TODO: Use `indexvec`.
/// An EVM instruction is a high level internal representation of an EVM opcode.
///
/// This is an index into [`Bytecode`] instructions.
///
/// Also known as `ic`, or instruction counter; not to be confused with SSA `inst`s.
pub(crate) type Inst = usize;

#[doc(hidden)] // Not public API.
pub struct Bytecode<'a>(pub(crate) BytecodeInner<'a>);

#[derive(Debug)]
pub(crate) enum BytecodeInner<'a> {
    Legacy(LegacyBytecode<'a>),
    Eof(EofBytecode<'a>),
}

impl<'a> Bytecode<'a> {
    pub(crate) fn new(code: &'a [u8], spec_id: SpecId) -> Result<Self> {
        if spec_id.is_enabled_in(SpecId::PRAGUE_EOF) && code.starts_with(&EOF_MAGIC_BYTES) {
            Ok(Self(BytecodeInner::Eof(EofBytecode::decode(code, spec_id)?)))
        } else {
            Ok(Self(BytecodeInner::Legacy(LegacyBytecode::new(code, spec_id, None))))
        }
    }

    pub(crate) fn analyze(&mut self) -> Result<()> {
        match &mut self.0 {
            BytecodeInner::Legacy(bytecode) => bytecode.analyze(),
            BytecodeInner::Eof(bytecode) => bytecode.analyze(),
        }
    }

    pub(crate) fn as_legacy_slice(&self) -> &[LegacyBytecode<'a>] {
        match &self.0 {
            BytecodeInner::Legacy(bytecode) => std::slice::from_ref(bytecode),
            BytecodeInner::Eof(eof) => &eof.sections,
        }
    }

    pub(crate) fn as_eof(&self) -> Option<&Eof> {
        match &self.0 {
            BytecodeInner::Legacy(_) => None,
            BytecodeInner::Eof(eof) => Some(&eof.code),
        }
    }
}

impl fmt::Debug for Bytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            BytecodeInner::Legacy(bytecode) => bytecode.fmt(f),
            BytecodeInner::Eof(bytecode) => bytecode.fmt(f),
        }
    }
}

impl fmt::Display for Bytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            BytecodeInner::Legacy(bytecode) => bytecode.fmt(f),
            BytecodeInner::Eof(bytecode) => bytecode.fmt(f),
        }
    }
}

/// EVM bytecode.
pub(crate) struct LegacyBytecode<'a> {
    /// The original bytecode slice.
    pub(crate) code: &'a [u8],
    /// The instructions.
    insts: Vec<InstData>,
    /// `JUMPDEST` opcode map. `jumpdests[pc]` is `true` if `code[pc] == op::JUMPDEST`.
    jumpdests: BitVec,
    /// The [`SpecId`].
    pub(crate) spec_id: SpecId,
    /// Whether the bytecode contains dynamic jumps. Always false in EOF.
    has_dynamic_jumps: bool,
    /// Whether the bytecode will suspend execution.
    will_suspend: bool,
    /// Mapping from program counter to instruction.
    pc_to_inst: FxHashMap<u32, u32>,
    /// The EOF section index, if any.
    pub(crate) eof_section: Option<usize>,
}

impl<'a> LegacyBytecode<'a> {
    #[instrument(name = "new_bytecode", level = "debug", skip_all)]
    pub(crate) fn new(code: &'a [u8], spec_id: SpecId, eof_section: Option<usize>) -> Self {
        let is_eof = eof_section.is_some();

        let mut insts = Vec::with_capacity(code.len() + 8);
        // JUMPDEST analysis is not done in EOF.
        let mut jumpdests = if is_eof { BitVec::new() } else { BitVec::repeat(false, code.len()) };
        let mut pc_to_inst = FxHashMap::with_capacity_and_hasher(code.len(), Default::default());
        let op_infos = op_info_map(spec_id);
        for (inst, (pc, Opcode { opcode, immediate })) in
            OpcodesIter::new(code).with_pc().enumerate()
        {
            pc_to_inst.insert(pc as u32, inst as u32);

            if opcode == op::JUMPDEST && !is_eof {
                jumpdests.set(pc, true)
            }

            let mut data = 0;
            if let Some(imm) = immediate {
                // `pc` is at `opcode` right now, add 1 for the data.
                data = Immediate::pack(pc + 1, imm.len());
            }

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
            will_suspend: false,
            pc_to_inst,
            eof_section,
        };

        // Pad code to ensure there is at least one diverging instruction.
        if !is_eof && bytecode.insts.last().map_or(true, |last| !last.is_diverging(false)) {
            bytecode.insts.push(InstData::new(op::STOP));
        }

        bytecode
    }

    /// Returns an iterator over the opcodes.
    #[inline]
    pub(crate) fn opcodes(&self) -> OpcodesIter<'a> {
        OpcodesIter::new(self.code)
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
    ) -> impl DoubleEndedIterator<Item = (usize, &InstData)> + Clone + '_ {
        self.iter_all_insts().filter(|(_, data)| !data.is_dead_code())
    }

    /// Returns an iterator over the instructions.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn iter_mut_insts(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (usize, &mut InstData)> + '_ {
        self.iter_mut_all_insts().filter(|(_, data)| !data.is_dead_code())
    }

    /// Returns an iterator over all the instructions, including dead code.
    #[inline]
    pub(crate) fn iter_all_insts(
        &self,
    ) -> impl DoubleEndedIterator<Item = (usize, &InstData)> + ExactSizeIterator + Clone + '_ {
        self.insts.iter().enumerate()
    }

    /// Returns an iterator over all the instructions, including dead code.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn iter_mut_all_insts(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (usize, &mut InstData)> + ExactSizeIterator + '_ {
        self.insts.iter_mut().enumerate()
    }

    /// Runs a list of analysis passes on the instructions.
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn analyze(&mut self) -> Result<()> {
        if !self.is_eof() {
            self.static_jump_analysis();
            // NOTE: `mark_dead_code` must run after `static_jump_analysis` as it can mark
            // unreachable `JUMPDEST`s as dead code.
            self.mark_dead_code();
        }

        self.calc_will_suspend();
        self.construct_sections();

        Ok(())
    }

    /// Mark `PUSH<N>` followed by `JUMP[I]` as `STATIC_JUMP` and resolve the target.
    #[instrument(name = "sj", level = "debug", skip_all)]
    fn static_jump_analysis(&mut self) {
        debug_assert!(!self.is_eof());

        for jump_inst in 0..self.insts.len() {
            let jump = &self.insts[jump_inst];
            let Some(push_inst) = jump_inst.checked_sub(1) else {
                if jump.is_legacy_jump() {
                    trace!(jump_inst, target=?None::<()>, "found jump");
                    self.has_dynamic_jumps = true;
                }
                continue;
            };

            let push = &self.insts[push_inst];
            if !(push.is_push() && jump.is_legacy_jump()) {
                if jump.is_legacy_jump() {
                    trace!(jump_inst, target=?None::<()>, "found jump");
                    self.has_dynamic_jumps = true;
                }
                continue;
            }

            let imm_data = push.data;
            let imm = self.get_imm(imm_data);
            self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP;

            const USIZE_SIZE: usize = std::mem::size_of::<usize>();
            if imm.len() > USIZE_SIZE {
                trace!(jump_inst, "jump target too large");
                self.insts[jump_inst].flags |= InstFlags::INVALID_JUMP;
                continue;
            }

            let mut padded = [0; USIZE_SIZE];
            padded[USIZE_SIZE - imm.len()..].copy_from_slice(imm);
            let target_pc = usize::from_be_bytes(padded);
            if !self.is_valid_jump(target_pc) {
                trace!(jump_inst, target_pc, "invalid jump target");
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
                 jump_inst={jump_inst} target_pc={target_pc} target={target}",
            );
            self.insts[target].data = 1;

            // Set the target on the `JUMP` instruction.
            trace!(jump_inst, target, "found jump");
            self.insts[jump_inst].data = target as u32;
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
        debug_assert!(!self.is_eof());

        let mut iter = self.insts.iter_mut().enumerate();
        while let Some((i, data)) = iter.next() {
            if data.is_diverging(false) {
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
                    debug!("found dead code: {start}..{end}");
                }
            }
        }
    }

    /// Calculates whether the bytecode will suspend execution.
    ///
    /// This can only happen if the bytecode contains `*CALL*` or `*CREATE*` instructions.
    #[instrument(name = "suspend", level = "debug", skip_all)]
    fn calc_will_suspend(&mut self) {
        let is_eof = self.is_eof();
        let will_suspend = self.iter_insts().any(|(_, data)| data.will_suspend(is_eof));
        self.will_suspend = will_suspend;
    }

    /// Constructs the sections in the bytecode.
    #[instrument(name = "sections", level = "debug", skip_all)]
    fn construct_sections(&mut self) {
        let mut analysis = SectionAnalysis::default();
        for inst in 0..self.insts.len() {
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
    pub(crate) fn get_imm_of(&self, instr_data: &InstData) -> Option<&'a [u8]> {
        (instr_data.imm_len() > 0).then(|| self.get_imm(instr_data.data))
    }

    fn get_imm(&self, data: u32) -> &'a [u8] {
        let (offset, len) = Immediate::unpack(data);
        &self.code[offset..offset + len]
    }

    /// Returns `true` if the given program counter is a valid jump destination.
    fn is_valid_jump(&self, pc: usize) -> bool {
        self.jumpdests.get(pc).as_deref().copied() == Some(true)
    }

    /// Returns `true` if the bytecode has dynamic jumps.
    pub(crate) fn has_dynamic_jumps(&self) -> bool {
        self.has_dynamic_jumps
    }

    /// Returns `true` if the bytecode will suspend execution, to be resumed later.
    pub(crate) fn will_suspend(&self) -> bool {
        self.will_suspend
    }

    /// Returns `true` if the bytecode is EOF.
    pub(crate) fn is_eof(&self) -> bool {
        self.eof_section.is_some()
    }

    /// Returns `true` if the bytecode is small.
    ///
    /// This is arbitrarily chosen to speed up compilation for larger contracts.
    pub(crate) fn is_small(&self) -> bool {
        self.insts.len() < 2000
    }

    /// Returns `true` if the instruction is diverging.
    pub(crate) fn is_instr_diverging(&self, inst: Inst) -> bool {
        self.insts[inst].is_diverging(self.is_eof())
    }

    /// Converts a program counter (`self.code[ic]`) to an instruction (`self.inst(pc)`).
    #[inline]
    pub(crate) fn pc_to_inst(&self, pc: usize) -> usize {
        self.pc_to_inst[&(pc as u32)] as usize
    }

    /*
    /// Converts an instruction to a program counter.
    fn inst_to_pc(&self, inst: Inst) -> usize {
        self.insts[inst].pc as usize
    }
    */
}

impl fmt::Display for LegacyBytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header = format!("{:^6} | {:^6} | {:^80} | {}", "ic", "pc", "opcode", "instruction");
        writeln!(f, "{header}")?;
        writeln!(f, "{}", "-".repeat(header.len()))?;
        for (inst, (pc, opcode)) in self.opcodes().with_pc().enumerate() {
            let data = self.inst(inst);
            let opcode = opcode.to_string();
            writeln!(f, "{inst:>6} | {pc:>6} | {opcode:<80} | {data:?}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for LegacyBytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bytecode")
            .field("code", &hex::encode(self.code))
            .field("insts", &self.insts)
            .field("jumpdests", &hex::encode(bitvec_as_bytes(&self.jumpdests)))
            .field("spec_id", &self.spec_id)
            .field("has_dynamic_jumps", &self.has_dynamic_jumps)
            .field("will_suspend", &self.will_suspend)
            .finish()
    }
}

#[derive(Debug)]
pub(crate) struct EofBytecode<'a> {
    pub(crate) code: Cow<'a, Eof>,
    pub(crate) sections: Vec<LegacyBytecode<'a>>,
}

impl<'a> EofBytecode<'a> {
    // TODO: Accept revm Bytecode in the compiler
    #[allow(dead_code)]
    fn new(code: &'a Eof, spec_id: SpecId) -> Self {
        Self { code: Cow::Borrowed(code), sections: vec![] }.make_sections(spec_id)
    }

    fn decode(code: &'a [u8], spec_id: SpecId) -> Result<Self> {
        let code = Eof::decode(code.to_vec().into())?;
        Ok(Self { code: Cow::Owned(code), sections: vec![] }.make_sections(spec_id))
    }

    fn make_sections(mut self, spec_id: SpecId) -> Self {
        self.sections = self
            .code
            .body
            .code_section
            .iter()
            .enumerate()
            .map(|(section, code)| {
                // SAFETY: Code section `Bytes` outlives `self`.
                let code = unsafe { std::mem::transmute::<&[u8], &[u8]>(&code[..]) };
                LegacyBytecode::new(code, spec_id, Some(section))
            })
            .collect();
        self
    }

    fn analyze(&mut self) -> Result<()> {
        for section in &mut self.sections {
            section.analyze()?;
        }
        Ok(())
    }
}

impl fmt::Display for EofBytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, section) in self.sections.iter().enumerate() {
            writeln!(f, "# Section {i}")?;
            writeln!(f, "{section}")?;
        }
        Ok(())
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

impl fmt::Debug for InstData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InstData")
            .field("opcode", &self.to_op())
            .field("flags", &format_args!("{:?}", self.flags))
            .field("data", &self.data)
            .field("pc", &self.pc)
            .field("section", &self.section)
            .finish()
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
        if self.is_legacy_static_jump()
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
    pub(crate) fn to_op_in<'a>(&self, bytecode: &LegacyBytecode<'a>) -> Opcode<'a> {
        Opcode { opcode: self.opcode, immediate: bytecode.get_imm_of(self) }
    }

    /// Returns `true` if this instruction is a push instruction.
    #[inline]
    pub(crate) fn is_push(&self) -> bool {
        matches!(self.opcode, op::PUSH0..=op::PUSH32)
    }

    /// Returns `true` if this instruction is a jump instruction.
    #[inline]
    fn is_jump(&self, is_eof: bool) -> bool {
        if is_eof {
            self.is_eof_jump()
        } else {
            self.is_legacy_jump()
        }
    }

    /// Returns `true` if this instruction is an EOF jump instruction (`RJUMP`/`RJUMPI`/`RJUMPV`).
    #[inline]
    fn is_eof_jump(&self) -> bool {
        matches!(self.opcode, op::RJUMP | op::RJUMPI | op::RJUMPV)
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
    pub(crate) fn is_branching(&self, is_eof: bool) -> bool {
        self.is_jump(is_eof) || self.is_diverging(is_eof)
    }

    /// Returns `true` if we know that this instruction will stop execution.
    #[inline]
    pub(crate) fn is_diverging(&self, is_eof: bool) -> bool {
        #[cfg(test)]
        if self.opcode == TEST_SUSPEND {
            return false;
        }

        (self.opcode == op::JUMP && self.flags.contains(InstFlags::INVALID_JUMP))
            || self.flags.contains(InstFlags::DISABLED)
            || self.flags.contains(InstFlags::UNKNOWN)
            || matches!(self.opcode, op::STOP | op::RETURN | op::REVERT | op::INVALID)
            || (!is_eof && matches!(self.opcode, op::SELFDESTRUCT))
            || (is_eof && matches!(self.opcode, op::RETF | op::RETURNCONTRACT))
    }

    /// Returns `true` if this instruction will suspend execution.
    #[inline]
    pub(crate) const fn will_suspend(&self, is_eof: bool) -> bool {
        #[cfg(test)]
        if self.opcode == TEST_SUSPEND {
            return true;
        }

        if is_eof {
            matches!(
                self.opcode,
                op::EXTCALL | op::EXTDELEGATECALL | op::EXTSTATICCALL | op::EOFCREATE
            )
        } else {
            matches!(
                self.opcode,
                op::CALL
                    | op::CALLCODE
                    | op::DELEGATECALL
                    | op::STATICCALL
                    | op::CREATE
                    | op::CREATE2
            )
        }
    }
}

bitflags::bitflags! {
    /// [`InstrData`] flags.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) struct InstFlags: u8 {
        /// The `JUMP`/`JUMPI` target is known at compile time.
        /// This is implied for other jump instructions which are always static.
        const STATIC_JUMP = 1 << 0;
        /// The jump target is known to be invalid.
        /// Always returns [`InstructionResult::InvalidJump`] at runtime.
        const INVALID_JUMP = 1 << 1;

        /// The instruction is disabled in this EVM version.
        /// Always returns [`InstructionResult::NotActivated`] at runtime.
        const DISABLED = 1 << 2;
        /// The instruction is unknown.
        /// Always returns [`InstructionResult::NotFound`] at runtime.
        const UNKNOWN = 1 << 3;

        /// Skip generating instruction logic, but keep the gas calculation.
        const SKIP_LOGIC = 1 << 4;
        /// Don't generate any code.
        const DEAD_CODE = 1 << 5;
    }
}

/// Packed representation of an immediate value.
struct Immediate;

impl Immediate {
    fn pack(offset: usize, len: usize) -> u32 {
        debug_assert!(offset <= 1 << 26, "imm offset overflow: {offset} > (1 << 26)");
        debug_assert!(len <= 1 << 6, "imm length overflow: {len} > (1 << 6)");
        ((offset as u32) << 6) | len as u32
    }

    // `(offset, len)`
    fn unpack(data: u32) -> (usize, usize) {
        ((data >> 6) as usize, (data & ((1 << 6) - 1)) as usize)
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

    #[test]
    fn imm_packing() {
        let assert = |offset, len| {
            let packed = Immediate::pack(offset, len);
            assert_eq!(Immediate::unpack(packed), (offset, len), "packed: {packed}");
        };
        assert(0, 0);
        assert(0, 1);
        assert(0, 31);
        assert(0, 32);
        assert(1, 0);
        assert(1, 1);
        assert(1, 31);
        assert(1, 32);
    }

    #[test]
    fn test_suspend_is_free() {
        assert_eq!(op::OPCODE_INFO_JUMPTABLE[TEST_SUSPEND as usize], None);
    }
}

//! Internal EVM bytecode and opcode representation.

use bitvec::vec::BitVec;
use revm_bytecode::opcode as op;
use revm_primitives::{hardfork::SpecId, hex};
use revmc_backend::Result;
use rustc_hash::FxHashMap;
use std::fmt;

mod sections;
use sections::{Section, SectionAnalysis};

mod info;
pub use info::*;

mod opcode;
pub use opcode::*;

/// Noop opcode used to test suspend-resume.
#[cfg(any(feature = "__fuzzing", test))]
pub(crate) const TEST_SUSPEND: u8 = 0x25;

// TODO: Use `indexvec`.
/// An EVM instruction is a high level internal representation of an EVM opcode.
///
/// This is an index into [`Bytecode`] instructions.
///
/// Also known as `ic`, or instruction counter; not to be confused with SSA `inst`s.
pub(crate) type Inst = usize;

/// EVM bytecode.
#[doc(hidden)] // Not public API.
pub struct Bytecode<'a> {
    /// The original bytecode slice.
    pub(crate) code: &'a [u8],
    /// The instructions.
    insts: Vec<InstData>,
    /// `JUMPDEST` opcode map. `jumpdests[pc]` is `true` if `code[pc] == op::JUMPDEST`.
    jumpdests: BitVec,
    /// The [`SpecId`].
    pub(crate) spec_id: SpecId,
    /// Whether the bytecode contains dynamic jumps.
    has_dynamic_jumps: bool,
    /// Whether the bytecode may suspend execution.
    may_suspend: bool,
    /// Mapping from program counter to instruction.
    pc_to_inst: FxHashMap<u32, u32>,
}

impl<'a> Bytecode<'a> {
    #[instrument(name = "new_bytecode", level = "debug", skip_all)]
    pub(crate) fn new(code: &'a [u8], spec_id: SpecId) -> Self {
        let mut insts = Vec::with_capacity(code.len() + 8);
        let mut jumpdests = BitVec::repeat(false, code.len());
        let mut pc_to_inst = FxHashMap::with_capacity_and_hasher(code.len(), Default::default());
        let op_infos = op_info_map(spec_id);
        for (inst, (pc, Opcode { opcode, immediate: _ })) in
            OpcodesIter::new(code, spec_id).with_pc().enumerate()
        {
            pc_to_inst.insert(pc as u32, inst as u32);

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
            pc_to_inst,
        };

        // Pad code to ensure there is at least one diverging instruction.
        if bytecode.insts.last().is_none_or(|last| !last.is_diverging()) {
            bytecode.insts.push(InstData::new(op::STOP));
        }

        bytecode
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
        self.static_jump_analysis();
        // NOTE: `mark_dead_code` must run after `static_jump_analysis` as it can mark
        // unreachable `JUMPDEST`s as dead code.
        self.mark_dead_code();

        self.calc_may_suspend();

        self.construct_sections();

        Ok(())
    }

    /// Mark `PUSH<N>` followed by `JUMP[I]` as `STATIC_JUMP` and resolve the target.
    #[instrument(name = "sj", level = "debug", skip_all)]
    fn static_jump_analysis(&mut self) {
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

            let imm_opt = self.get_imm(push);
            if push.opcode != op::PUSH0 && imm_opt.is_none() {
                continue;
            }
            let imm = imm_opt.unwrap_or(&[]);
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
        let mut iter = self.insts.iter_mut().enumerate();
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
                    debug!("found dead code: {start}..{end}");
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
    pub(crate) fn pc_to_inst(&self, pc: usize) -> usize {
        match self.pc_to_inst.get(&(pc as u32)) {
            Some(&inst) => inst as usize,
            None => panic!("pc out of bounds: {pc}"),
        }
    }

    /// Returns the name for a basic block.
    pub(crate) fn op_block_name(&self, inst: usize, name: &str) -> String {
        use std::fmt::Write;

        if inst == usize::MAX {
            return format!("entry.{name}");
        }
        let data = self.inst(inst);

        let mut s = String::new();
        let _ = write!(s, "OP{inst}.{}", data.to_op());
        if !name.is_empty() {
            let _ = write!(s, ".{name}");
        }
        s
    }
}

impl fmt::Display for Bytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use std::fmt::Write;

        // First pass: collect lines with their text and comments.
        let mut lines: Vec<(String, String)> = Vec::new();

        lines.push((
            String::new(),
            format!(
                "spec_id={}, has_dynamic_jumps={}, may_suspend={}",
                self.spec_id, self.has_dynamic_jumps, self.may_suspend,
            ),
        ));
        lines.push((String::new(), String::new()));

        // Pre-pass: build inst -> block index map.
        let mut inst_to_block = FxHashMap::default();
        {
            let mut block_idx = 0usize;
            let mut need_header = true;
            for (inst, data) in self.iter_all_insts() {
                if data.is_dead_code() {
                    continue;
                }
                if !data.section.is_empty() || need_header {
                    inst_to_block.insert(inst, block_idx);
                    block_idx += 1;
                    need_header = false;
                }
                if data.is_branching() {
                    need_header = true;
                }
            }
        }

        let mut block_idx = 0usize;
        let mut need_header = true;
        for (inst, data) in self.iter_all_insts() {
            if data.is_dead_code() {
                continue;
            }

            // Block header.
            if !data.section.is_empty() || need_header {
                if inst > 0 {
                    lines.push((String::new(), String::new()));
                }
                let mut header = format!("bb{block_idx}:");
                let mut comment = String::new();
                if !data.section.is_empty() {
                    write!(
                        comment,
                        "gas={}, stack_in={}, max_growth={}",
                        data.section.gas_cost, data.section.inputs, data.section.max_growth,
                    )
                    .unwrap();
                }
                // Pad header to align with indented instructions.
                while header.len() < 2 {
                    header.push(' ');
                }
                lines.push((header, comment));
                block_idx += 1;
                need_header = false;
            }

            // Instruction text.
            let mut text = String::from("  ");
            let opcode = data.to_op_in(self);
            write!(text, "{opcode}").unwrap();
            if data.flags.contains(InstFlags::INVALID_JUMP) {
                text.push_str(" -> INVALID");
            } else if data.is_legacy_static_jump() {
                let target = data.data as usize;
                match inst_to_block.get(&target) {
                    Some(b) => write!(text, " bb{b}").unwrap(),
                    None => write!(text, " inst {target}").unwrap(),
                }
            }

            // Comment with pc and flags/behavior.
            let mut comment = format!("pc={}", data.pc);
            let flags = data.flags;
            if flags.contains(InstFlags::SKIP_LOGIC) {
                comment.push_str(", skip");
            }
            if flags.contains(InstFlags::DEAD_CODE) {
                comment.push_str(", dead");
            }
            if flags.contains(InstFlags::DISABLED) {
                comment.push_str(", disabled");
            }
            if flags.contains(InstFlags::UNKNOWN) {
                comment.push_str(", unknown");
            }
            if flags.contains(InstFlags::INVALID_JUMP) {
                comment.push_str(", invalid_jump");
            }
            if data.may_suspend() {
                comment.push_str(", suspends");
            }
            if data.is_reachable_jumpdest(self.has_dynamic_jumps) {
                comment.push_str(", reachable");
            }

            lines.push((text, comment));

            if data.is_branching() {
                need_header = true;
            }
        }

        // Second pass: find max text width and write with aligned comments.
        let max_text_width = lines.iter().map(|(t, _)| t.len()).max().unwrap_or(0);
        let comment_col = max_text_width.clamp(4, 20);
        for (text, comment) in &lines {
            if text.is_empty() && comment.is_empty() {
                writeln!(f)?;
            } else if comment.is_empty() {
                writeln!(f, "{text}")?;
            } else {
                writeln!(f, "{text:<comment_col$} ; {comment}")?;
            }
        }

        Ok(())
    }
}

impl fmt::Debug for Bytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bytecode")
            .field("code", &hex::encode(self.code))
            .field("insts", &self.insts)
            .field("jumpdests", &hex::encode(bitvec_as_bytes(&self.jumpdests)))
            .field("spec_id", &self.spec_id)
            .field("has_dynamic_jumps", &self.has_dynamic_jumps)
            .field("may_suspend", &self.may_suspend)
            .finish()
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

    #[test]
    fn display_format() {
        // pc=0: PUSH1 0x01 (condition)
        // pc=2: PUSH1 0x07 (jump target, skip_logic)
        // pc=4: JUMPI -> JUMPDEST at pc=7 (static jump)
        // pc=5: STOP (fallthrough)
        // pc=6: ADD (dead code, between STOP and JUMPDEST)
        // pc=7: JUMPDEST (reachable)
        // pc=8: PUSH1 0x02
        // pc=10: STOP
        let code: &[u8] = &[
            op::PUSH1,
            0x01, // push condition
            op::PUSH1,
            0x07,      // push jump target
            op::JUMPI, // conditional jump
            op::STOP,  // fallthrough
            op::ADD,   // dead code
            op::JUMPDEST,
            op::PUSH1,
            0x02, // jumped-to block
            op::STOP,
        ];
        let mut bytecode = Bytecode::new(code, SpecId::OSAKA);
        bytecode.analyze().unwrap();

        let actual = format!("{bytecode}");
        snapbox::assert_data_eq!(
            actual,
            snapbox::str![[r#"
             ; spec_id=Osaka, has_dynamic_jumps=false, may_suspend=false

bb0:         ; gas=16, stack_in=0, max_growth=2
  PUSH1 0x01 ; pc=0
  PUSH1 0x07 ; pc=2, skip
  JUMPI bb2  ; pc=4

bb1:
  STOP       ; pc=5

bb2:         ; gas=4, stack_in=0, max_growth=1
  JUMPDEST   ; pc=7, reachable
  PUSH1 0x02 ; pc=8
  STOP       ; pc=10

"#]]
        );
    }
}

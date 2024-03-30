//! Internal EVM bytecode and opcode representation.

use bitvec::vec::BitVec;
use color_eyre::Result;
use revm_interpreter::opcode as op;
use revm_primitives::SpecId;
use std::fmt;

mod info;
pub use info::*;

mod opcode;
pub use opcode::*;

// TODO: Use `indexvec`.
/// An EVM instruction is a high level internal representation of an EVM opcode.
///
/// This is an index into [`Bytecode`] instructions.
///
/// Also known as `ic`, or instruction counter; not to be confused with SSA `inst`s.
pub(crate) type Inst = usize;

/// EVM bytecode.
#[derive(Debug)]
pub(crate) struct Bytecode<'a> {
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
}

impl<'a> Bytecode<'a> {
    pub(crate) fn new(code: &'a [u8], spec_id: SpecId) -> Self {
        let mut insts = Vec::with_capacity(code.len() + 1);
        let mut jumpdests = BitVec::repeat(false, code.len());
        let op_infos = op_info_map(spec_id);
        for (pc, Opcode { opcode, immediate }) in OpcodesIter::new(code).with_pc() {
            let mut data = 0;
            match opcode {
                op::JUMPDEST => jumpdests.set(pc, true),
                _ => {
                    if let Some(imm) = immediate {
                        // `pc` is at `opcode` right now, add 1 for the data.
                        data = Immediate::pack(pc + 1, imm.len());
                    }
                }
            }

            let mut flags = InstrFlags::empty();
            let info = op_infos[opcode as usize];
            if info.is_unknown() {
                flags |= InstrFlags::UNKNOWN;
            }
            if info.is_disabled() {
                flags |= InstrFlags::DISABLED;
            }
            let static_gas = info.static_gas().unwrap_or(u16::MAX);

            insts.push(InstData { opcode, flags, static_gas, data, pc: pc as u32 });
        }

        let mut bytecode = Self { code, insts, jumpdests, spec_id, has_dynamic_jumps: false };

        // Pad code to ensure there is at least one diverging instruction.
        if bytecode.insts.last().map_or(true, |last| !last.is_diverging(bytecode.is_eof())) {
            bytecode.insts.push(InstData::new(op::STOP));
        }

        bytecode
    }

    /// Returns the instruction at the given instruction counter.
    #[inline]
    pub(crate) fn inst(&self, inst: Inst) -> &InstData {
        &self.insts[inst]
    }

    /// Returns a mutable reference the instruction at the given instruction counter.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn instr_mut(&mut self, inst: Inst) -> &mut InstData {
        &mut self.insts[inst]
    }

    /// Returns an iterator over the instructions.
    #[inline]
    pub(crate) fn iter_insts(&self) -> impl Iterator<Item = (usize, &InstData)> + '_ {
        self.iter_all_insts().filter(|(_, data)| !data.flags.contains(InstrFlags::DEAD_CODE))
    }

    /// Returns an iterator over all the instructions, including dead code.
    #[inline]
    pub(crate) fn iter_all_insts(&self) -> impl Iterator<Item = (usize, &InstData)> + '_ {
        self.insts.iter().enumerate()
    }

    /// Runs a list of analysis passes on the instructions.
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn analyze(&mut self) -> Result<()> {
        trace_time!("static_jump_analysis", || self.static_jump_analysis());
        // NOTE: `mark_dead_code` must run after `static_jump_analysis` as it can mark unreachable
        // `JUMPDEST`s as dead code.
        trace_time!("mark_dead_code", || self.mark_dead_code());

        Ok(())
    }

    /// Mark `PUSH<N>` followed by `JUMP[I]` as `STATIC_JUMP` and resolve the target.
    #[instrument(name = "sj", level = "debug", skip_all)]
    fn static_jump_analysis(&mut self) {
        for inst in 0..self.insts.len() {
            let jump_inst = inst + 1;
            let Some(jump) = self.insts.get(jump_inst) else { continue };
            let push = &self.insts[inst];
            if !(push.is_push() && jump.is_jump()) {
                if jump.is_jump() {
                    debug!(jump_inst, "found dynamic jump");
                    self.has_dynamic_jumps = true;
                }
                continue;
            }

            let imm_data = push.data;
            let imm = self.get_imm(imm_data);
            self.insts[jump_inst].flags |= InstrFlags::STATIC_JUMP;

            const USIZE_SIZE: usize = std::mem::size_of::<usize>();
            if imm.len() > USIZE_SIZE {
                debug!(jump_inst, "jump target too large");
                self.insts[jump_inst].flags |= InstrFlags::INVALID_JUMP;
                continue;
            }

            let mut padded = [0; USIZE_SIZE];
            padded[USIZE_SIZE - imm.len()..].copy_from_slice(imm);
            let target_pc = usize::from_be_bytes(padded);
            if !self.is_valid_jump(target_pc) {
                debug!(jump_inst, target_pc, "invalid jump target");
                self.insts[jump_inst].flags |= InstrFlags::INVALID_JUMP;
                continue;
            }

            self.insts[inst].flags |= InstrFlags::SKIP_LOGIC;
            let target = self.pc_to_inst(target_pc);

            // Mark the `JUMPDEST` as reachable.
            if !self.is_eof() {
                debug_assert_eq!(
                    self.insts[target],
                    op::JUMPDEST,
                    "is_valid_jump returned true for non-JUMPDEST: \
                     jump_inst={jump_inst} target_pc={target_pc} target={target}",
                );
                self.insts[target].data = 1;
            }

            // Set the target on the `JUMP` instruction.
            debug!(jump_inst, target, "resolved jump target");
            self.insts[jump_inst].data = target as u32;
        }
    }

    /// Mark unreachable instructions as `DEAD_CODE` to not generate any code for them.
    ///
    /// This pass is technically unnecessary as the backend will very likely optimize any
    /// unreachable code that we generate, but this is trivial for us to do and significantly speeds
    /// up code generation.
    ///
    /// Before EOF, we can simply mark all instructions that are between diverging instructions and
    /// `JUMPDEST`s.
    ///
    /// After EOF, TODO.
    #[instrument(name = "dce", level = "debug", skip_all)]
    fn mark_dead_code(&mut self) {
        let is_eof = self.is_eof();
        let mut iter = self.insts.iter_mut().enumerate();
        while let Some((i, data)) = iter.next() {
            if data.is_diverging(is_eof) {
                let mut end = i;
                for (j, data) in &mut iter {
                    end = j;
                    if data.is_reachable_jumpdest(self.has_dynamic_jumps) {
                        break;
                    }
                    data.flags |= InstrFlags::DEAD_CODE;
                }
                let start = i + 1;
                if end > start {
                    debug!("found dead code: {start}..{end}");
                }
            }
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

    /// Returns `true` if the bytecode is EOF.
    pub(crate) fn is_eof(&self) -> bool {
        false
    }

    /// Returns `true` if the instruction is diverging.
    pub(crate) fn is_instr_diverging(&self, inst: Inst) -> bool {
        self.insts[inst].is_diverging(self.is_eof())
    }

    // TODO: is it worth it to make this a map?
    /// Converts a program counter (`self.code[ic]`) to an instruction (`self.inst(pc)`).
    fn pc_to_inst(&self, pc: usize) -> usize {
        debug_assert!(pc < self.code.len(), "pc out of bounds: {pc}");
        let (inst, (_, op)) = OpcodesIter::new(self.code)
            .with_pc() // pc
            .enumerate() // inst
            // Don't go until the end because we know `pc` are yielded in order.
            .take_while(|(_ic, (pc2, _))| *pc2 <= pc)
            .find(|(_ic, (pc2, _))| *pc2 == pc)
            .unwrap_or_else(|| panic!("pc to inst conversion failed; pc={pc}"));
        debug_assert_eq!(self.insts[inst].to_op_in(self), op, "pc={pc} inst={inst}");
        inst
    }
}

/// A single instruction in the bytecode.
#[derive(Clone)]
pub(crate) struct InstData {
    /// The opcode byte.
    pub(crate) opcode: u8,
    /// Flags.
    pub(crate) flags: InstrFlags,
    /// Static gas. Stored as `u16::MAX` if the gas is dynamic.
    static_gas: u16,
    /// Instruction-specific data:
    /// - if the instruction has immediate data, this is a packed offset+length into the bytecode;
    /// - `JUMP{,I} && STATIC_JUMP in kind`: the jump target, `Instr`;
    /// - `JUMPDEST`: `1` if the jump destination is reachable, `0` otherwise;
    /// - otherwise: no meaning.
    pub(crate) data: u32,
    /// The program counter, meaning `code[pc]` is this instruction's opcode.
    pub(crate) pc: u32,
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
        f.debug_struct("OpcodeData")
            .field("opcode", &self.to_op())
            .field("flags", &self.flags)
            .field("data", &self.data)
            .finish()
    }
}

impl InstData {
    /// Creates a new instruction data with the given opcode byte.
    /// Note that this may not be a valid instruction.
    #[inline]
    const fn new(opcode: u8) -> Self {
        Self { opcode, flags: InstrFlags::empty(), static_gas: 0, data: 0, pc: 0 }
    }

    /// Returns the length of the immediate data of this instruction.
    #[inline]
    pub(crate) const fn imm_len(&self) -> usize {
        imm_len(self.opcode)
    }

    /// Converts this instruction to a raw opcode. Note that the immediate data is not resolved.
    #[inline]
    pub(crate) const fn to_op(&self) -> Opcode<'static> {
        Opcode { opcode: self.opcode, immediate: None }
    }

    /// Converts this instruction to a raw opcode in the given bytecode.
    #[inline]
    pub(crate) fn to_op_in<'a>(&self, bytecode: &Bytecode<'a>) -> Opcode<'a> {
        Opcode { opcode: self.opcode, immediate: bytecode.get_imm_of(self) }
    }

    /// Returns the static gas for this instruction, if any.
    #[inline]
    pub(crate) fn static_gas(&self) -> Option<u16> {
        (self.static_gas != u16::MAX).then_some(self.static_gas)
    }

    /// Returns `true` if this instruction is a push instruction.
    pub(crate) fn is_push(&self) -> bool {
        matches!(self.opcode, op::PUSH1..=op::PUSH32)
    }

    /// Returns `true` if this instruction is a jump instruction.
    pub(crate) fn is_jump(&self) -> bool {
        matches!(self.opcode, op::JUMP | op::JUMPI)
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
        self.flags.contains(InstrFlags::DEAD_CODE)
    }

    /// Returns `true` if we know that this instruction will stop execution.
    #[inline]
    pub(crate) const fn is_diverging(&self, is_eof: bool) -> bool {
        // TODO: SELFDESTRUCT will not be diverging in the future.
        self.flags.contains(InstrFlags::INVALID_JUMP)
            || self.flags.contains(InstrFlags::DISABLED)
            || self.flags.contains(InstrFlags::UNKNOWN)
            || matches!(self.opcode, op::STOP | op::RETURN | op::REVERT | op::INVALID)
            || (self.opcode == op::SELFDESTRUCT && !is_eof)
    }
}

bitflags::bitflags! {
    /// [`InstrData`] flags.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct InstrFlags: u8 {
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
}

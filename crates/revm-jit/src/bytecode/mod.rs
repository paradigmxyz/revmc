//! Internal EVM bytecode and opcode representation.

use bitvec::vec::BitVec;
use color_eyre::{eyre::bail, Result};
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
/// Also known as `ic`, or instruction counter; not to be confused with SSA `instr`s.
pub(crate) type Instr = usize;

/// EVM bytecode.
#[derive(Debug)]
pub(crate) struct Bytecode<'a> {
    /// The original bytecode slice.
    pub(crate) code: &'a [u8],
    /// The instructions.
    instrs: Vec<InstrData>,
    /// `JUMPDEST` map.
    jumpdests: BitVec,
    /// The [`SpecId`].
    pub(crate) spec_id: SpecId,
}

impl<'a> Bytecode<'a> {
    pub(crate) fn new(code: &'a [u8], spec_id: SpecId) -> Self {
        let mut instrs = Vec::with_capacity(code.len() + 1);
        let mut jumpdests = BitVec::repeat(false, code.len());
        let op_infos = op_info_map(spec_id);
        for (pc, Opcode { opcode, immediate }) in OpcodesIter::new(code).with_pc() {
            let mut data = 0;
            match opcode {
                op::JUMPDEST => jumpdests.set(pc, true),
                op::PC => data = pc as u32,
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

            instrs.push(InstrData { opcode, flags, static_gas, data });
        }

        let mut bytecode = Self { code, instrs, jumpdests, spec_id };

        // Pad code to ensure there is at least one diverging instruction.
        if bytecode.instrs.last().map_or(true, |last| !last.is_diverging(bytecode.is_eof())) {
            bytecode.instrs.push(InstrData::new(op::STOP));
        }

        bytecode
    }

    /// Returns the instruction at the given instruction counter.
    #[inline]
    pub(crate) fn instr(&self, instr: Instr) -> InstrData {
        self.instrs[instr]
    }

    /// Returns a mutable reference the instruction at the given instruction counter.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn instr_mut(&mut self, instr: Instr) -> &mut InstrData {
        &mut self.instrs[instr]
    }

    /// Returns an iterator over the instructions.
    #[inline]
    pub(crate) fn iter_instrs(&self) -> impl Iterator<Item = (usize, InstrData)> + '_ {
        self.instrs
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, data)| !data.flags.contains(InstrFlags::DEAD_CODE))
    }

    /// Runs a list of analysis passes on the instructions.
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn analyze(&mut self) -> Result<()> {
        trace_time!("static_jump_analysis", || self.static_jump_analysis());
        // NOTE: `mark_dead_code` must run after `static_jump_analysis` as it can mark unreachable
        // `JUMPDEST`s as dead code.
        trace_time!("mark_dead_code", || self.mark_dead_code());
        self.no_dynamic_jumps()
    }

    /// Mark `PUSH<N>` followed by `JUMP[I]` as `STATIC_JUMP` and resolve the target.
    #[instrument(name = "sj", level = "debug", skip_all)]
    fn static_jump_analysis(&mut self) {
        for instr in 0..self.instrs.len() {
            let jump_instr = instr + 1;
            let Some(next) = self.instrs.get(jump_instr) else { continue };
            let data = &self.instrs[instr];
            if !(matches!(data.opcode, op::PUSH1..=op::PUSH32)
                && matches!(next.opcode, op::JUMP | op::JUMPI))
            {
                continue;
            }

            let imm_data = data.data;
            let imm = self.get_imm(imm_data);
            self.instrs[jump_instr].flags |= InstrFlags::STATIC_JUMP;

            const USIZE_SIZE: usize = std::mem::size_of::<usize>();
            if imm.len() > USIZE_SIZE {
                debug!(jump_instr, "jump target too large");
                self.instrs[jump_instr].flags |= InstrFlags::INVALID_JUMP;
                continue;
            }

            let mut padded = [0; USIZE_SIZE];
            padded[USIZE_SIZE - imm.len()..].copy_from_slice(imm);
            let target_pc = usize::from_be_bytes(padded);
            if !self.is_valid_jump(target_pc) {
                debug!(jump_instr, target_pc, "invalid jump target");
                self.instrs[jump_instr].flags |= InstrFlags::INVALID_JUMP;
                continue;
            }

            self.instrs[instr].flags |= InstrFlags::SKIP_LOGIC;
            let target = self.pc_to_instr(target_pc);

            // Mark the `JUMPDEST` as reachable.
            if !self.is_eof() {
                debug_assert_eq!(
                    self.instrs[target],
                    op::JUMPDEST,
                    "is_valid_jump returned true for non-JUMPDEST: \
                     jump_instr={jump_instr} target_pc={target_pc} target={target}",
                );
                self.instrs[target].data = 1;
            }

            // Set the target on the `JUMP` instruction.
            debug!(jump_instr, target, "resolved jump target");
            self.instrs[jump_instr].data = target as u32;
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
        let mut iter = self.instrs.iter_mut().enumerate();
        while let Some((i, data)) = iter.next() {
            if data.is_diverging(is_eof) {
                let mut end = i;
                for (j, data) in &mut iter {
                    end = j;
                    if data.is_reachable_jumpdest() {
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

    /// Ensure that there are no dynamic jumps.
    fn no_dynamic_jumps(&mut self) -> Result<()> {
        let mut instrs = Vec::new();
        for (instr, data) in self.instrs.iter().enumerate() {
            if matches!(data.opcode, op::JUMP | op::JUMPI)
                && !data.flags.contains(InstrFlags::STATIC_JUMP)
            {
                instrs.push(instr);
            }
        }
        if !instrs.is_empty() {
            bail!("dynamic jumps are not yet implemented; instrs={instrs:?}");
        }
        Ok(())
    }

    /// Returns the raw opcode.
    pub(crate) fn raw_opcode(&self, instr: Instr) -> Opcode<'a> {
        self.instrs[instr].to_op_in(self)
    }

    pub(crate) fn get_imm_of(&self, instr_data: InstrData) -> Option<&'a [u8]> {
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

    /// Returns `true` if the bytecode is EOF.
    pub(crate) fn is_eof(&self) -> bool {
        false
    }

    /// Returns `true` if the instruction is diverging.
    pub(crate) fn is_instr_diverging(&self, instr: Instr) -> bool {
        self.instrs[instr].is_diverging(self.is_eof())
    }

    // TODO: is it worth it to make this a map?
    /// Converts a program counter (`self.code[ic]`) to an instruction (`self.instr(pc)`).
    fn pc_to_instr(&self, pc: usize) -> usize {
        debug_assert!(pc < self.code.len(), "pc out of bounds: {pc}");
        let (instr, (_, op)) = OpcodesIter::new(self.code)
            .with_pc() // pc
            .enumerate() // instr
            // Don't go until the end because we know `pc` are yielded in order.
            .take_while(|(_ic, (pc2, _))| *pc2 <= pc)
            .find(|(_ic, (pc2, _))| *pc2 == pc)
            .unwrap_or_else(|| panic!("pc to instr conversion failed; pc={pc}"));
        debug_assert_eq!(self.instrs[instr].to_op_in(self), op, "pc={pc} instr={instr}");
        instr
    }
}

/// A single instruction in the bytecode.
#[derive(Clone, Copy)]
pub(crate) struct InstrData {
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
    /// - `PC`: the program counter, meaning `self.code[pc]` is the opcode;
    /// - otherwise: no meaning.
    pub(crate) data: u32,
}

impl PartialEq<u8> for InstrData {
    #[inline]
    fn eq(&self, other: &u8) -> bool {
        self.opcode == *other
    }
}

impl PartialEq<InstrData> for u8 {
    #[inline]
    fn eq(&self, other: &InstrData) -> bool {
        *self == other.opcode
    }
}

impl fmt::Debug for InstrData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpcodeData")
            .field("opcode", &self.to_op())
            .field("flags", &self.flags)
            .field("data", &self.data)
            .finish()
    }
}

impl InstrData {
    /// Creates a new instruction data with the given opcode byte.
    /// Note that this may not be a valid instruction.
    #[inline]
    const fn new(opcode: u8) -> Self {
        Self { opcode, flags: InstrFlags::empty(), static_gas: 0, data: 0 }
    }

    /// Returns the length of the immediate data of this instruction.
    #[inline]
    pub(crate) const fn imm_len(self) -> usize {
        imm_len(self.opcode)
    }

    /// Converts this instruction to a raw opcode. Note that the immediate data is not resolved.
    #[inline]
    pub(crate) const fn to_op(self) -> Opcode<'static> {
        Opcode { opcode: self.opcode, immediate: None }
    }

    /// Converts this instruction to a raw opcode in the given bytecode.
    #[inline]
    pub(crate) fn to_op_in<'a>(self, bytecode: &Bytecode<'a>) -> Opcode<'a> {
        Opcode { opcode: self.opcode, immediate: bytecode.get_imm_of(self) }
    }

    /// Returns the static gas for this instruction, if any.
    #[inline]
    pub(crate) fn static_gas(&self) -> Option<u16> {
        (self.static_gas != u16::MAX).then_some(self.static_gas)
    }

    /// Returns `true` if we know that this instruction will stop execution.
    #[inline]
    pub(crate) const fn is_diverging(self, is_eof: bool) -> bool {
        // TODO: SELFDESTRUCT will not be diverging in the future.
        self.flags.contains(InstrFlags::INVALID_JUMP)
            || self.flags.contains(InstrFlags::DISABLED)
            || self.flags.contains(InstrFlags::UNKNOWN)
            || matches!(self.opcode, op::STOP | op::RETURN | op::REVERT | op::INVALID)
            || (self.opcode == op::SELFDESTRUCT && !is_eof)
    }

    /// Returns `true` if this instruction is a reachable `JUMPDEST`.
    #[inline]
    pub(crate) const fn is_reachable_jumpdest(self) -> bool {
        self.opcode == op::JUMPDEST && self.data == 1
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

//! Internal EVM bytecode and opcode representation.

use bitvec::vec::BitVec;
use color_eyre::{eyre::bail, Result};
use revm_interpreter::opcode as op;
use revm_primitives::SpecId;
use std::fmt;

mod info;
pub use info::*;

mod raw;
pub use raw::*;

/// EVM bytecode.
#[derive(Debug)]
pub(crate) struct Bytecode<'a> {
    /// The original bytecode slice.
    pub(crate) code: &'a [u8],
    /// The parsed opcodes.
    opcodes: Vec<OpcodeData>,
    /// `JUMPDEST` map.
    pub(crate) jumpdests: BitVec,
    /// The [`SpecId`].
    pub(crate) spec: SpecId,
}

impl<'a> Bytecode<'a> {
    pub(crate) fn new(code: &'a [u8], spec_id: SpecId) -> Self {
        let mut opcodes = Vec::with_capacity(code.len());
        let mut jumpdests = BitVec::repeat(false, code.len());
        let op_infos = op_info_map(spec_id);
        for (pc, RawOpcode { opcode, immediate }) in RawBytecodeIter::new(code).with_pc() {
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

            let mut flags = OpcodeFlags::empty();
            let info = op_infos[opcode as usize];
            if info.is_unknown() {
                flags |= OpcodeFlags::UNKNOWN;
            }
            if info.is_disabled() {
                flags |= OpcodeFlags::DISABLED;
            }
            let static_gas = info.static_gas().unwrap_or(u16::MAX);

            opcodes.push(OpcodeData { opcode, flags, static_gas, data });
        }

        // Pad code to ensure there is at least one diverging opcode.
        if opcodes.last().map_or(true, |last| !last.is_diverging()) {
            opcodes.push(OpcodeData::new(op::STOP));
        }

        Self { code, opcodes, jumpdests, spec: spec_id }
    }

    /// Returns the opcode at the given instruction counter.
    #[inline]
    pub(crate) fn opcode(&self, ic: usize) -> OpcodeData {
        self.opcodes[ic]
    }

    /// Returns a mutable reference the opcode at the given instruction counter.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn opcode_mut(&mut self, ic: usize) -> &mut OpcodeData {
        &mut self.opcodes[ic]
    }

    /// Returns an iterator over the opcodes with their instruction counters.
    #[inline]
    pub(crate) fn iter_opcodes(&self) -> impl Iterator<Item = (usize, OpcodeData)> + '_ {
        self.opcodes
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, data)| !data.flags.contains(OpcodeFlags::DEAD_CODE))
    }

    /// Runs a list of analysis passes on the opcodes.
    pub(crate) fn analyze(&mut self) -> Result<()> {
        self.static_jump_analysis();
        self.mark_dead_code();
        self.no_dynamic_jumps()
    }

    /// Mark `PUSH<N>` followed by `JUMP[I]` as `STATIC_JUMP` and resolve the target.
    fn static_jump_analysis(&mut self) {
        for i in 0..self.opcodes.len() {
            let Some(next) = self.opcodes.get(i + 1) else { continue };
            let opcode = &self.opcodes[i];
            if !(matches!(opcode.opcode, op::PUSH1..=op::PUSH32)
                && matches!(next.opcode, op::JUMP | op::JUMPI))
            {
                continue;
            }

            let imm_data = opcode.data;
            let imm = self.get_imm(imm_data);
            self.opcodes[i + 1].flags |= OpcodeFlags::STATIC_JUMP;

            const USIZE_SIZE: usize = std::mem::size_of::<usize>();
            if imm.len() > USIZE_SIZE {
                self.opcodes[i + 1].flags |= OpcodeFlags::INVALID_JUMP;
                continue;
            }

            let mut padded = [0; USIZE_SIZE];
            padded[USIZE_SIZE - imm.len()..].copy_from_slice(imm);
            let target = usize::from_be_bytes(padded);
            if !self.is_valid_jump(target) {
                self.opcodes[i + 1].flags |= OpcodeFlags::INVALID_JUMP;
                continue;
            }

            self.opcodes[i].flags |= OpcodeFlags::SKIP_LOGIC;
            let ic = self.pc_to_ic(target);
            debug_assert_eq!(
                self.opcodes[ic],
                op::JUMPDEST,
                "is_valid_jump returned true for non-JUMPDEST: target={target} ic={ic}",
            );
            self.opcodes[i + 1].data = ic as u32;
        }
    }

    /// Ensure that there are no dynamic jumps.
    fn no_dynamic_jumps(&mut self) -> Result<()> {
        for (ic, data) in self.opcodes.iter().enumerate() {
            if matches!(data.opcode, op::JUMP | op::JUMPI)
                && !data.flags.contains(OpcodeFlags::STATIC_JUMP)
            {
                bail!(
                    "dynamic jumps are not yet implemented; ic={ic} opcode={}",
                    data.to_raw_in(self)
                );
            }
        }
        Ok(())
    }

    /// Mark unreachable opcodes as `DEAD_CODE` to not generate any code for them.
    ///
    /// This pass is technically unnecessary as the backend will very likely optimize any
    /// unreachable code that we generate, but we might as well not generate it in the first place
    /// since it's trivial for us to do so.
    ///
    /// Before EOF, we can simply mark all opcodes that are between diverging opcodes and
    /// `JUMPDEST`s.
    ///
    /// After EOF, TODO.
    fn mark_dead_code(&mut self) {
        let mut iter = self.opcodes.iter_mut().enumerate();
        while let Some((i, data)) = iter.next() {
            if data.is_diverging() {
                let mut end = i;
                for (j, data) in &mut iter {
                    end = j;
                    if data.opcode == op::JUMPDEST {
                        break;
                    }
                    data.flags |= OpcodeFlags::DEAD_CODE;
                }
                debug!("found dead code: {start}..{end}", start = i + 1);
            }
        }
    }

    pub(crate) fn get_imm_of(&self, opcode: OpcodeData) -> Option<&'a [u8]> {
        (opcode.imm_len() > 0).then(|| self.get_imm(opcode.data))
    }

    fn get_imm(&self, data: u32) -> &'a [u8] {
        let (offset, len) = Immediate::unpack(data);
        &self.code[offset..offset + len]
    }

    fn is_valid_jump(&self, pc: usize) -> bool {
        self.jumpdests.get(pc).as_deref().copied() == Some(true)
    }

    // TODO: is it worth it to make this a map?
    fn pc_to_ic(&self, pc: usize) -> usize {
        let (ic, (_, op)) = RawBytecodeIter::new(self.code)
            .with_pc()
            .enumerate()
            // Don't go until the end because we know it's sorted.
            .take_while(|(_ic, (pc2, _))| *pc2 <= pc)
            .find(|(_ic, (pc2, _))| *pc2 == pc)
            .unwrap();
        debug_assert_eq!(self.opcodes[ic].to_raw_in(self), op, "pc {pc}, ic {ic}");
        ic
    }
}

/// A single instruction in the bytecode.
#[derive(Clone, Copy)]
pub(crate) struct OpcodeData {
    /// The opcode byte.
    pub(crate) opcode: u8,
    /// Opcode flags.
    pub(crate) flags: OpcodeFlags,
    /// Opcode static gas. Stored as `u16::MAX` if the gas is dynamic.
    static_gas: u16,
    /// Opcode-specific data:
    /// - if the opcode has immediate data, this is a packed offset+length into the bytecode;
    /// - `STATIC_JUMP in kind`: the jump target, already converted to an index into `opcodes`;
    /// - `opcode == op::PC`: the program counter, meaning `self.code[pc]` is the opcode;
    /// - otherwise: no meaning.
    pub(crate) data: u32,
}

impl PartialEq<u8> for OpcodeData {
    #[inline]
    fn eq(&self, other: &u8) -> bool {
        self.opcode == *other
    }
}

impl PartialEq<OpcodeData> for u8 {
    #[inline]
    fn eq(&self, other: &OpcodeData) -> bool {
        *self == other.opcode
    }
}

impl fmt::Debug for OpcodeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpcodeData")
            .field("opcode", &RawOpcode { opcode: self.opcode, immediate: None })
            .field("flags", &self.flags)
            .field("data", &self.data)
            .finish()
    }
}

impl OpcodeData {
    /// Creates a new opcode data with the given byte. Note that this may not be a valid opcode.
    #[inline]
    const fn new(opcode: u8) -> Self {
        Self { opcode, flags: OpcodeFlags::empty(), static_gas: 0, data: 0 }
    }

    /// Returns the length of the immediate data for this opcode.
    #[inline]
    pub(crate) const fn imm_len(self) -> usize {
        imm_len(self.opcode)
    }

    /// Converts this opcode to a raw opcode. Note that the immediate data is not resolved.
    #[inline]
    pub(crate) const fn to_raw(self) -> RawOpcode<'static> {
        RawOpcode { opcode: self.opcode, immediate: None }
    }

    /// Converts this opcode to a raw opcode in the given bytecode.
    #[inline]
    pub(crate) fn to_raw_in<'a>(self, bytecode: &Bytecode<'a>) -> RawOpcode<'a> {
        RawOpcode { opcode: self.opcode, immediate: bytecode.get_imm_of(self) }
    }

    /// Returns `true` if we know that this opcode will stop execution.
    #[inline]
    pub(crate) const fn is_diverging(self) -> bool {
        // TODO: SELFDESTRUCT will not be diverging in the future.
        self.flags.contains(OpcodeFlags::INVALID_JUMP)
            || self.flags.contains(OpcodeFlags::DISABLED)
            || self.flags.contains(OpcodeFlags::UNKNOWN)
            || matches!(
                self.opcode,
                op::STOP | op::RETURN | op::REVERT | op::INVALID | op::SELFDESTRUCT
            )
    }

    /// Returns the static gas for this opcode, if any.
    #[inline]
    pub(crate) fn static_gas(&self) -> Option<u16> {
        (self.static_gas != u16::MAX).then_some(self.static_gas)
    }
}

bitflags::bitflags! {
    /// [`OpcodeData`] flags.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct OpcodeFlags: u8 {
        /// The `JUMP`/`JUMPI` target is known at compile time.
        /// This is implied for other jump instructions which are always static.
        const STATIC_JUMP = 1 << 0;
        /// The jump target is known to be invalid.
        /// Always returns [`InstructionResult::InvalidJump`] at runtime.
        const INVALID_JUMP = 1 << 1;

        /// The opcode is disabled in this EVM version.
        /// Always returns [`InstructionResult::NotActivated`] at runtime.
        const DISABLED = 1 << 2;
        /// The opcode is unknown.
        /// Always returns [`InstructionResult::NotFound`] at runtime.
        const UNKNOWN = 1 << 3;

        /// Skip generating opcode logic, but keep the gas calculation.
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

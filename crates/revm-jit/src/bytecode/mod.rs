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
    pub(crate) spec_id: SpecId,
}

impl<'a> Bytecode<'a> {
    pub(crate) fn new(code: &'a [u8], spec_id: SpecId) -> Self {
        let mut opcodes = Vec::with_capacity(code.len() + 1);
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

        let mut bytecode = Self { code, opcodes, jumpdests, spec_id };

        // Pad code to ensure there is at least one diverging opcode.
        let is_eof = bytecode.is_eof();
        if bytecode.opcodes.last().map_or(true, |last| !last.is_diverging(is_eof)) {
            bytecode.opcodes.push(OpcodeData::new(op::STOP));
        }

        bytecode
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
        for ic in 0..self.opcodes.len() {
            let jic = ic + 1;
            let Some(next) = self.opcodes.get(jic) else { continue };
            let opcode = &self.opcodes[ic];
            if !(matches!(opcode.opcode, op::PUSH1..=op::PUSH32)
                && matches!(next.opcode, op::JUMP | op::JUMPI))
            {
                continue;
            }

            let imm_data = opcode.data;
            let imm = self.get_imm(imm_data);
            self.opcodes[jic].flags |= OpcodeFlags::STATIC_JUMP;

            const USIZE_SIZE: usize = std::mem::size_of::<usize>();
            if imm.len() > USIZE_SIZE {
                debug!(jic, "jump target too large");
                self.opcodes[jic].flags |= OpcodeFlags::INVALID_JUMP;
                continue;
            }

            let mut padded = [0; USIZE_SIZE];
            padded[USIZE_SIZE - imm.len()..].copy_from_slice(imm);
            let target = usize::from_be_bytes(padded);
            if !self.is_valid_jump(target) {
                debug!(jic, target, "invalid jump target");
                self.opcodes[jic].flags |= OpcodeFlags::INVALID_JUMP;
                continue;
            }

            self.opcodes[ic].flags |= OpcodeFlags::SKIP_LOGIC;
            let target_ic = self.pc_to_ic(target);

            // Mark the `JUMPDEST` as reachable.
            if !self.is_eof() {
                debug_assert_eq!(
                    self.opcodes[target_ic],
                    op::JUMPDEST,
                    "is_valid_jump returned true for non-JUMPDEST: \
                     jic={jic} target={target} target_ic={target_ic}",
                );
                self.opcodes[target_ic].data = 1;
            }

            // Set the target on the `JUMP` opcode.
            debug!(jic, target_ic, "resolved jump target");
            self.opcodes[jic].data = target_ic as u32;
        }
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
    #[instrument(name = "dce", level = "debug", skip_all)]
    fn mark_dead_code(&mut self) {
        let is_eof = self.is_eof();
        let mut iter = self.opcodes.iter_mut().enumerate();
        while let Some((i, data)) = iter.next() {
            if data.is_diverging(is_eof) {
                let mut end = i;
                for (j, data) in &mut iter {
                    end = j;
                    if data.is_reachable_jumpdest() {
                        break;
                    }
                    data.flags |= OpcodeFlags::DEAD_CODE;
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
        let mut ics = Vec::new();
        for (ic, data) in self.opcodes.iter().enumerate() {
            if matches!(data.opcode, op::JUMP | op::JUMPI)
                && !data.flags.contains(OpcodeFlags::STATIC_JUMP)
            {
                ics.push(ic);
            }
        }
        if !ics.is_empty() {
            bail!("dynamic jumps are not yet implemented; ics={ics:?}");
        }
        Ok(())
    }

    /// Returns the raw opcode.
    pub(crate) fn raw_opcode(&self, ic: usize) -> RawOpcode<'a> {
        self.opcodes[ic].to_raw_in(self)
    }

    pub(crate) fn get_imm_of(&self, opcode: OpcodeData) -> Option<&'a [u8]> {
        (opcode.imm_len() > 0).then(|| self.get_imm(opcode.data))
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

    /// Returns `true` if the opcode at the given instruction counter is diverging.
    pub(crate) fn is_opcode_diverging(&self, ic: usize) -> bool {
        self.opcodes[ic].is_diverging(self.is_eof())
    }

    // TODO: is it worth it to make this a map?
    /// Converts a program counter (`self.code[ic]`) to an instruction counter (`self.opcode(pc)`).
    fn pc_to_ic(&self, pc: usize) -> usize {
        debug_assert!(pc < self.code.len(), "pc out of bounds: {pc}");
        let (ic, (_, op)) = RawBytecodeIter::new(self.code)
            .with_pc() // pc
            .enumerate() // ic
            // Don't go until the end because we know it's sorted.
            .take_while(|(_ic, (pc2, _))| *pc2 <= pc)
            .find(|(_ic, (pc2, _))| *pc2 == pc)
            .unwrap_or_else(|| panic!("pc to ic conversion failed; pc={pc}"));
        debug_assert_eq!(self.opcodes[ic].to_raw_in(self), op, "pc={pc} ic={ic}");
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
    /// - `JUMP{,I} && STATIC_JUMP in kind`: the jump target, already converted to an index into
    ///   `opcodes`;
    /// - `JUMPDEST`: `1` if the jump destination is reachable, `0` otherwise;
    /// - `PC`: the program counter, meaning `self.code[pc]` is the opcode;
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

    /// Returns the static gas for this opcode, if any.
    #[inline]
    pub(crate) fn static_gas(&self) -> Option<u16> {
        (self.static_gas != u16::MAX).then_some(self.static_gas)
    }

    /// Returns `true` if we know that this opcode will stop execution.
    #[inline]
    pub(crate) const fn is_diverging(self, is_eof: bool) -> bool {
        // TODO: SELFDESTRUCT will not be diverging in the future.
        self.flags.contains(OpcodeFlags::INVALID_JUMP)
            || self.flags.contains(OpcodeFlags::DISABLED)
            || self.flags.contains(OpcodeFlags::UNKNOWN)
            || matches!(
                self.opcode,
                op::STOP | op::RETURN | op::REVERT | op::INVALID | op::SELFDESTRUCT
            )
            || (self.opcode == op::SELFDESTRUCT && !is_eof)
    }

    /// Returns `true` if this opcode is a reachable `JUMPDEST`.
    #[inline]
    pub(crate) const fn is_reachable_jumpdest(self) -> bool {
        self.opcode == op::JUMPDEST && self.data == 1
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

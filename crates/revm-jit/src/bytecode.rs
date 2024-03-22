//! Internal EVM bytecode representation and helpers.

use bitvec::vec::BitVec;
use color_eyre::{eyre::bail, Result};
use revm_interpreter::{opcode as op, OPCODE_JUMPMAP};
use revm_primitives::SpecId;
use std::{fmt, slice};

/// EVM bytecode.
#[derive(Debug)]
pub(crate) struct Bytecode<'a> {
    /// The original bytecode slice.
    pub(crate) code: &'a [u8],
    /// The parsed opcodes.
    pub(crate) opcodes: Vec<OpcodeData>,
    /// `JUMPDEST` map.
    pub(crate) jumpdests: BitVec,
    /// The [`SpecId`].
    #[allow(dead_code)]
    pub(crate) spec: SpecId,
}

impl<'a> Bytecode<'a> {
    pub(crate) fn new(code: &'a [u8], spec: SpecId) -> Self {
        let mut opcodes = Vec::with_capacity(code.len());
        let mut jumpdests = BitVec::repeat(false, code.len());
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

            let mut flags = OpcodeFlags::NORMAL;
            if !op_enabled_in(opcode, spec) {
                flags |= OpcodeFlags::DISABLED;
            }

            opcodes.push(OpcodeData { opcode, flags, data });
        }

        // Pad code to ensure there is at least one diverging opcode.
        if opcodes.last().map_or(true, |last| last.opcode != op::STOP) {
            opcodes.push(OpcodeData { opcode: op::STOP, flags: OpcodeFlags::NORMAL, data: 0 });
        }

        Self { code, opcodes, jumpdests, spec }
    }

    pub(crate) fn analyze(&mut self) -> Result<()> {
        for i in 0..self.opcodes.len() {
            let Some(next) = self.opcodes.get(i + 1) else { continue };
            let opcode = &self.opcodes[i];
            if matches!(opcode.opcode, op::PUSH1..=op::PUSH32)
                && matches!(next.opcode, op::JUMP | op::JUMPI)
            {
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
                if self.is_valid_jump(target) {
                    self.opcodes[i].flags |= OpcodeFlags::SKIP_LOGIC;
                    let ic = self.pc_to_ic(target);
                    debug_assert_eq!(
                        self.opcodes[ic],
                        op::JUMPDEST,
                        "is_valid_jump returned true for non-JUMPDEST: target={target} ic={ic}",
                    );
                    self.opcodes[i + 1].data = ic as u32;
                } else {
                    self.opcodes[i + 1].flags |= OpcodeFlags::INVALID_JUMP;
                }
                continue;
            }

            if matches!(opcode.opcode, op::JUMP | op::JUMPI)
                && !opcode.flags.contains(OpcodeFlags::STATIC_JUMP)
            {
                bail!("dynamic jumps are not yet implemented");
            }
        }
        Ok(())
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
}

bitflags::bitflags! {
    /// [`OpcodeData`] flags.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct OpcodeFlags: u8 {
        /// No modifiers.
        const NORMAL = 0;
        /// The `JUMP`/`JUMPI` target is known at compile time.
        /// This is implied for other jump instructions which are always static.
        const STATIC_JUMP = 1 << 0;
        /// The jump target is known to be invalid.
        /// Always returns `InvalidJump` at runtime.
        const INVALID_JUMP = 1 << 1;

        /// Skip gas accounting code.
        const SKIP_GAS = 1 << 3;
        /// Skip generating code.
        const SKIP_LOGIC = 1 << 4;

        /// The opcode is not enabled in the current EVM version.
        /// Always returns `NotActivated`.
        const DISABLED = 1 << 5;
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

/// A bytecode iterator that yields opcodes and their immediate data, alongside the program counter.
///
/// Created by calling [`RawBytecodeIter::with_pc`].
#[derive(Debug)]
pub struct RawBytecodeIterWithPc<'a> {
    iter: RawBytecodeIter<'a>,
    pc: usize,
}

impl<'a> Iterator for RawBytecodeIterWithPc<'a> {
    type Item = (usize, RawOpcode<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|opcode| {
            let pc = self.pc;
            self.pc += 1;
            if let Some(imm) = opcode.immediate {
                self.pc += imm.len();
            }
            (pc, opcode)
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl std::iter::FusedIterator for RawBytecodeIterWithPc<'_> {}

/// A bytecode iterator that yields opcodes and their immediate data.
///
/// If the bytecode is not well-formed, the iterator will still yield opcodes, but the immediate
/// data may be incorrect. For example, if the bytecode is `PUSH2 0x69`, the iterator will yield
/// `PUSH2, None`.
#[derive(Clone, Debug)]
pub struct RawBytecodeIter<'a> {
    iter: slice::Iter<'a, u8>,
}

impl fmt::Display for RawBytecodeIter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, op) in self.clone().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{op}")?;
        }
        Ok(())
    }
}

impl<'a> RawBytecodeIter<'a> {
    /// Create a new iterator over the given bytecode slice.
    #[inline]
    pub fn new(slice: &'a [u8]) -> Self {
        Self { iter: slice.iter() }
    }

    /// Returns a new iterator that also yields the program counter alongside the opcode and
    /// immediate data.
    #[inline]
    pub fn with_pc(self) -> RawBytecodeIterWithPc<'a> {
        RawBytecodeIterWithPc { iter: self, pc: 0 }
    }

    /// Returns the inner iterator.
    #[inline]
    pub fn inner(&self) -> &slice::Iter<'a, u8> {
        &self.iter
    }

    /// Returns the inner iterator.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut slice::Iter<'a, u8> {
        &mut self.iter
    }

    /// Returns the inner iterator.
    #[inline]
    pub fn into_inner(self) -> slice::Iter<'a, u8> {
        self.iter
    }
}

impl<'a> Iterator for RawBytecodeIter<'a> {
    type Item = RawOpcode<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().copied().map(|opcode| {
            let len = imm_len(opcode);
            let immediate = if len > 0 {
                let r = self.iter.as_slice().get(..len);
                // TODO: Use `advance_by` when stable.
                self.iter.by_ref().take(len).for_each(drop);
                r
            } else {
                None
            };
            RawOpcode { opcode, immediate }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.len();
        ((len != 0) as usize, Some(len))
    }
}

impl std::iter::FusedIterator for RawBytecodeIter<'_> {}

/// An opcode and its immediate data. Returned by [`RawBytecodeIter`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RawOpcode<'a> {
    /// The opcode.
    pub opcode: u8,
    /// The immediate data, if any.
    pub immediate: Option<&'a [u8]>,
}

impl fmt::Debug for RawOpcode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for RawOpcode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match OPCODE_JUMPMAP[self.opcode as usize] {
            Some(s) => f.write_str(s),
            None => write!(f, "UNKNOWN({:02x})", self.opcode),
        }?;
        match self.immediate {
            Some(imm) => write!(f, " {}", revm_primitives::hex::encode_prefixed(imm)),
            None => Ok(()),
        }
    }
}

/// Returns the length of the immediate data for the given opcode, or `0` if none.
#[inline]
pub const fn imm_len(op: u8) -> usize {
    match op {
        op::PUSH1..=op::PUSH32 => (op - op::PUSH0) as usize,
        // op::DATALOADN => 1,
        //
        // op::RJUMP => 2,
        // op::RJUMPI => 2,
        // op::RJUMPV => 2,
        // op::CALLF => 2,
        // op::JUMPF => 2,
        // op::DUPN => 1,
        // op::SWAPN => 1,
        // op::EXCHANGE => 1,
        _ => 0,
    }
}

/// Returns a string representation of the given bytecode.
pub fn format_bytecode(bytecode: &[u8]) -> String {
    RawBytecodeIter::new(bytecode).to_string()
}

/// Returns `true` if the opcode is enabled in the given [`SpecId`].
pub fn op_enabled_in(opcode: u8, spec: SpecId) -> bool {
    const FT: SpecId = SpecId::FRONTIER;
    // TODO
    #[rustfmt::skip]
    static SPEC_ID_MAP: [SpecId; 256] = [
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
        FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT, FT,
    ];
    spec == SpecId::LATEST || SpecId::enabled(spec, SPEC_ID_MAP[opcode as usize])
}
#[cfg(test)]
mod tests {
    use super::*;
    use revm_interpreter::opcode as op;

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
    fn iter_basic() {
        let bytecode = [0x01, 0x02, 0x03, 0x04, 0x05];
        let mut iter = RawBytecodeIter::new(&bytecode);

        assert_eq!(iter.next(), Some(RawOpcode { opcode: 0x01, immediate: None }));
        assert_eq!(iter.next(), Some(RawOpcode { opcode: 0x02, immediate: None }));
        assert_eq!(iter.next(), Some(RawOpcode { opcode: 0x03, immediate: None }));
        assert_eq!(iter.next(), Some(RawOpcode { opcode: 0x04, immediate: None }));
        assert_eq!(iter.next(), Some(RawOpcode { opcode: 0x05, immediate: None }));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_with_imm() {
        let bytecode = [op::PUSH0, op::PUSH1, 0x69, op::PUSH2, 0x01, 0x02];
        let mut iter = RawBytecodeIter::new(&bytecode);

        assert_eq!(iter.next(), Some(RawOpcode { opcode: op::PUSH0, immediate: None }));
        assert_eq!(iter.next(), Some(RawOpcode { opcode: op::PUSH1, immediate: Some(&[0x69]) }));
        assert_eq!(
            iter.next(),
            Some(RawOpcode { opcode: op::PUSH2, immediate: Some(&[0x01, 0x02]) })
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_with_imm_too_short() {
        let bytecode = [op::PUSH2, 0x69];
        let mut iter = RawBytecodeIter::new(&bytecode);

        assert_eq!(iter.next(), Some(RawOpcode { opcode: op::PUSH2, immediate: None }));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn display() {
        let bytecode = [op::PUSH0, op::PUSH1, 0x69, op::PUSH2, 0x01, 0x02];
        let s = format_bytecode(&bytecode);
        assert_eq!(s, "PUSH0 PUSH1 0x69 PUSH2 0x0102");
    }
}

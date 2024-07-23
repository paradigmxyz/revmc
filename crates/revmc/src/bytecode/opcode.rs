use crate::{op_info_map, OpcodeInfo};
use revm_interpreter::{opcode as op, OPCODE_INFO_JUMPTABLE};
use revm_primitives::{Bytes, Eof, SpecId, EOF_MAGIC_BYTES};
use std::{fmt, slice};

/// A bytecode iterator that yields opcodes and their immediate data, alongside the program counter.
///
/// Created by calling [`OpcodesIter::with_pc`].
#[derive(Debug)]
pub struct OpcodesIterWithPc<'a> {
    iter: OpcodesIter<'a>,
    pc: usize,
}

impl<'a> Iterator for OpcodesIterWithPc<'a> {
    type Item = (usize, Opcode<'a>);

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

impl std::iter::FusedIterator for OpcodesIterWithPc<'_> {}

/// An iterator that yields opcodes and their immediate data.
///
/// If the bytecode is not well-formed, the iterator will still yield opcodes, but the immediate
/// data may be incorrect. For example, if the bytecode is `PUSH2 0x69`, the iterator will yield
/// `PUSH2, None`.
#[derive(Clone, Debug)]
pub struct OpcodesIter<'a> {
    iter: slice::Iter<'a, u8>,
    info: &'static [OpcodeInfo; 256],
}

impl fmt::Display for OpcodesIter<'_> {
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

impl<'a> OpcodesIter<'a> {
    /// Create a new iterator over the given bytecode slice.
    #[inline]
    pub fn new(slice: &'a [u8], spec_id: SpecId) -> Self {
        Self { iter: slice.iter(), info: op_info_map(spec_id) }
    }

    /// Returns a new iterator that also yields the program counter alongside the opcode and
    /// immediate data.
    #[inline]
    pub fn with_pc(self) -> OpcodesIterWithPc<'a> {
        OpcodesIterWithPc { iter: self, pc: 0 }
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

impl<'a> Iterator for OpcodesIter<'a> {
    type Item = Opcode<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|&opcode| {
            let info = self.info[opcode as usize];
            if info.is_unknown() || info.is_disabled() {
                return Opcode { opcode, immediate: None };
            }

            let mut len = min_imm_len(opcode) as usize;
            if opcode == op::RJUMPV {
                if let Some(&max_case) = self.iter.as_slice().first() {
                    len += (max_case as usize + 1) * 2;
                }
            }
            let immediate = if len > 0 {
                let r = self.iter.as_slice().get(..len);
                // TODO: Use `advance_by` when stable.
                self.iter.by_ref().take(len).for_each(drop);
                r
            } else {
                None
            };
            Opcode { opcode, immediate }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.len();
        ((len != 0) as usize, Some(len))
    }
}

impl std::iter::FusedIterator for OpcodesIter<'_> {}

/// An opcode and its immediate data. Returned by [`OpcodesIter`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Opcode<'a> {
    /// The opcode.
    pub opcode: u8,
    /// The immediate data, if any.
    pub immediate: Option<&'a [u8]>,
}

impl fmt::Debug for Opcode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Opcode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match OPCODE_INFO_JUMPTABLE[self.opcode as usize] {
            Some(s) => f.write_str(s.name()),
            None => write!(f, "UNKNOWN(0x{:02x})", self.opcode),
        }?;
        match self.immediate {
            Some(imm) => write!(f, " {}", revm_primitives::hex::encode_prefixed(imm)),
            None => Ok(()),
        }
    }
}

/// Returns the length of the immediate data for the given opcode, or `0` if none.
///
/// This is the full length for all opcodes that have an immediate, except for `RJUMPV`, which
/// currently is the only opcode which has a variable length immediate.
#[inline]
pub const fn min_imm_len(op: u8) -> u8 {
    if let Some(info) = &OPCODE_INFO_JUMPTABLE[op as usize] {
        info.immediate_size()
    } else {
        0
    }
}

/// Returns the number of input and output stack elements of the given opcode.
#[inline]
pub const fn stack_io(op: u8) -> (u8, u8) {
    if let Some(info) = &OPCODE_INFO_JUMPTABLE[op as usize] {
        (info.inputs(), info.outputs())
    } else {
        (0, 0)
    }
}

/// Returns a string representation of the given bytecode.
pub fn format_bytecode(bytecode: &[u8], spec_id: SpecId) -> String {
    let mut w = String::new();
    format_bytecode_to(bytecode, spec_id, &mut w).unwrap();
    w
}

/// Formats an EVM bytecode to the given writer.
pub fn format_bytecode_to<W: fmt::Write + ?Sized>(
    bytecode: &[u8],
    spec_id: SpecId,
    w: &mut W,
) -> fmt::Result {
    if spec_id.is_enabled_in(SpecId::PRAGUE_EOF) && bytecode.starts_with(&EOF_MAGIC_BYTES) {
        write!(w, "{:#?}", Eof::decode(Bytes::copy_from_slice(bytecode)))
    } else {
        write!(w, "{}", OpcodesIter::new(bytecode, spec_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm_interpreter::opcode as op;

    const DEF_SPEC: SpecId = SpecId::ARROW_GLACIER;

    #[test]
    fn iter_basic() {
        let bytecode = [0x01, 0x02, 0x03, 0x04, 0x05];
        let mut iter = OpcodesIter::new(&bytecode, DEF_SPEC);

        assert_eq!(iter.next(), Some(Opcode { opcode: 0x01, immediate: None }));
        assert_eq!(iter.next(), Some(Opcode { opcode: 0x02, immediate: None }));
        assert_eq!(iter.next(), Some(Opcode { opcode: 0x03, immediate: None }));
        assert_eq!(iter.next(), Some(Opcode { opcode: 0x04, immediate: None }));
        assert_eq!(iter.next(), Some(Opcode { opcode: 0x05, immediate: None }));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_with_imm() {
        let bytecode = [op::PUSH0, op::PUSH1, 0x69, op::PUSH2, 0x01, 0x02];
        let mut iter = OpcodesIter::new(&bytecode, DEF_SPEC);

        assert_eq!(iter.next(), Some(Opcode { opcode: op::PUSH0, immediate: None }));
        assert_eq!(iter.next(), Some(Opcode { opcode: op::PUSH1, immediate: Some(&[0x69]) }));
        assert_eq!(iter.next(), Some(Opcode { opcode: op::PUSH2, immediate: Some(&[0x01, 0x02]) }));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_with_imm_too_short() {
        let bytecode = [op::PUSH2, 0x69];
        let mut iter = OpcodesIter::new(&bytecode, DEF_SPEC);

        assert_eq!(iter.next(), Some(Opcode { opcode: op::PUSH2, immediate: None }));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn display() {
        let bytecode = [op::PUSH0, op::PUSH1, 0x69, op::PUSH2, 0x01, 0x02];
        let s = format_bytecode(&bytecode, DEF_SPEC);
        assert_eq!(s, "PUSH0 PUSH1 0x69 PUSH2 0x0102");
    }
}

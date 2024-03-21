use revm_interpreter::{opcode as op, OPCODE_JUMPMAP};
use std::{fmt, slice};

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

#[cfg(test)]
mod tests {
    use super::*;
    use revm_interpreter::opcode as op;

    #[test]
    fn basic() {
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
    fn with_imm() {
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
    fn with_imm_too_short() {
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

use revm_interpreter::opcode as op;
use std::fmt;

/// The stack I/O of an opcode.
#[derive(Clone, Copy)]
pub struct StackIo(u16);

impl fmt::Debug for StackIo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("StackIo").field(&(self.input(), self.output())).finish()
    }
}

impl StackIo {
    const MAP: &'static [Self; 256] = &Self::make_map();

    /// Returns the stack I/O for the given opcode.
    #[inline]
    pub const fn new(op: u8) -> Self {
        Self::MAP[op as usize]
    }

    /// Returns the number of input stack elements.
    #[inline]
    pub const fn input(&self) -> u8 {
        self.0 as u8
    }

    /// Returns the number of output stack elements.
    #[inline]
    pub const fn output(&self) -> u8 {
        (self.0 >> 8) as u8
    }

    /// Returns the number of `(input, output)` stack elements.
    #[inline]
    pub const fn both(&self) -> (u8, u8) {
        (self.input(), self.output())
    }

    #[allow(clippy::identity_op)]
    const fn make_map() -> [Self; 256] {
        let mut map = [Self(0); 256];
        macro_rules! set {
            ($($op:ident = $i:expr, $o:expr;)*) => {
                $(
                    map[op::$op as usize] = Self(($o << 8) | $i);
                )*
            };
        }
        set! {
            // op = input, output;
            STOP = 0, 0;

            ADD        = 2, 1;
            MUL        = 2, 1;
            SUB        = 2, 1;
            DIV        = 2, 1;
            SDIV       = 2, 1;
            MOD        = 2, 1;
            SMOD       = 2, 1;
            ADDMOD     = 3, 1;
            MULMOD     = 3, 1;
            EXP        = 2, 1;
            SIGNEXTEND = 2, 1;
            // 0x0C
            // 0x0D
            // 0x0E
            // 0x0F
            LT     = 2, 1;
            GT     = 2, 1;
            SLT    = 2, 1;
            SGT    = 2, 1;
            EQ     = 2, 1;
            ISZERO = 1, 1;
            AND    = 2, 1;
            OR     = 2, 1;
            XOR    = 2, 1;
            NOT    = 1, 1;
            BYTE   = 2, 1;
            SHL    = 2, 1;
            SHR    = 2, 1;
            SAR    = 2, 1;
            // 0x1E
            // 0x1F
            KECCAK256 = 2, 1;
            // 0x21
            // 0x22
            // 0x23
            // 0x24
            // 0x25
            // 0x26
            // 0x27
            // 0x28
            // 0x29
            // 0x2A
            // 0x2B
            // 0x2C
            // 0x2D
            // 0x2E
            // 0x2F
            ADDRESS      = 0, 1;
            BALANCE      = 1, 1;
            ORIGIN       = 0, 1;
            CALLER       = 0, 1;
            CALLVALUE    = 0, 1;
            CALLDATALOAD = 1, 1;
            CALLDATASIZE = 0, 1;
            CALLDATACOPY = 3, 0;
            CODESIZE     = 0, 1;
            CODECOPY     = 3, 0;

            GASPRICE       = 0, 1;
            EXTCODESIZE    = 1, 1;
            EXTCODECOPY    = 4, 0;
            RETURNDATASIZE = 0, 1;
            RETURNDATACOPY = 3, 0;
            EXTCODEHASH    = 1, 1;
            BLOCKHASH      = 1, 1;
            COINBASE       = 0, 1;
            TIMESTAMP      = 0, 1;
            NUMBER         = 0, 1;
            DIFFICULTY     = 0, 1;
            GASLIMIT       = 0, 1;
            CHAINID        = 0, 1;
            SELFBALANCE    = 0, 1;
            BASEFEE        = 0, 1;
            BLOBHASH       = 1, 1;
            BLOBBASEFEE    = 0, 1;
            // 0x4B
            // 0x4C
            // 0x4D
            // 0x4E
            // 0x4F
            POP      = 1, 0;
            MLOAD    = 1, 1;
            MSTORE   = 2, 0;
            MSTORE8  = 2, 0;
            SLOAD    = 1, 1;
            SSTORE   = 2, 0;
            JUMP     = 1, 0;
            JUMPI    = 2, 0;
            PC       = 0, 1;
            MSIZE    = 0, 1;
            GAS      = 0, 1;
            JUMPDEST = 0, 0;
            TLOAD    = 1, 1;
            TSTORE   = 2, 0;
            MCOPY    = 3, 0;

            PUSH0  = 0, 1;
            PUSH1  = 0, 1;
            PUSH2  = 0, 1;
            PUSH3  = 0, 1;
            PUSH4  = 0, 1;
            PUSH5  = 0, 1;
            PUSH6  = 0, 1;
            PUSH7  = 0, 1;
            PUSH8  = 0, 1;
            PUSH9  = 0, 1;
            PUSH10 = 0, 1;
            PUSH11 = 0, 1;
            PUSH12 = 0, 1;
            PUSH13 = 0, 1;
            PUSH14 = 0, 1;
            PUSH15 = 0, 1;
            PUSH16 = 0, 1;
            PUSH17 = 0, 1;
            PUSH18 = 0, 1;
            PUSH19 = 0, 1;
            PUSH20 = 0, 1;
            PUSH21 = 0, 1;
            PUSH22 = 0, 1;
            PUSH23 = 0, 1;
            PUSH24 = 0, 1;
            PUSH25 = 0, 1;
            PUSH26 = 0, 1;
            PUSH27 = 0, 1;
            PUSH28 = 0, 1;
            PUSH29 = 0, 1;
            PUSH30 = 0, 1;
            PUSH31 = 0, 1;
            PUSH32 = 0, 1;

            DUP1  = 1, 2;
            DUP2  = 2, 3;
            DUP3  = 3, 4;
            DUP4  = 4, 5;
            DUP5  = 5, 6;
            DUP6  = 6, 7;
            DUP7  = 7, 8;
            DUP8  = 8, 9;
            DUP9  = 9, 10;
            DUP10 = 10, 11;
            DUP11 = 11, 12;
            DUP12 = 12, 13;
            DUP13 = 13, 14;
            DUP14 = 14, 15;
            DUP15 = 15, 16;
            DUP16 = 16, 17;

            SWAP1  = 2, 2;
            SWAP2  = 3, 3;
            SWAP3  = 4, 4;
            SWAP4  = 5, 5;
            SWAP5  = 6, 6;
            SWAP6  = 7, 7;
            SWAP7  = 8, 8;
            SWAP8  = 9, 9;
            SWAP9  = 10, 10;
            SWAP10 = 11, 11;
            SWAP11 = 12, 12;
            SWAP12 = 13, 13;
            SWAP13 = 14, 14;
            SWAP14 = 15, 15;
            SWAP15 = 16, 16;
            SWAP16 = 17, 17;

            LOG0 = 2 + 0, 0;
            LOG1 = 2 + 1, 0;
            LOG2 = 2 + 2, 0;
            LOG3 = 2 + 3, 0;
            LOG4 = 2 + 4, 0;
            // 0xA5
            // 0xA6
            // 0xA7
            // 0xA8
            // 0xA9
            // 0xAA
            // 0xAB
            // 0xAC
            // 0xAD
            // 0xAE
            // 0xAF
            // 0xB0
            // 0xB1
            // 0xB2
            // 0xB3
            // 0xB4
            // 0xB5
            // 0xB6
            // 0xB7
            // 0xB8
            // 0xB9
            // 0xBA
            // 0xBB
            // 0xBC
            // 0xBD
            // 0xBE
            // 0xBF
            // 0xC0
            // 0xC1
            // 0xC2
            // 0xC3
            // 0xC4
            // 0xC5
            // 0xC6
            // 0xC7
            // 0xC8
            // 0xC9
            // 0xCA
            // 0xCB
            // 0xCC
            // 0xCD
            // 0xCE
            // 0xCF
            // 0xD0
            // 0xD1
            // 0xD2
            // 0xD3
            // 0xD4
            // 0xD5
            // 0xD6
            // 0xD7
            // 0xD8
            // 0xD9
            // 0xDA
            // 0xDB
            // 0xDC
            // 0xDD
            // 0xDE
            // 0xDF
            // 0xE0
            // 0xE1
            // 0xE2
            // 0xE3
            // 0xE4
            // 0xE5
            // 0xE6
            // 0xE7
            // 0xE8
            // 0xE9
            // 0xEA
            // 0xEB
            // 0xEC
            // 0xED
            // 0xEE
            // 0xEF
            CREATE       = 3, 1;
            CALL         = 7, 1;
            CALLCODE     = 7, 1;
            RETURN       = 2, 0;
            DELEGATECALL = 6, 1;
            CREATE2      = 4, 1;
            // 0xF6
            // 0xF7
            // 0xF8
            // 0xF9
            STATICCALL   = 6, 1;
            // 0xFB
            // 0xFC
            REVERT       = 2, 0;
            INVALID      = 0, 0;
            SELFDESTRUCT = 1, 0;
        };
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_io() {
        assert_eq!(StackIo::new(op::PUSH0).both(), (0, 1));
        assert_eq!(StackIo::new(op::CALL).both(), (7, 1));
    }
}

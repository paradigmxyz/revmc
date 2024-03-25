use revm_interpreter::{gas, opcode as op};
use revm_primitives::{spec_to_generic, SpecId};

/// Opcode information.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpcodeInfo(u16);

impl OpcodeInfo {
    /// The unknown flag.
    pub const UNKNOWN: u16 = 0b1000_0000_0000_0000;
    /// The dynamic flag.
    pub const DYNAMIC: u16 = 0b0100_0000_0000_0000;
    /// The disabled flag.
    pub const DISABLED: u16 = 0b0010_0000_0000_0000;
    /// The mask for the gas cost.
    pub const MASK: u16 = 0b0001_1111_1111_1111;

    /// Creates a new gas info with the given gas cost.
    #[inline]
    pub const fn new(gas: u16) -> Self {
        Self(gas)
    }

    /// Returns `true` if the opcode is unknown.
    #[inline]
    pub const fn is_unknown(self) -> bool {
        self.0 == Self::UNKNOWN
    }

    /// Returns `true` if the gas cost is dynamic.
    #[inline]
    pub const fn is_dynamic(self) -> bool {
        self.0 & Self::DYNAMIC != 0
    }

    /// Returns `true` if the opcode is known, but disabled in the current EVM version.
    #[inline]
    pub const fn is_disabled(self) -> bool {
        self.0 & Self::DISABLED != 0
    }

    /// Returns the gas cost of the opcode.
    #[inline]
    pub const fn static_gas(self) -> Option<u16> {
        if self.is_dynamic() {
            None
        } else {
            Some(self.0 & Self::MASK)
        }
    }

    /// Sets the unknown flag.
    #[inline]
    pub fn set_unknown(&mut self) {
        self.0 = Self::UNKNOWN;
    }

    /// Sets the dynamic flag.
    #[inline]
    pub fn set_dynamic(&mut self) {
        self.0 |= Self::DYNAMIC;
    }

    /// Sets the disabled flag.
    #[inline]
    pub fn set_disabled(&mut self) {
        self.0 |= Self::DISABLED;
    }

    /// Sets the gas cost.
    ///
    /// # Panics
    ///
    /// Panics if the gas cost is greater than [`Self::MASK`].
    #[inline]
    #[track_caller]
    pub fn set_gas(&mut self, gas: u16) {
        assert!(gas <= Self::MASK);
        self.0 = (self.0 & !Self::MASK) | (gas & Self::MASK);
    }
}

/// Returns the static gas map for the given spec ID.
///
/// The map will contain the gas costs for each opcode if it is known statically, or `0` if the
/// opcode is not known or its gas cost is dynamic.
pub const fn op_info_map(spec_id: SpecId) -> &'static [OpcodeInfo; 256] {
    spec_to_generic!(spec_id, {
        const MAP: &[OpcodeInfo; 256] = &make_map(<SPEC as revm_primitives::Spec>::SPEC_ID);
        MAP
    })
}

#[allow(unused_mut)]
const fn make_map(spec_id: SpecId) -> [OpcodeInfo; 256] {
    const DYNAMIC: u16 = OpcodeInfo::DYNAMIC;

    let mut map = [OpcodeInfo(OpcodeInfo::UNKNOWN); 256];
    macro_rules! set {
        ($($op:ident = $gas:expr $(, if $spec:ident)? ;)*) => {
            $(
                let mut g = $gas;
                $(
                    if (spec_id as u8) < SpecId::$spec as u8 {
                        g |= OpcodeInfo::DISABLED;
                    }
                )?
                map[op::$op as usize] = OpcodeInfo::new(g);
            )*
        };
    }
    set! {
        STOP = 0;

        ADD        = 3;
        MUL        = 5;
        SUB        = 3;
        DIV        = 5;
        SDIV       = 5;
        MOD        = 5;
        SMOD       = 5;
        ADDMOD     = 8;
        MULMOD     = 8;
        EXP        = DYNAMIC;
        SIGNEXTEND = 5;
        // 0x0C
        // 0x0D
        // 0x0E
        // 0x0F
        LT     = 3;
        GT     = 3;
        SLT    = 3;
        SGT    = 3;
        EQ     = 3;
        ISZERO = 3;
        AND    = 3;
        OR     = 3;
        XOR    = 3;
        NOT    = 3;
        BYTE   = 3;
        SHL    = 3, if CONSTANTINOPLE;
        SHR    = 3, if CONSTANTINOPLE;
        SAR    = 3, if CONSTANTINOPLE;
        // 0x1E
        // 0x1F
        KECCAK256 = DYNAMIC;
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
        ADDRESS      = 2;
        BALANCE      = 20;
        ORIGIN       = 2;
        CALLER       = 2;
        CALLVALUE    = 2;
        CALLDATALOAD = 3;
        CALLDATASIZE = 2;
        CALLDATACOPY = DYNAMIC;
        CODESIZE     = 2;
        CODECOPY     = DYNAMIC;

        GASPRICE       = 2;
        EXTCODESIZE    = 20;
        EXTCODECOPY    = DYNAMIC;
        RETURNDATASIZE = 2, if BYZANTIUM;
        RETURNDATACOPY = DYNAMIC, if BYZANTIUM;
        EXTCODEHASH    = DYNAMIC, if BERLIN;
        BLOCKHASH      = 20;
        COINBASE       = 2;
        TIMESTAMP      = 2;
        NUMBER         = 2;
        DIFFICULTY     = 2;
        GASLIMIT       = 2;
        CHAINID        = 2, if ISTANBUL;
        SELFBALANCE    = 2, if ISTANBUL;
        BASEFEE        = 2, if LONDON;
        BLOBHASH       = 2, if CANCUN;
        BLOBBASEFEE    = 2, if CANCUN;
        // 0x4B
        // 0x4C
        // 0x4D
        // 0x4E
        // 0x4F
        POP      = 2;
        MLOAD    = DYNAMIC;
        MSTORE   = DYNAMIC;
        MSTORE8  = DYNAMIC;
        SLOAD    = sload_cost(spec_id);
        SSTORE   = DYNAMIC;
        JUMP     = 8;
        JUMPI    = 10;
        PC       = 2;
        MSIZE    = 2;
        GAS      = 2;
        JUMPDEST = 1;
        TLOAD    = 100, if CANCUN;
        TSTORE   = 100, if CANCUN;
        MCOPY    = DYNAMIC, if CANCUN;

        PUSH0  = 2, if SHANGHAI;
        PUSH1  = 3;
        PUSH2  = 3;
        PUSH3  = 3;
        PUSH4  = 3;
        PUSH5  = 3;
        PUSH6  = 3;
        PUSH7  = 3;
        PUSH8  = 3;
        PUSH9  = 3;
        PUSH10 = 3;
        PUSH11 = 3;
        PUSH12 = 3;
        PUSH13 = 3;
        PUSH14 = 3;
        PUSH15 = 3;
        PUSH16 = 3;
        PUSH17 = 3;
        PUSH18 = 3;
        PUSH19 = 3;
        PUSH20 = 3;
        PUSH21 = 3;
        PUSH22 = 3;
        PUSH23 = 3;
        PUSH24 = 3;
        PUSH25 = 3;
        PUSH26 = 3;
        PUSH27 = 3;
        PUSH28 = 3;
        PUSH29 = 3;
        PUSH30 = 3;
        PUSH31 = 3;
        PUSH32 = 3;

        DUP1  = 3;
        DUP2  = 3;
        DUP3  = 3;
        DUP4  = 3;
        DUP5  = 3;
        DUP6  = 3;
        DUP7  = 3;
        DUP8  = 3;
        DUP9  = 3;
        DUP10 = 3;
        DUP11 = 3;
        DUP12 = 3;
        DUP13 = 3;
        DUP14 = 3;
        DUP15 = 3;
        DUP16 = 3;

        SWAP1  = 3;
        SWAP2  = 3;
        SWAP3  = 3;
        SWAP4  = 3;
        SWAP5  = 3;
        SWAP6  = 3;
        SWAP7  = 3;
        SWAP8  = 3;
        SWAP9  = 3;
        SWAP10 = 3;
        SWAP11 = 3;
        SWAP12 = 3;
        SWAP13 = 3;
        SWAP14 = 3;
        SWAP15 = 3;
        SWAP16 = 3;

        LOG0 = DYNAMIC;
        LOG1 = DYNAMIC;
        LOG2 = DYNAMIC;
        LOG3 = DYNAMIC;
        LOG4 = DYNAMIC;
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
        CREATE       = DYNAMIC;
        CALL         = DYNAMIC;
        CALLCODE     = DYNAMIC;
        RETURN       = DYNAMIC;
        DELEGATECALL = DYNAMIC;
        CREATE2      = DYNAMIC;
        // 0xF6
        // 0xF7
        // 0xF8
        // 0xF9
        STATICCALL   = DYNAMIC;
        // 0xFB
        // 0xFC
        REVERT       = 0;
        INVALID      = 0;
        SELFDESTRUCT = 0;
    }
    map
}

/// [`gas::sload_cost`]
const fn sload_cost(spec_id: SpecId) -> u16 {
    if enabled(spec_id, SpecId::BERLIN) {
        OpcodeInfo::DYNAMIC
        // if is_cold {
        //     COLD_SLOAD_COST
        // } else {
        //     WARM_STORAGE_READ_COST
        // }
    } else if enabled(spec_id, SpecId::ISTANBUL) {
        // EIP-1884: Repricing for trie-size-dependent opcodes
        gas::INSTANBUL_SLOAD_GAS as _
    } else if enabled(spec_id, SpecId::TANGERINE) {
        // EIP-150: Gas cost changes for IO-heavy operations
        200
    } else {
        50
    }
}

const fn enabled(spec_id: SpecId, fork: SpecId) -> bool {
    spec_id as u8 >= fork as u8
}

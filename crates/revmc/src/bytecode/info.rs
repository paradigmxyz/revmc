use revm_bytecode::opcode as op;
use revm_interpreter::{
    host::DummyHost, instructions::instruction_table_gas_changes_spec, interpreter::EthInterpreter,
};
use revm_primitives::hardfork::SpecId;

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
    pub const MASK: u16 = 0b0000_1111_1111_1111;

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

    /// Returns `true` if the opcode is known, but disabled in the current EVM
    /// version.
    #[inline]
    pub const fn is_disabled(self) -> bool {
        self.0 & Self::DISABLED != 0
    }

    /// Returns the base gas cost of the opcode.
    ///
    /// This may not be the final/full gas cost of the opcode as it may also have a dynamic cost.
    #[inline]
    pub const fn base_gas(self) -> u16 {
        self.0 & Self::MASK
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

/// Returns the static info map for the given `SpecId`.
pub fn op_info_map(spec_id: SpecId) -> &'static [OpcodeInfo; 256] {
    use std::sync::OnceLock;
    static MAPS: OnceLock<[[OpcodeInfo; 256]; 32]> = OnceLock::new();
    let maps = MAPS.get_or_init(|| {
        let mut maps = [[OpcodeInfo(OpcodeInfo::UNKNOWN); 256]; 32];
        for (i, map) in maps.iter_mut().enumerate() {
            if let Ok(spec) = SpecId::try_from(i as u8) {
                *map = make_map(spec);
            }
        }
        maps
    });
    &maps[spec_id as usize]
}

/// Opcodes that have a dynamic gas component in addition to (or instead of) the static cost.
///
/// - `[1]`: Not dynamic in all `SpecId`s, but gas is calculated dynamically in a builtin.
/// - `[2]`: Dynamic with a base cost. Only the dynamic part is paid in builtins.
const DYNAMIC_OPCODES: &[u8] = &[
    op::EXP,          // [2]
    op::KECCAK256,    // [2]
    op::BALANCE,      // [1]
    op::CALLDATACOPY, // [2]
    op::CODECOPY,     // [2]
    op::EXTCODESIZE,  // [1]
    op::EXTCODECOPY,
    op::RETURNDATACOPY, // [2]
    op::EXTCODEHASH,    // [1]
    op::MLOAD,          // [2]
    op::MSTORE,         // [2]
    op::MSTORE8,        // [2]
    op::SLOAD,          // [1]
    op::SSTORE,
    op::MCOPY, // [2]
    op::LOG0,  // [2]
    op::LOG1,  // [2]
    op::LOG2,  // [2]
    op::LOG3,  // [2]
    op::LOG4,  // [2]
    op::CREATE,
    op::CALL,
    op::CALLCODE,
    op::RETURN,
    op::DELEGATECALL,
    op::CREATE2,
    op::STATICCALL,
    op::REVERT,
    op::SELFDESTRUCT,
];

/// Opcodes that are gated behind a specific `SpecId`, paired with the spec they were introduced in.
const SPEC_GATED_OPCODES: &[(u8, SpecId)] = &[
    (op::SHL, SpecId::CONSTANTINOPLE),
    (op::SHR, SpecId::CONSTANTINOPLE),
    (op::SAR, SpecId::CONSTANTINOPLE),
    (op::CLZ, SpecId::OSAKA),
    (op::RETURNDATASIZE, SpecId::BYZANTIUM),
    (op::RETURNDATACOPY, SpecId::BYZANTIUM),
    (op::EXTCODEHASH, SpecId::CONSTANTINOPLE),
    (op::CHAINID, SpecId::ISTANBUL),
    (op::SELFBALANCE, SpecId::ISTANBUL),
    (op::BASEFEE, SpecId::LONDON),
    (op::BLOBHASH, SpecId::CANCUN),
    (op::BLOBBASEFEE, SpecId::CANCUN),
    (op::TLOAD, SpecId::CANCUN),
    (op::TSTORE, SpecId::CANCUN),
    (op::MCOPY, SpecId::CANCUN),
    (op::PUSH0, SpecId::SHANGHAI),
    (op::DELEGATECALL, SpecId::HOMESTEAD),
    (op::CREATE2, SpecId::PETERSBURG),
    (op::STATICCALL, SpecId::BYZANTIUM),
    (op::REVERT, SpecId::BYZANTIUM),
];

/// Opcodes present in the upstream instruction table but not supported by revmc (e.g. EOF-only).
const UNSUPPORTED_OPCODES: &[u8] = &[op::DUPN, op::SWAPN, op::EXCHANGE, op::SLOTNUM];

fn make_map(spec_id: SpecId) -> [OpcodeInfo; 256] {
    let table = instruction_table_gas_changes_spec::<EthInterpreter, DummyHost>(spec_id);

    let mut map = [OpcodeInfo(OpcodeInfo::UNKNOWN); 256];

    for i in 0..256u16 {
        let op = i as u8;

        // Skip opcodes not defined in revm's opcode table.
        if revm_bytecode::opcode::OpCode::new(op).is_none() {
            continue;
        }

        // Skip opcodes not supported by revmc (e.g. EOF-only).
        if UNSUPPORTED_OPCODES.contains(&op) {
            continue;
        }

        let static_gas = table[op as usize].static_gas();

        // LOG opcodes: upstream only uses the base LOG cost as static gas and handles per-topic
        // cost dynamically. revmc deducts the full static portion (base + n * LOGTOPIC) upfront,
        // so add the per-topic cost here.
        let gas = if (op::LOG0..=op::LOG4).contains(&op) {
            let n_topics = (op - op::LOG0) as u64;
            let full = static_gas + n_topics * revm_interpreter::instructions::gas::LOGTOPIC;
            full as u16
        } else {
            assert!(
                static_gas <= OpcodeInfo::MASK as u64,
                "static gas for opcode 0x{op:02X} exceeds OpcodeInfo capacity: {static_gas}"
            );
            static_gas as u16
        };

        let mut info = OpcodeInfo::new(gas);

        if DYNAMIC_OPCODES.contains(&op) {
            info.set_dynamic();
        }

        map[op as usize] = info;
    }

    // Apply spec-gating: mark opcodes as disabled if the current spec is before their introduction.
    for &(op, required_spec) in SPEC_GATED_OPCODES {
        if (spec_id as u8) < (required_spec as u8) {
            map[op as usize].set_disabled();
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clz_flags() {
        let arrow_glacier = op_info_map(SpecId::ARROW_GLACIER);
        let cancun = op_info_map(SpecId::CANCUN);
        let osaka = op_info_map(SpecId::OSAKA);

        let clz_ag = arrow_glacier[op::CLZ as usize];
        let clz_cancun = cancun[op::CLZ as usize];
        let clz_osaka = osaka[op::CLZ as usize];

        eprintln!(
            "CLZ on ARROW_GLACIER: is_unknown={}, is_disabled={}, raw={:#06x}",
            clz_ag.is_unknown(),
            clz_ag.is_disabled(),
            clz_ag.0
        );
        eprintln!(
            "CLZ on CANCUN: is_unknown={}, is_disabled={}, raw={:#06x}",
            clz_cancun.is_unknown(),
            clz_cancun.is_disabled(),
            clz_cancun.0
        );
        eprintln!(
            "CLZ on OSAKA: is_unknown={}, is_disabled={}, raw={:#06x}",
            clz_osaka.is_unknown(),
            clz_osaka.is_disabled(),
            clz_osaka.0
        );

        assert!(!clz_ag.is_unknown(), "CLZ should not be unknown on pre-OSAKA");
        assert!(clz_ag.is_disabled(), "CLZ should be disabled on ARROW_GLACIER");
        assert!(!clz_cancun.is_unknown(), "CLZ should not be unknown on pre-OSAKA");
        assert!(clz_cancun.is_disabled(), "CLZ should be disabled on CANCUN");
        assert!(!clz_osaka.is_unknown(), "CLZ should not be unknown on OSAKA");
        assert!(!clz_osaka.is_disabled(), "CLZ should not be disabled on OSAKA");
    }

    #[test]
    fn test_gas_values() {
        let cancun = op_info_map(SpecId::CANCUN);

        // Basic arithmetic.
        assert_eq!(cancun[op::ADD as usize].base_gas(), 3);
        assert_eq!(cancun[op::MUL as usize].base_gas(), 5);
        assert_eq!(cancun[op::ADDMOD as usize].base_gas(), 8);

        // EXP: base 10, dynamic.
        assert_eq!(cancun[op::EXP as usize].base_gas(), 10);
        assert!(cancun[op::EXP as usize].is_dynamic());

        // PUSH/DUP/SWAP gas.
        assert_eq!(cancun[op::PUSH1 as usize].base_gas(), 3);
        assert_eq!(cancun[op::DUP1 as usize].base_gas(), 3);
        assert_eq!(cancun[op::SWAP1 as usize].base_gas(), 3);
        assert_eq!(cancun[op::PUSH0 as usize].base_gas(), 2);

        // LOG: base + n * LOGTOPIC.
        assert_eq!(cancun[op::LOG0 as usize].base_gas(), 375);
        assert_eq!(cancun[op::LOG1 as usize].base_gas(), 750);
        assert_eq!(cancun[op::LOG2 as usize].base_gas(), 1125);
        assert_eq!(cancun[op::LOG3 as usize].base_gas(), 1500);
        assert_eq!(cancun[op::LOG4 as usize].base_gas(), 1875);
        assert!(cancun[op::LOG0 as usize].is_dynamic());

        // Memory ops: dynamic with base cost 3.
        assert_eq!(cancun[op::MLOAD as usize].base_gas(), 3);
        assert!(cancun[op::MLOAD as usize].is_dynamic());
        assert_eq!(cancun[op::KECCAK256 as usize].base_gas(), 30);
        assert!(cancun[op::KECCAK256 as usize].is_dynamic());

        // Storage: dynamic, warm storage read cost in Berlin+.
        assert!(cancun[op::SLOAD as usize].is_dynamic());
        assert_eq!(
            cancun[op::SLOAD as usize].base_gas(),
            revm_interpreter::instructions::gas::WARM_STORAGE_READ_COST as u16
        );

        // Transient storage (Cancun).
        assert_eq!(cancun[op::TLOAD as usize].base_gas(), 100);
        assert!(!cancun[op::TLOAD as usize].is_disabled());

        // Unknown opcode.
        assert!(cancun[0x0C].is_unknown());

        // Spec-gated: PUSH0 disabled before Shanghai.
        let pre_shanghai = op_info_map(SpecId::MERGE);
        assert!(pre_shanghai[op::PUSH0 as usize].is_disabled());
        assert!(!cancun[op::PUSH0 as usize].is_disabled());

        // Spec changes: SLOAD gas differs across specs.
        let frontier = op_info_map(SpecId::FRONTIER);
        assert_eq!(frontier[op::SLOAD as usize].base_gas(), 50); // base cost
    }
}

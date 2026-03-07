//! Gas calculation utilities.

use revm_primitives::hardfork::SpecId;

pub use revm_interpreter::gas::*;

/// Calculate SLOAD gas cost (pre-Berlin only).
#[inline]
pub const fn sload_cost(spec_id: SpecId, is_cold: bool) -> u64 {
    if spec_id.is_enabled_in(SpecId::BERLIN) {
        if is_cold {
            COLD_SLOAD_COST
        } else {
            WARM_STORAGE_READ_COST
        }
    } else if spec_id.is_enabled_in(SpecId::ISTANBUL) {
        ISTANBUL_SLOAD_GAS
    } else if spec_id.is_enabled_in(SpecId::TANGERINE) {
        200
    } else {
        50
    }
}

// These are overridden to only account for the dynamic cost.

/// `LOG` opcode cost calculation.
#[inline]
pub const fn dyn_log_cost(len: u64) -> Option<u64> {
    LOGDATA.checked_mul(len)
}

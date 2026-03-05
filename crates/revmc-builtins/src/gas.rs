//! Gas calculation utilities.

use revm_interpreter::{SStoreResult, StateLoad};
use revm_primitives::{hardfork::SpecId, U256};

pub use revm_context_interface::journaled_state::AccountLoad;
pub use revm_interpreter::gas::*;

/// `const` Option `?`.
#[allow(unused_macros)]
macro_rules! tri {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => return None,
        }
    };
}

/// Gas cost for SELFDESTRUCT operation.
pub const SELFDESTRUCT: i64 = SELFDESTRUCT_REFUND;

/// Minimum gas required for callee.
pub const MIN_CALLEE_GAS: u64 = 2300;

/// Calculate gas cost per word.
#[inline]
pub const fn cost_per_word(len: u64, per_word_cost: u64) -> Option<u64> {
    let words = len.div_ceil(32);
    words.checked_mul(per_word_cost)
}

/// Returns warm/cold cost based on is_cold flag.
#[inline]
pub const fn warm_cold_cost(is_cold: bool) -> u64 {
    if is_cold {
        COLD_ACCOUNT_ACCESS_COST
    } else {
        WARM_STORAGE_READ_COST
    }
}

/// Returns warm/cold cost with delegation support.
#[inline]
pub fn warm_cold_cost_with_delegation(state: StateLoad<bool>) -> u64 {
    if state.is_cold {
        COLD_ACCOUNT_ACCESS_COST
    } else {
        WARM_STORAGE_READ_COST
    }
}

/// Calculate EXTCODECOPY gas cost.
#[inline]
pub fn extcodecopy_cost(spec_id: SpecId, len: u64, is_cold: bool) -> Option<u64> {
    let base = if spec_id.is_enabled_in(SpecId::BERLIN) {
        warm_cold_cost(is_cold)
    } else if spec_id.is_enabled_in(SpecId::TANGERINE) {
        700
    } else {
        20
    };
    cost_per_word(len, COPY).map(|word_cost| base.saturating_add(word_cost))
}

/// Calculate SLOAD gas cost.
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

/// Calculate SSTORE gas cost.
#[inline]
pub fn sstore_cost(
    spec_id: SpecId,
    result: &SStoreResult,
    remaining_gas: u64,
    is_cold: bool,
) -> Option<u64> {
    if spec_id.is_enabled_in(SpecId::BERLIN) {
        let base = if is_cold { COLD_SLOAD_COST } else { 0 };
        let cost = if result.original_value == result.new_value {
            WARM_STORAGE_READ_COST
        } else if result.original_value == result.present_value {
            if result.original_value.is_zero() {
                SSTORE_SET
            } else {
                WARM_SSTORE_RESET
            }
        } else {
            WARM_STORAGE_READ_COST
        };
        Some(base.saturating_add(cost))
    } else if spec_id.is_enabled_in(SpecId::ISTANBUL) {
        let stipend = 2300;
        if remaining_gas <= stipend {
            return None;
        }
        let cost = if result.original_value == result.new_value {
            ISTANBUL_SLOAD_GAS
        } else if result.original_value == result.present_value {
            if result.original_value.is_zero() {
                SSTORE_SET
            } else {
                SSTORE_RESET
            }
        } else {
            ISTANBUL_SLOAD_GAS
        };
        Some(cost)
    } else {
        let cost = if result.present_value.is_zero() && !result.new_value.is_zero() {
            SSTORE_SET
        } else {
            SSTORE_RESET
        };
        Some(cost)
    }
}

/// Calculate SSTORE refund.
#[inline]
pub fn sstore_refund(spec_id: SpecId, result: &SStoreResult) -> i64 {
    if spec_id.is_enabled_in(SpecId::BERLIN) {
        // EIP-3529: Reduction in refunds (London+).
        // Replace `REFUND_SSTORE_CLEARS` (15000) with
        // `WARM_SSTORE_RESET + ACCESS_LIST_STORAGE_KEY` (4800).
        let sstore_clears_refund = if spec_id.is_enabled_in(SpecId::LONDON) {
            (WARM_SSTORE_RESET + ACCESS_LIST_STORAGE_KEY) as i64
        } else {
            REFUND_SSTORE_CLEARS
        };

        let mut refund = 0i64;
        if result.original_value != result.present_value
            && result.original_value == result.new_value
        {
            if result.original_value.is_zero() {
                refund += (SSTORE_SET - WARM_STORAGE_READ_COST) as i64;
            } else {
                refund += (WARM_SSTORE_RESET - WARM_STORAGE_READ_COST) as i64;
            }
        }
        if !result.present_value.is_zero() && result.new_value.is_zero() {
            refund += sstore_clears_refund;
        }
        if !result.original_value.is_zero()
            && result.present_value.is_zero()
            && result.new_value == result.original_value
        {
            refund -= sstore_clears_refund;
        }
        refund
    } else if spec_id.is_enabled_in(SpecId::ISTANBUL) {
        let mut refund = 0i64;
        if result.original_value != result.present_value
            && result.original_value == result.new_value
        {
            if result.original_value.is_zero() {
                refund += (SSTORE_SET - ISTANBUL_SLOAD_GAS) as i64;
            } else {
                refund += (SSTORE_RESET - ISTANBUL_SLOAD_GAS) as i64;
            }
        }
        if !result.present_value.is_zero() && result.new_value.is_zero() {
            refund += REFUND_SSTORE_CLEARS;
        }
        if !result.original_value.is_zero()
            && result.present_value.is_zero()
            && result.new_value == result.original_value
        {
            refund -= REFUND_SSTORE_CLEARS;
        }
        refund
    } else if !result.present_value.is_zero() && result.new_value.is_zero() {
        REFUND_SSTORE_CLEARS
    } else {
        0
    }
}

/// Calculate CALL gas cost.
#[inline]
pub fn call_cost(
    spec_id: SpecId,
    transfers_value: bool,
    account_load: StateLoad<AccountLoad>,
) -> u64 {
    let mut gas = if spec_id.is_enabled_in(SpecId::BERLIN) {
        warm_cold_cost(account_load.is_cold)
            + if let Some(is_cold) = account_load.data.is_delegate_account_cold {
                warm_cold_cost(is_cold)
            } else {
                0
            }
    } else if spec_id.is_enabled_in(SpecId::TANGERINE) {
        700
    } else {
        40
    };

    if transfers_value {
        gas += CALLVALUE;
    }

    if account_load.data.is_empty
        && (transfers_value || !spec_id.is_enabled_in(SpecId::SPURIOUS_DRAGON))
    {
        gas += NEWACCOUNT;
    }

    gas
}

/// Calculate CREATE2 gas cost.
#[inline]
pub fn create2_cost(len: u64) -> Option<u64> {
    cost_per_word(len, KECCAK256WORD).map(|x| CREATE.saturating_add(x))
}

/// Calculate initcode gas cost.
#[inline]
pub const fn initcode_cost(len: u64) -> u64 {
    let words = len.div_ceil(32);
    words.saturating_mul(INITCODE_WORD_COST)
}

// These are overridden to only account for the dynamic cost.

/// `EXP` opcode cost calculation.
#[inline]
pub fn dyn_exp_cost(spec_id: SpecId, power: U256) -> Option<u64> {
    #[inline]
    const fn log2floor(value: U256) -> u64 {
        let mut l: u64 = 256;
        let mut i = 3;
        loop {
            if value.as_limbs()[i] == 0u64 {
                l -= 64;
            } else {
                l -= value.as_limbs()[i].leading_zeros() as u64;
                if l == 0 {
                    return l;
                } else {
                    return l - 1;
                }
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
        l
    }

    if power == U256::ZERO {
        Some(0)
    } else {
        // EIP-160: EXP cost increase
        let gas_byte =
            U256::from(if spec_id.is_enabled_in(SpecId::SPURIOUS_DRAGON) { 50 } else { 10 });
        let gas = gas_byte.checked_mul(U256::from(log2floor(power) / 8 + 1))?;
        u64::try_from(gas).ok()
    }
}

/// `LOG` opcode cost calculation.
#[inline]
pub const fn dyn_log_cost(len: u64) -> Option<u64> {
    LOGDATA.checked_mul(len)
}

/// `KECCAK256` opcode cost calculation.
#[inline]
pub const fn dyn_keccak256_cost(len: u64) -> Option<u64> {
    cost_per_word(len, KECCAK256WORD)
}

/// `*COPY` opcodes cost calculation.
#[inline]
pub const fn dyn_verylowcopy_cost(len: u64) -> Option<u64> {
    cost_per_word(len, COPY)
}

/// `EXP` opcode gas cost (base + dynamic).
#[inline]
pub fn exp_cost(spec_id: SpecId, power: U256) -> Option<u64> {
    dyn_exp_cost(spec_id, power).map(|dyn_cost| EXP.saturating_add(dyn_cost))
}

/// `LOG` opcode gas cost (base + topics + dynamic).
#[inline]
pub const fn log_cost(n_topics: u8, len: u64) -> Option<u64> {
    let base = LOG + (LOGTOPIC * n_topics as u64);
    match dyn_log_cost(len) {
        Some(dyn_cost) => Some(base.saturating_add(dyn_cost)),
        None => None,
    }
}

/// `KECCAK256` opcode gas cost (base + dynamic).
#[inline]
pub const fn keccak256_cost(len: u64) -> Option<u64> {
    match dyn_keccak256_cost(len) {
        Some(dyn_cost) => Some(KECCAK256.saturating_add(dyn_cost)),
        None => None,
    }
}

/// `CALLDATACOPY`, `CODECOPY`, `RETURNDATACOPY` opcode gas cost (base + dynamic).
#[inline]
pub const fn verylowcopy_cost(len: u64) -> Option<u64> {
    match dyn_verylowcopy_cost(len) {
        Some(dyn_cost) => Some(VERYLOW.saturating_add(dyn_cost)),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_cost() {
        for (spec_id, power) in [
            (SpecId::CANCUN, U256::from(0)),
            (SpecId::CANCUN, U256::from(1)),
            (SpecId::CANCUN, U256::from(69)),
        ] {
            assert_eq!(
                super::exp_cost(spec_id, power).unwrap(),
                EXP + dyn_exp_cost(spec_id, power).unwrap(),
            );
        }
    }

    #[test]
    fn log_cost() {
        for n_topics in [0, 1, 2] {
            for len in [0, 1, 69] {
                assert_eq!(
                    super::log_cost(n_topics, len).unwrap(),
                    LOG + (LOGTOPIC * n_topics as u64) + dyn_log_cost(len).unwrap(),
                );
            }
        }
    }

    #[test]
    fn keccak256_cost() {
        for len in [0, 1, 69] {
            assert_eq!(
                super::keccak256_cost(len).unwrap(),
                KECCAK256 + dyn_keccak256_cost(len).unwrap(),
            );
        }
    }

    #[test]
    fn verylowcopy_cost() {
        for len in [0, 1, 69] {
            assert_eq!(
                super::verylowcopy_cost(len).unwrap(),
                VERYLOW + dyn_verylowcopy_cost(len).unwrap(),
            );
        }
    }
}

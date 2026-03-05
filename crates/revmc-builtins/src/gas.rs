//! Gas calculation utilities.

use revm_primitives::{hardfork::SpecId, U256};

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

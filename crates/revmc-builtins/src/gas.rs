//! Gas calculation utilities.

use revm_primitives::{SpecId, U256};

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

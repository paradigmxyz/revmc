//! Gas calculation utilities.

use revm_primitives::hardfork::SpecId;

pub use revm_interpreter::gas::*;

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

// These are overridden to only account for the dynamic cost.

/// `LOG` opcode cost calculation.
#[inline]
pub const fn dyn_log_cost(len: u64) -> Option<u64> {
    LOGDATA.checked_mul(len)
}

/// `KECCAK256` opcode cost calculation.
#[inline]
pub const fn dyn_keccak256_cost(len: u64) -> Option<u64> {
    let words = len.div_ceil(32);
    words.checked_mul(KECCAK256WORD)
}

/// `*COPY` opcodes cost calculation.
#[inline]
pub const fn dyn_verylowcopy_cost(len: u64) -> Option<u64> {
    let words = len.div_ceil(32);
    words.checked_mul(COPY)
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

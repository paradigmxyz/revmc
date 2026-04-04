//! Peephole optimizations applied during translation.
//!
//! These fire when abstract interpretation has proven one or more operands are constant,
//! replacing expensive opaque builtins (DIV, MOD, SDIV, SMOD) with native LLVM operations
//! that it can optimize further (e.g. pow2 udiv → lshr, pow2 urem → and).

use super::translate::FunctionCx;
use crate::{Backend, Builder, InstData, IntCC};
use revm_bytecode::opcode as op;
use revm_primitives::U256;

/// i256 INT_MIN: 1 << 255.
const INT_MIN: U256 = U256::from_limbs([0, 0, 0, 1 << 63]);

impl<'a, B: Backend> FunctionCx<'a, B> {
    /// Tries to emit optimized inline code for an instruction whose operands are partially known.
    ///
    /// Returns `true` if the peephole fired and code was emitted, `false` to fall through to the
    /// normal translation.
    pub(super) fn try_peephole(&mut self, data: &InstData) -> bool {
        match data.opcode {
            op::DIV => self.peephole_div(),
            op::SDIV => self.peephole_sdiv(),
            op::MOD => self.peephole_mod(),
            op::SMOD => self.peephole_smod(),
            _ => false,
        }
    }

    /// DIV a, b => a / b.
    fn peephole_div(&mut self) -> bool {
        let [dividend, divisor] = self.const_operands();
        match divisor {
            // x / 0 => 0.
            Some(U256::ZERO) => {
                self.pop_ignore(2);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.push(zero);
            }
            // x / 1 => x.
            Some(U256::ONE) => {
                let [a, _] = self.popn();
                self.push(a);
            }
            // x / C (pow2) => native udiv, LLVM lowers to lshr.
            Some(d) if d.is_power_of_two() => {
                let [a, _] = self.popn();
                let d = self.bcx.iconst_256(d);
                let r = self.bcx.udiv(a, d);
                self.push(r);
            }
            // 0 / x => 0 (EVM: 0 / 0 = 0).
            _ if dividend == Some(U256::ZERO) => {
                self.pop_ignore(2);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.push(zero);
            }
            _ => return false,
        }
        true
    }

    /// SDIV a, b => signed(a) / signed(b), rounded toward zero.
    fn peephole_sdiv(&mut self) -> bool {
        let [dividend, divisor] = self.const_operands();
        match divisor {
            // x / 0 => 0.
            Some(U256::ZERO) => {
                self.pop_ignore(2);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.push(zero);
            }
            // x / 1 => x.
            Some(U256::ONE) => {
                let [a, _] = self.popn();
                self.push(a);
            }
            // x / -1 => -x.
            Some(U256::MAX) => {
                let [a, _] = self.popn();
                let zero = self.bcx.iconst_256(U256::ZERO);
                let r = self.bcx.isub(zero, a);
                self.push(r);
            }
            // x / INT_MIN => zext(x == INT_MIN).
            Some(INT_MIN) => {
                let [a, _] = self.popn();
                let min = self.bcx.iconst_256(INT_MIN);
                let cmp = self.bcx.icmp(IntCC::Equal, a, min);
                let r = self.bcx.zext(self.word_type, cmp);
                self.push(r);
            }
            // 0 / x => 0 (EVM: 0 / 0 = 0).
            _ if dividend == Some(U256::ZERO) => {
                self.pop_ignore(2);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.push(zero);
            }
            _ => return false,
        }
        true
    }

    /// MOD a, b => a % b.
    fn peephole_mod(&mut self) -> bool {
        let [dividend, modulus] = self.const_operands();
        match modulus {
            // x % 0 => 0, x % 1 => 0.
            Some(U256::ZERO) | Some(U256::ONE) => {
                self.pop_ignore(2);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.push(zero);
            }
            // x % C (pow2) => native urem, LLVM lowers to and.
            Some(m) if m.is_power_of_two() => {
                let [a, _] = self.popn();
                let m = self.bcx.iconst_256(m);
                let r = self.bcx.urem(a, m);
                self.push(r);
            }
            // 0 % x => 0 (EVM: 0 % 0 = 0).
            _ if dividend == Some(U256::ZERO) => {
                self.pop_ignore(2);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.push(zero);
            }
            _ => return false,
        }
        true
    }

    /// SMOD a, b => signed(a) % signed(b). Result sign matches dividend.
    fn peephole_smod(&mut self) -> bool {
        let [dividend, modulus] = self.const_operands();
        match modulus {
            // x % 0 => 0, x % 1 => 0, x % -1 => 0.
            Some(U256::ZERO) | Some(U256::ONE) | Some(U256::MAX) => {
                self.pop_ignore(2);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.push(zero);
            }
            // 0 % x => 0 (EVM: 0 % 0 = 0).
            _ if dividend == Some(U256::ZERO) => {
                self.pop_ignore(2);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.push(zero);
            }
            _ => return false,
        }
        true
    }
}

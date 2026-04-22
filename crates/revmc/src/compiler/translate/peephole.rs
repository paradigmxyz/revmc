//! Peephole optimizations applied during translation.
//!
//! These fire when abstract interpretation has proven one or more operands are constant,
//! replacing expensive opaque builtins (DIV, MOD, SDIV, SMOD, ADDMOD, MULMOD) with native
//! LLVM operations that it can optimize further (e.g. pow2 udiv → lshr, pow2 urem → and).

use super::FunctionCx;
use crate::{Backend, Builder, InstData, IntCC};
use revm_bytecode::opcode as op;
use revm_interpreter::InstructionResult;
use revm_primitives::U256;
use revmc_builtins::Builtin;

/// i256 INT_MIN: 1 << 255.
const INT_MIN: U256 = U256::ONE.wrapping_shl(255);

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
            op::ADDMOD => self.peephole_addmod(),
            op::MULMOD => self.peephole_mulmod(),
            op::EXP => self.peephole_exp(),
            op::SIGNEXTEND => self.peephole_signextend(),
            op::BYTE => self.peephole_byte(),

            op::CALLDATALOAD | op::MLOAD | op::SLOAD => self.peephole_load(data.opcode),
            op::MSTORE => self.peephole_mstore(),

            op::KECCAK256 => self.peephole_keccak256(),
            op::RETURN | op::REVERT => self.peephole_return(data.opcode),

            _ => false,
        }
    }

    /// DIV a, b => a / b.
    ///
    /// General constant divisors could use native LLVM `udiv`, but i256 division
    /// generates very bloated code (~100+ instructions for the reciprocal multiply),
    /// so we only emit native ops for powers of two where LLVM lowers to `lshr`.
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
    ///
    /// Pow2 cases are intentionally skipped: signed division rounds toward zero while
    /// arithmetic shift right rounds toward negative infinity, so the semantics differ.
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
    ///
    /// Same as DIV: only pow2 moduli use native `urem` (LLVM lowers to `and`).
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

    /// ADDMOD a, b, N => (a + b) % N.
    fn peephole_addmod(&mut self) -> bool {
        let [a, b, modulus] = self.const_operands();
        match modulus {
            // (a + b) % 0 => 0, (a + b) % 1 => 0.
            Some(U256::ZERO) | Some(U256::ONE) => {
                self.pop_ignore(3);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.push(zero);
            }
            _ => match (a, b) {
                // (0 + 0) % N => 0.
                (Some(U256::ZERO), Some(U256::ZERO)) => {
                    self.pop_ignore(3);
                    let zero = self.bcx.iconst_256(U256::ZERO);
                    self.push(zero);
                }
                _ => return false,
            },
        }
        true
    }

    /// MULMOD a, b, N => (a * b) % N.
    fn peephole_mulmod(&mut self) -> bool {
        let [a, b, modulus] = self.const_operands();
        match modulus {
            // (a * b) % 0 => 0, (a * b) % 1 => 0.
            Some(U256::ZERO) | Some(U256::ONE) => {
                self.pop_ignore(3);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.push(zero);
            }
            _ => {
                // a * 0 => 0, 0 * b => 0.
                if a == Some(U256::ZERO) || b == Some(U256::ZERO) {
                    self.pop_ignore(3);
                    let zero = self.bcx.iconst_256(U256::ZERO);
                    self.push(zero);
                } else {
                    return false;
                }
            }
        }
        true
    }

    /// EXP base, exp => base ** exp.
    ///
    /// Dynamic gas is folded into the section gas cost by `SectionsAnalysis` when the
    /// exponent is a compile-time constant, so the section gas check already covers it.
    fn peephole_exp(&mut self) -> bool {
        let [_base, exponent] = self.const_operands();
        match exponent {
            // x ** 0 => 1.
            Some(U256::ZERO) => {
                self.pop_ignore(2);
                let one = self.bcx.iconst_256(U256::from(1));
                self.push(one);
            }
            // x ** 1 => x.
            Some(U256::ONE) => {
                let [a, _] = self.popn();
                self.push(a);
            }
            // x ** 2 => x * x.
            Some(e) if e == U256::from(2) => {
                let [a, _] = self.popn();
                let r = self.bcx.imul(a, a);
                self.push(r);
            }
            _ => return false,
        }
        true
    }

    /// SIGNEXTEND ext, x => sign-extend x from (ext+1) bytes.
    fn peephole_signextend(&mut self) -> bool {
        let [ext, _x] = self.const_operands();
        match ext {
            // SIGNEXTEND(e, x) with e >= 31 => x (no-op, value already fills 32 bytes).
            Some(e) if e >= U256::from(31) => {
                let [_, x] = self.popn();
                self.push(x);
            }
            _ => return false,
        }
        true
    }

    /// BYTE index, value => `value[index]`.
    fn peephole_byte(&mut self) -> bool {
        let [index, _value] = self.const_operands();
        match index {
            // BYTE(i, x) with i >= 32 => 0.
            Some(i) if i >= U256::from(32) => {
                self.pop_ignore(2);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.push(zero);
            }
            _ => return false,
        }
        true
    }

    fn peephole_load(&mut self, op: u8) -> bool {
        if let [Some(offset)] = self.const_operands()
            && let Ok(offset) = u64::try_from(offset)
        {
            if op == op::CALLDATALOAD {
                let sp = self.sp_from_top(1);
                let offset = self.bcx.iconst(self.isize_type, offset as i64);
                self.emit_calldataload_inline(offset, sp);
                return true;
            }
            let builtin = match op {
                op::MLOAD => Builtin::MloadC,
                op::SLOAD => Builtin::SloadC,
                _ => unreachable!(),
            };
            let sp = self.sp_from_top(1);
            let offset = self.bcx.iconst(self.isize_type, offset as i64);
            let args = &[self.ecx, sp, offset];
            self.call_fallible_builtin(builtin, args);
            true
        } else {
            false
        }
    }

    fn peephole_mstore(&mut self) -> bool {
        let [offset, value] = self.const_operands();
        let offset = offset.and_then(|x| u64::try_from(x).ok());
        let value = value.and_then(|x| u64::try_from(x).ok());
        match (offset, value) {
            (Some(offset), None) => {
                let offset = self.bcx.iconst(self.isize_type, offset as i64);
                let sp = self.sp_after_inputs_with(&[1]);
                self.call_fallible_builtin(Builtin::MstoreCD, &[self.ecx, offset, sp]);
                true
            }
            (None, Some(value)) => {
                let value = self.bcx.iconst(self.isize_type, value as i64);
                let _ = self.sp_after_inputs_with(&[0]);
                let sp = self.sp_from_top(1);
                self.call_fallible_builtin(Builtin::MstoreDC, &[self.ecx, sp, value]);
                true
            }
            (Some(offset), Some(value)) => {
                let offset = self.bcx.iconst(self.isize_type, offset as i64);
                let value = self.bcx.iconst(self.isize_type, value as i64);
                self.call_fallible_builtin(Builtin::MstoreCC, &[self.ecx, offset, value]);
                true
            }
            (None, None) => false,
        }
    }

    fn peephole_keccak256(&mut self) -> bool {
        if let Some((offset, len)) = self.const_memory_operands(self.const_operands()) {
            let offset = self.bcx.iconst(self.isize_type, offset as i64);
            let len = self.bcx.iconst(self.isize_type, len as i64);
            let sp = self.sp_after_inputs_with(&[]);
            self.call_fallible_builtin(Builtin::Keccak256CC, &[self.ecx, sp, offset, len]);
            true
        } else {
            false
        }
    }

    fn peephole_return(&mut self, op: u8) -> bool {
        if let Some((offset, len)) = self.const_memory_operands(self.const_operands()) {
            let ir = match op {
                op::RETURN => InstructionResult::Return,
                op::REVERT => InstructionResult::Revert,
                _ => unreachable!(),
            };
            let ir_const = self.bcx.iconst(self.i8_type, ir as i64);
            let offset = self.bcx.iconst(self.isize_type, offset as i64);
            let len = self.bcx.iconst(self.isize_type, len as i64);
            let _ = self.call_builtin(Builtin::DoReturnCC, &[self.ecx, offset, len, ir_const]);
            self.bcx.unreachable();
            true
        } else {
            false
        }
    }

    /* utils */

    fn const_memory_operands(&self, args: [Option<U256>; 2]) -> Option<(u64, u64)> {
        if let [offset, Some(len)] = args
            && let Ok(len) = u64::try_from(len)
            // For len == 0 offset is ignored.
            && let Some(offset) = offset
                .and_then(|offset| u64::try_from(offset).ok())
                .or_else(|| (len == 0).then_some(0))
        {
            Some((offset, len))
        } else {
            None
        }
    }
}

//! Constant folding for EVM arithmetic during abstract interpretation.

use super::{
    super::{Interner, U256Idx},
    block_analysis::AbsValue,
};
use crate::InstData;
use revm_bytecode::opcode as op;
use revm_interpreter::instructions::i256::{i256_cmp, i256_div, i256_mod};
use revm_primitives::U256;
use std::cmp::Ordering;

/// Try to constant-fold an instruction.
///
/// `code_len` is the length of the bytecode (for `CODESIZE`).
pub(crate) fn try_const_fold(
    inst: &InstData,
    inputs: &[AbsValue],
    interner: &mut Interner<U256Idx, U256, alloy_primitives::map::FbBuildHasher<32>>,
    code_len: usize,
) -> Option<AbsValue> {
    let opcode = inst.opcode;
    let result = match opcode {
        // 0 -> 1
        op::CODESIZE => U256::from(code_len),
        op::PC => U256::from(inst.pc),

        // 1 -> 1
        op::ISZERO | op::NOT | op::CLZ => {
            let &[AbsValue::Const(ai)] = inputs else {
                return None;
            };
            let a = *interner.get(ai);
            match opcode {
                op::ISZERO => U256::from(a.is_zero()),
                op::NOT => !a,
                op::CLZ => U256::from(a.leading_zeros()),
                _ => unreachable!(),
            }
        }

        // 2 -> 1
        op::ADD
        | op::MUL
        | op::SUB
        | op::DIV
        | op::SDIV
        | op::MOD
        | op::SMOD
        | op::EXP
        | op::SIGNEXTEND
        | op::LT
        | op::GT
        | op::SLT
        | op::SGT
        | op::EQ
        | op::AND
        | op::OR
        | op::XOR
        | op::BYTE
        | op::SHL
        | op::SHR
        | op::SAR => {
            let &[AbsValue::Const(bi), AbsValue::Const(ai)] = inputs else {
                return None;
            };
            let a = *interner.get(ai);
            let b = *interner.get(bi);
            match opcode {
                op::ADD => a.wrapping_add(b),
                op::MUL => a.wrapping_mul(b),
                op::SUB => a.wrapping_sub(b),
                op::DIV => {
                    if !b.is_zero() {
                        a.wrapping_div(b)
                    } else {
                        U256::ZERO
                    }
                }
                op::SDIV => i256_div(a, b),
                op::MOD => {
                    if !b.is_zero() {
                        a.wrapping_rem(b)
                    } else {
                        U256::ZERO
                    }
                }
                op::SMOD => i256_mod(a, b),
                op::EXP => a.pow(b),
                op::SIGNEXTEND => {
                    if a < U256::from(31) {
                        let ext = a.as_limbs()[0];
                        let bit_index = (8 * ext + 7) as usize;
                        let bit = b.bit(bit_index);
                        let mask = (U256::from(1) << bit_index) - U256::from(1);
                        if bit { b | !mask } else { b & mask }
                    } else {
                        b
                    }
                }
                op::LT => U256::from(a < b),
                op::GT => U256::from(a > b),
                op::SLT => U256::from(i256_cmp(&a, &b) == Ordering::Less),
                op::SGT => U256::from(i256_cmp(&a, &b) == Ordering::Greater),
                op::EQ => U256::from(a == b),
                op::AND => a & b,
                op::OR => a | b,
                op::XOR => a ^ b,
                op::BYTE => {
                    let i = a.saturating_to::<usize>();
                    if i < 32 { U256::from(b.byte(31 - i)) } else { U256::ZERO }
                }
                op::SHL => {
                    let shift = a.saturating_to::<usize>();
                    if shift < 256 { b << shift } else { U256::ZERO }
                }
                op::SHR => {
                    let shift = a.saturating_to::<usize>();
                    if shift < 256 { b >> shift } else { U256::ZERO }
                }
                op::SAR => {
                    let shift = a.saturating_to::<usize>();
                    if shift < 256 {
                        b.arithmetic_shr(shift)
                    } else if b.bit(255) {
                        U256::MAX
                    } else {
                        U256::ZERO
                    }
                }
                _ => unreachable!(),
            }
        }

        // 3 -> 1
        op::ADDMOD | op::MULMOD => {
            let &[AbsValue::Const(ci), AbsValue::Const(bi), AbsValue::Const(ai)] = inputs else {
                return None;
            };
            let a = *interner.get(ai);
            let b = *interner.get(bi);
            let n = *interner.get(ci);
            match opcode {
                op::ADDMOD => a.add_mod(b, n),
                op::MULMOD => a.mul_mod(b, n),
                _ => unreachable!(),
            }
        }

        _ => return None,
    };
    Some(AbsValue::Const(interner.intern(result)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::passes::block_analysis::tests::{Inst, analyze_code};
    use revm_primitives::U256;

    /// Builds bytecode that pushes operands, executes `opcode`, sinks the result
    /// with `PUSH0 MSTORE STOP`, analyzes it, and returns the folded result.
    ///
    /// Operands are in natural order matching the yellow paper specification:
    /// for `SUB(a, b)` = `a - b`, pass `&[a, b]` (first popped = first element).
    fn const_fold(opcode: u8, operands: &[U256]) -> Option<U256> {
        let mut code = Vec::new();
        let mut num_insts = 0usize;
        // Push in reverse so that operands[0] ends up on TOS.
        for v in operands.iter().rev() {
            let bytes = v.to_be_bytes::<32>();
            let lz = bytes.iter().position(|&b| b != 0).unwrap_or(32);
            if lz == 32 {
                code.push(op::PUSH0);
            } else {
                let len = 32 - lz;
                code.push(op::PUSH0 + len as u8);
                code.extend_from_slice(&bytes[lz..]);
            }
            num_insts += 1;
        }
        code.push(opcode);
        num_insts += 1;
        // Sink: PUSH0 + MSTORE + STOP to consume the result.
        let mstore_inst = num_insts + 1;
        code.extend_from_slice(&[op::PUSH0, op::MSTORE, op::STOP]);

        let bytecode = analyze_code(code);
        // The folded result is operand 1 (second from top) at MSTORE.
        bytecode.const_operand(Inst::from_usize(mstore_inst), 1)
    }

    #[test]
    fn const_fold_0_to_1() {
        // CODESIZE: code = CODESIZE(1) + PUSH0(1) + MSTORE(1) + STOP(1) = 4 bytes.
        assert_eq!(const_fold(op::CODESIZE, &[]), Some(U256::from(4)));

        // PC: at position 0.
        assert_eq!(const_fold(op::PC, &[]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_1_to_1() {
        // ISZERO.
        assert_eq!(const_fold(op::ISZERO, &[U256::ZERO]), Some(U256::from(1)));
        assert_eq!(const_fold(op::ISZERO, &[U256::from(42)]), Some(U256::ZERO));

        // NOT.
        assert_eq!(const_fold(op::NOT, &[U256::ZERO]), Some(U256::MAX));
        assert_eq!(const_fold(op::NOT, &[U256::MAX]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_add() {
        assert_eq!(const_fold(op::ADD, &[U256::from(3), U256::from(4)]), Some(U256::from(7)));
        assert_eq!(const_fold(op::ADD, &[U256::MAX, U256::from(1)]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_mul() {
        assert_eq!(const_fold(op::MUL, &[U256::from(3), U256::from(7)]), Some(U256::from(21)));
        assert_eq!(const_fold(op::MUL, &[U256::from(5), U256::ZERO]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_sub() {
        // SUB(10, 3) = 7.
        assert_eq!(const_fold(op::SUB, &[U256::from(10), U256::from(3)]), Some(U256::from(7)));
        assert_eq!(const_fold(op::SUB, &[U256::ZERO, U256::from(1)]), Some(U256::MAX));
    }

    #[test]
    fn const_fold_div() {
        // DIV(10, 3) = 3.
        assert_eq!(const_fold(op::DIV, &[U256::from(10), U256::from(3)]), Some(U256::from(3)));
        assert_eq!(const_fold(op::DIV, &[U256::from(7), U256::ZERO]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_sdiv() {
        assert_eq!(const_fold(op::SDIV, &[U256::from(10), U256::from(3)]), Some(U256::from(3)));
        let neg10 = U256::ZERO.wrapping_sub(U256::from(10));
        let neg3 = U256::ZERO.wrapping_sub(U256::from(3));
        // SDIV(-10, 3) = -3.
        assert_eq!(const_fold(op::SDIV, &[neg10, U256::from(3)]), Some(neg3));
        assert_eq!(const_fold(op::SDIV, &[U256::from(7), U256::ZERO]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_mod() {
        // MOD(10, 3) = 1.
        assert_eq!(const_fold(op::MOD, &[U256::from(10), U256::from(3)]), Some(U256::from(1)));
        assert_eq!(const_fold(op::MOD, &[U256::from(7), U256::ZERO]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_smod() {
        let neg8 = U256::ZERO.wrapping_sub(U256::from(8));
        let neg3 = U256::ZERO.wrapping_sub(U256::from(3));
        let neg2 = U256::ZERO.wrapping_sub(U256::from(2));
        // SMOD(-8, 3) = -2.
        assert_eq!(const_fold(op::SMOD, &[neg8, U256::from(3)]), Some(neg2));
        // SMOD(-8, -3) = -2.
        assert_eq!(const_fold(op::SMOD, &[neg8, neg3]), Some(neg2));
    }

    #[test]
    fn const_fold_addmod() {
        // ADDMOD(10, 10, 8) = 4.
        assert_eq!(
            const_fold(op::ADDMOD, &[U256::from(10), U256::from(10), U256::from(8)]),
            Some(U256::from(4)),
        );
        assert_eq!(
            const_fold(op::ADDMOD, &[U256::from(1), U256::from(2), U256::ZERO]),
            Some(U256::ZERO),
        );
    }

    #[test]
    fn const_fold_mulmod() {
        // MULMOD(10, 10, 8) = 4.
        assert_eq!(
            const_fold(op::MULMOD, &[U256::from(10), U256::from(10), U256::from(8)]),
            Some(U256::from(4)),
        );
    }

    #[test]
    fn const_fold_exp() {
        // EXP(2, 10) = 1024.
        assert_eq!(const_fold(op::EXP, &[U256::from(2), U256::from(10)]), Some(U256::from(1024)));
        // EXP(5, 0) = 1.
        assert_eq!(const_fold(op::EXP, &[U256::from(5), U256::ZERO]), Some(U256::from(1)));
    }

    #[test]
    fn const_fold_signextend() {
        // SIGNEXTEND(0, 0xFF) -> all ones.
        assert_eq!(const_fold(op::SIGNEXTEND, &[U256::ZERO, U256::from(0xFF)]), Some(U256::MAX));
        // SIGNEXTEND(0, 0x7F) -> 0x7F.
        assert_eq!(
            const_fold(op::SIGNEXTEND, &[U256::ZERO, U256::from(0x7F)]),
            Some(U256::from(0x7F)),
        );
        // ext >= 31 -> no-op.
        assert_eq!(
            const_fold(op::SIGNEXTEND, &[U256::from(31), U256::from(0xFF)]),
            Some(U256::from(0xFF)),
        );
    }

    #[test]
    fn const_fold_lt_gt_eq() {
        // LT(1, 2) = 1.
        assert_eq!(const_fold(op::LT, &[U256::from(1), U256::from(2)]), Some(U256::from(1)));
        assert_eq!(const_fold(op::LT, &[U256::from(2), U256::from(1)]), Some(U256::ZERO));
        // GT(2, 1) = 1.
        assert_eq!(const_fold(op::GT, &[U256::from(2), U256::from(1)]), Some(U256::from(1)));
        assert_eq!(const_fold(op::GT, &[U256::from(1), U256::from(2)]), Some(U256::ZERO));
        assert_eq!(const_fold(op::EQ, &[U256::from(5), U256::from(5)]), Some(U256::from(1)));
        assert_eq!(const_fold(op::EQ, &[U256::from(5), U256::from(6)]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_slt_sgt() {
        let neg1 = U256::MAX;
        // SLT(-1, 1) = 1.
        assert_eq!(const_fold(op::SLT, &[neg1, U256::from(1)]), Some(U256::from(1)));
        // SLT(1, -1) = 0.
        assert_eq!(const_fold(op::SLT, &[U256::from(1), neg1]), Some(U256::ZERO));
        // SGT(1, -1) = 1.
        assert_eq!(const_fold(op::SGT, &[U256::from(1), neg1]), Some(U256::from(1)));
    }

    #[test]
    fn const_fold_and_or_xor() {
        assert_eq!(
            const_fold(op::AND, &[U256::from(0xFF), U256::from(0x0F)]),
            Some(U256::from(0x0F)),
        );
        assert_eq!(
            const_fold(op::OR, &[U256::from(0xF0), U256::from(0x0F)]),
            Some(U256::from(0xFF)),
        );
        assert_eq!(const_fold(op::XOR, &[U256::from(0xFF), U256::from(0xFF)]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_byte() {
        // BYTE(31, 0xFF) = 0xFF (last byte).
        assert_eq!(
            const_fold(op::BYTE, &[U256::from(31), U256::from(0xFF)]),
            Some(U256::from(0xFF)),
        );
        // BYTE(0, 0xFF) = 0 (0xFF is in byte 31 only).
        assert_eq!(const_fold(op::BYTE, &[U256::ZERO, U256::from(0xFF)]), Some(U256::ZERO));
        // Out of bounds.
        assert_eq!(const_fold(op::BYTE, &[U256::from(32), U256::from(0xFF)]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_shl_shr() {
        // SHL(1, 0x80) = 0x100.
        assert_eq!(
            const_fold(op::SHL, &[U256::from(1), U256::from(0x80)]),
            Some(U256::from(0x100)),
        );
        assert_eq!(const_fold(op::SHL, &[U256::from(256), U256::from(1)]), Some(U256::ZERO));
        // SHR(1, 0x80) = 0x40.
        assert_eq!(const_fold(op::SHR, &[U256::from(1), U256::from(0x80)]), Some(U256::from(0x40)),);
        assert_eq!(const_fold(op::SHR, &[U256::from(256), U256::from(1)]), Some(U256::ZERO));
    }

    #[test]
    fn const_fold_sar() {
        // SAR(1, 2) = 1.
        assert_eq!(const_fold(op::SAR, &[U256::from(1), U256::from(2)]), Some(U256::from(1)));
        // SAR(4, -1) = -1.
        assert_eq!(const_fold(op::SAR, &[U256::from(4), U256::MAX]), Some(U256::MAX));
        // SAR(256, -1) = -1.
        assert_eq!(const_fold(op::SAR, &[U256::from(256), U256::MAX]), Some(U256::MAX));
        // SAR(256, 1) = 0.
        assert_eq!(const_fold(op::SAR, &[U256::from(256), U256::from(1)]), Some(U256::ZERO));
    }
}

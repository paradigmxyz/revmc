#![allow(
    clippy::needless_update,
    unreachable_pub,
    dead_code,
    missing_docs,
    missing_debug_implementations
)]

use crate::*;
use primitives::SpecId;
use revm_interpreter::{
    analysis::{EofError, EofValidationError},
    gas, opcode as op, CallInputs, CreateInputs, Gas, InstructionResult, InterpreterAction,
    InterpreterResult,
};
use revm_primitives::{hex, keccak256, Address, Bytes, LogData, B256, KECCAK_EMPTY};

#[macro_use]
mod macros;

mod meta;

mod fibonacci;
mod resume;

mod runner;
pub use runner::*;

const I256_MAX: U256 = U256::from_limbs([
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0x7FFFFFFFFFFFFFFF,
]);

tests! {
    ret {
        empty(@raw {}),
        no_stop(@raw {
            bytecode: &[op::PUSH0],
            expected_stack: &[U256::ZERO],
            expected_gas: 2,
        }),
        stop(@raw {
            bytecode: &[op::STOP],
            expected_gas: 0,
        }),
        invalid(@raw {
            bytecode: &[op::INVALID],
            expected_return: InstructionResult::InvalidFEOpcode,
            expected_gas: 0,
        }),
        unknown(@raw {
            bytecode: &[0x21],
            expected_return: InstructionResult::OpcodeNotFound,
            expected_gas: 0,
        }),
        underflow1(@raw {
            bytecode: &[op::ADD],
            expected_return: InstructionResult::StackUnderflow,
            expected_gas: 3,
        }),
        underflow2(@raw {
            bytecode: &[op::PUSH0, op::ADD],
            expected_return: InstructionResult::StackUnderflow,
            expected_stack: &[U256::ZERO],
            expected_gas: 5,
        }),
        underflow3(@raw {
            bytecode: &[op::PUSH0, op::POP, op::ADD],
            expected_return: InstructionResult::StackUnderflow,
            expected_gas: 7,
        }),
        underflow4(@raw {
            bytecode: &[op::PUSH0, op::ADD, op::POP],
            expected_return: InstructionResult::StackUnderflow,
            expected_stack: &[U256::ZERO],
            expected_gas: 5,
        }),
        // LLVM is slow on this, but it passes.
        // overflow_not0(@raw {
        //     bytecode: &[op::PUSH0; 1023],
        //     expected_return: InstructionResult::Stop,
        //     expected_stack: &[0_U256; 1023],
        //     expected_gas: 2 * 1023,
        // }),
        overflow_not1(@raw {
            bytecode: &[op::PUSH0; 1024],
            expected_return: InstructionResult::Stop,
            expected_stack: &[0_U256; 1024],
            expected_gas: 2 * 1024,
        }),
        overflow0(@raw {
            bytecode: &[op::PUSH0; 1025],
            expected_return: InstructionResult::StackOverflow,
            expected_stack: &[0_U256; 1024],
            expected_gas: 2 * 1025,
        }),
        overflow1(@raw {
            bytecode: &[op::PUSH0; 1026],
            expected_return: InstructionResult::StackOverflow,
            expected_stack: &[0_U256; 1024],
            expected_gas: 2 * 1025,
        }),
    }

    spec_id {
        push0_merge(@raw {
            bytecode: &[op::PUSH0],
            spec_id: SpecId::MERGE,
            expected_return: InstructionResult::NotActivated,
            expected_gas: 0,
        }),
        push0_shanghai(@raw {
            bytecode: &[op::PUSH0],
            spec_id: SpecId::SHANGHAI,
            expected_stack: &[U256::ZERO],
            expected_gas: 2,
        }),
        push0_cancun(@raw {
            bytecode: &[op::PUSH0],
            spec_id: SpecId::CANCUN,
            expected_stack: &[U256::ZERO],
            expected_gas: 2,
        }),

        pre_eof_in_legacy(@raw {
            bytecode: &[op::PUSH0, op::PUSH0, op::SWAPN, 0],
            spec_id: SpecId::CANCUN,
            expected_return: InstructionResult::EOFOpcodeDisabledInLegacy,
            expected_stack: &[U256::ZERO, U256::ZERO],
            expected_gas: 2 + 2,
        }),
        eof_in_legacy(@raw {
            bytecode: &[op::PUSH0, op::PUSH0, op::SWAPN, 0],
            spec_id: SpecId::PRAGUE_EOF,
            expected_return: InstructionResult::EOFOpcodeDisabledInLegacy,
            expected_stack: &[U256::ZERO, U256::ZERO],
            expected_gas: 2 + 2,
        }),
        eof_in_legacy_returncontract(@raw {
            bytecode: &[op::PUSH0, op::PUSH0, op::RETURNCONTRACT, 0],
            expected_return: InstructionResult::ReturnContractInNotInitEOF,
            expected_stack: &[U256::ZERO, U256::ZERO],
            expected_gas: 2 + 2,
        }),
    }

    stack {
        pop(@raw {
            bytecode: &[op::PUSH1, 1, op::POP],
            expected_gas: 3 + 2,
        }),
        dup(@raw {
            bytecode: &[op::PUSH1, 1, op::DUP1],
            expected_stack: &[1_U256, 1_U256],
            expected_gas: 3 + 3,
        }),
        dupn(@raw {
            bytecode: &eof(&[op::PUSH1, 1, op::DUPN, 0, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[1_U256, 1_U256],
            expected_gas: 3 + 3,
        }),
        swap(@raw {
            bytecode: &[op::PUSH1, 1, op::PUSH1, 2, op::SWAP1],
            expected_stack: &[2_U256, 1_U256],
            expected_gas: 3 + 3 + 3,
        }),
        swapn(@raw {
            bytecode: &eof(&[op::PUSH1, 1, op::PUSH1, 2, op::SWAPN, 0, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[2_U256, 1_U256],
            expected_gas: 3 + 3 + 3,
        }),
        exchange(@raw {
            bytecode: &eof(&[op::PUSH1, 1, op::PUSH1, 2, op::PUSH1, 3, op::EXCHANGE, 0, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[2_U256, 1_U256, 3_U256],
            expected_gas: 3 + 3 + 3 + 3,
        }),
    }

    control_flow {
        basic_jump(@raw {
            bytecode: &[op::PUSH1, 3, op::JUMP, op::JUMPDEST, op::PUSH1, 69],
            expected_stack: &[69_U256],
            expected_gas: 3 + 8 + 1 + 3,
        }),
        unmodified_stack_after_push_jump(@raw {
            bytecode: &[op::PUSH1, 3, op::JUMP, op::JUMPDEST, op::PUSH0, op::ADD],
            expected_return: InstructionResult::StackUnderflow,
            expected_stack: &[U256::ZERO],
            expected_gas: 3 + 8 + 1 + 2 + 3,
        }),
        bad_jump(@raw {
            bytecode: &[op::JUMP],
            expected_return: InstructionResult::StackUnderflow,
            expected_gas: 8,
        }),
        bad_jumpi1(@raw {
            bytecode: &[op::JUMPI],
            expected_return: InstructionResult::StackUnderflow,
            expected_gas: 10,
        }),
        bad_jumpi2(@raw {
            bytecode: &[op::PUSH0, op::JUMPI],
            expected_return: InstructionResult::StackUnderflow,
            expected_stack: &[0_U256],
            expected_gas: 2 + 10,
        }),
        // TODO: Doesn't pass on aarch64 (???)
        // bad_jumpi3(@raw {
        //     bytecode: &[op::JUMPDEST, op::PUSH0, op::JUMPI],
        //     expected_return: InstructionResult::StackUnderflow,
        //     expected_stack: &[0_U256],
        //     expected_gas: 1 + 2 + 10,
        // }),

        basic_jumpi1(@raw {
            bytecode: &[op::JUMPDEST, op::PUSH0, op::PUSH0, op::JUMPI, op::PUSH1, 69],
            expected_stack: &[69_U256],
            expected_gas: 1 + 2 + 2 + 10 + 3,
        }),
        basic_jumpi1_lazy_invalid_target(@raw {
            bytecode: &[op::PUSH0, op::PUSH0, op::JUMPI, op::PUSH1, 69],
            expected_stack: &[69_U256],
            expected_gas: 2 + 2 + 10 + 3,
        }),
        basic_jumpi2(@raw {
            bytecode: &[op::PUSH1, 1, op::PUSH1, 5, op::JUMPI, op::JUMPDEST, op::PUSH1, 69],
            expected_stack: &[69_U256],
            expected_gas: 3 + 3 + 10 + 1 + 3,
        }),
        basic_jumpi2_lazy_invalid_target(@raw {
            bytecode: &[op::PUSH1, 1, op::PUSH0, op::JUMPI, op::PUSH1, 69],
            expected_return: InstructionResult::InvalidJump,
            expected_gas: 3 + 2 + 10,
        }),
        unmodified_stack_after_push_jumpi(@raw {
            bytecode: &[op::PUSH1, 1, op::PUSH1, 5, op::JUMPI, op::JUMPDEST, op::PUSH0, op::ADD],
            expected_return: InstructionResult::StackUnderflow,
            expected_stack: &[U256::ZERO],
            expected_gas: 3 + 3 + 10 + 1 + 2 + 3,
        }),

        basic_loop(@raw {
            bytecode: &[
                op::PUSH1, 3,  // i=3
                op::JUMPDEST,  // i
                op::PUSH1, 1,  // 1, i
                op::SWAP1,     // i, 1
                op::SUB,       // i-1
                op::DUP1,      // i-1, i-1
                op::PUSH1, 2,  // dst, i-1, i-1
                op::JUMPI,     // i=i-1
                op::POP,       //
                op::PUSH1, 69, // 69
            ],
            expected_stack: &[69_U256],
            expected_gas: 3 + (1 + 3 + 3 + 3 + 3 + 3 + 10) * 3 + 2 + 3,
        }),

        pc(@raw {
            bytecode: &[op::PC, op::PC, op::PUSH1, 69, op::PC, op::PUSH0, op::PC],
            expected_stack: &[0_U256, 1_U256, 69_U256, 4_U256, 0_U256, 6_U256],
            expected_gas: 2 + 2 + 3 + 2 + 2 + 2,
        }),

        rjump1(@raw {
            bytecode: &eof(&[op::RJUMP, 0x00, 0x00, op::PUSH1, 69, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[69_U256],
            expected_gas: 2 + 3,
        }),
        rjumpi1(@raw {
            bytecode: &eof(&[op::PUSH0, op::RJUMPI, 0x00, 0x03, op::PUSH1, 69, op::STOP, op::PUSH1, 42, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[69_U256],
            expected_gas: 2 + 4 + 3,
        }),
        rjumpi2(@raw {
            bytecode: &eof(&[op::PUSH1, 1, op::RJUMPI, 0x00, 0x03, op::PUSH1, 69, op::STOP, op::PUSH1, 42, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[42_U256],
            expected_gas: 3 + 4 + 3,
        }),
        rjumpv1(@raw {
            bytecode: &rjumpv_code::<0>(),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[20_U256],
            expected_gas: 10,
        }),
        rjumpv2(@raw {
            bytecode: &rjumpv_code::<1>(),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[30_U256],
            expected_gas: 10,
        }),
        rjumpv3(@raw {
            bytecode: &rjumpv_code::<2>(),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[40_U256],
            expected_gas: 10,
        }),
        rjumpv4(@raw {
            bytecode: &rjumpv_code::<3>(),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[10_U256],
            expected_gas: 10,
        }),
        rjumpv5(@raw {
            bytecode: &rjumpv_code::<69>(),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[10_U256],
            expected_gas: 10,
        }),
        rjumpv_overflow(@raw {
            // RJUMPV 0x0200030000fff6
            bytecode: &eof(&hex!("6000e20200030000fff65b5b0061201560015500")),
            spec_id: SpecId::PRAGUE_EOF,
            expected_gas: 113,
        }),
    }

    subroutines {
        callf(@raw {
            bytecode: &eof_sections(&[
                &[op::CALLF, 0x00, 0x01, op::PUSH1, 1, op::STOP],
                &[op::CALLF, 0x00, 0x02, op::PUSH1, 2, op::RETF],
                &[                       op::PUSH1, 3, op::RETF],
            ]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[3_U256, 2_U256, 1_U256],
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
        }),
        jumpf1(@raw {
            bytecode: &eof_sections(&[
                &[op::CALLF, 0x00, 0x01, op::PUSH1, 1, op::STOP],
                &[op::JUMPF, 0x00, 0x02, op::PUSH1, 2, op::RETF],
                &[                       op::PUSH1, 3, op::RETF],
            ]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[3_U256, 1_U256],
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
        }),
        jumpf2(@raw {
            bytecode: &eof_sections_unchecked(&[
                &[op::PUSH1, 1, op::JUMPF, 0x00, 0x01],
                &[op::PUSH1, 2, op::STOP],
            ]).raw,
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[1_U256, 2_U256],
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
        }),
    }

    arith {
        add1(op::ADD, 0_U256, 0_U256 => 0_U256),
        add2(op::ADD, 1_U256, 2_U256 => 3_U256),
        add3(op::ADD, 255_U256, 255_U256 => 510_U256),
        add4(op::ADD, U256::MAX, 1_U256 => 0_U256),
        add5(op::ADD, U256::MAX, 2_U256 => 1_U256),

        sub1(op::SUB, 3_U256, 2_U256 => 1_U256),
        sub2(op::SUB, 1_U256, 2_U256 => -1_U256),
        sub3(op::SUB, 1_U256, 3_U256 => (-1_U256).wrapping_sub(1_U256)),
        sub4(op::SUB, 255_U256, 255_U256 => 0_U256),

        mul1(op::MUL, 1_U256, 2_U256 => 2_U256),
        mul2(op::MUL, 32_U256, 32_U256 => 1024_U256),
        mul3(op::MUL, U256::MAX, 2_U256 => U256::MAX.wrapping_sub(1_U256)),

        div1(op::DIV, 32_U256, 32_U256 => 1_U256),
        div2(op::DIV, 1_U256, 2_U256 => 0_U256),
        div3(op::DIV, 2_U256, 2_U256 => 1_U256),
        div4(op::DIV, 3_U256, 2_U256 => 1_U256),
        div5(op::DIV, 4_U256, 2_U256 => 2_U256),
        div_by_zero1(op::DIV, 0_U256, 0_U256 => 0_U256),
        div_by_zero2(op::DIV, 32_U256, 0_U256 => 0_U256),

        rem1(op::MOD, 32_U256, 32_U256 => 0_U256),
        rem2(op::MOD, 1_U256, 2_U256 => 1_U256),
        rem3(op::MOD, 2_U256, 2_U256 => 0_U256),
        rem4(op::MOD, 3_U256, 2_U256 => 1_U256),
        rem5(op::MOD, 4_U256, 2_U256 => 0_U256),
        rem_by_zero1(op::MOD, 0_U256, 0_U256 => 0_U256),
        rem_by_zero2(op::MOD, 32_U256, 0_U256 => 0_U256),

        sdiv1(op::SDIV, 32_U256, 32_U256 => 1_U256),
        sdiv2(op::SDIV, 1_U256, 2_U256 => 0_U256),
        sdiv3(op::SDIV, 2_U256, 2_U256 => 1_U256),
        sdiv4(op::SDIV, 3_U256, 2_U256 => 1_U256),
        sdiv5(op::SDIV, 4_U256, 2_U256 => 2_U256),
        sdiv_by_zero1(op::SDIV, 0_U256, 0_U256 => 0_U256),
        sdiv_by_zero2(op::SDIV, 32_U256, 0_U256 => 0_U256),
        sdiv_min_by_1(op::SDIV, I256_MIN, 1_U256 => -I256_MIN),
        sdiv_min_by_minus_1(op::SDIV, I256_MIN, -1_U256 => I256_MIN),
        sdiv_max1(op::SDIV, I256_MAX, 1_U256 => I256_MAX),
        sdiv_max2(op::SDIV, I256_MAX, -1_U256 => -I256_MAX),

        srem1(op::SMOD, 32_U256, 32_U256 => 0_U256),
        srem2(op::SMOD, 1_U256, 2_U256 => 1_U256),
        srem3(op::SMOD, 2_U256, 2_U256 => 0_U256),
        srem4(op::SMOD, 3_U256, 2_U256 => 1_U256),
        srem5(op::SMOD, 4_U256, 2_U256 => 0_U256),
        srem_by_zero1(op::SMOD, 0_U256, 0_U256 => 0_U256),
        srem_by_zero2(op::SMOD, 32_U256, 0_U256 => 0_U256),

        addmod1(op::ADDMOD, 1_U256, 2_U256, 3_U256 => 0_U256),
        addmod2(op::ADDMOD, 1_U256, 2_U256, 4_U256 => 3_U256),
        addmod3(op::ADDMOD, 1_U256, 2_U256, 2_U256 => 1_U256),
        addmod4(op::ADDMOD, 32_U256, 32_U256, 69_U256 => 64_U256),

        mulmod1(op::MULMOD, 0_U256, 0_U256, 1_U256 => 0_U256),
        mulmod2(op::MULMOD, 69_U256, 0_U256, 1_U256 => 0_U256),
        mulmod3(op::MULMOD, 0_U256, 1_U256, 2_U256 => 0_U256),
        mulmod4(op::MULMOD, 69_U256, 1_U256, 2_U256 => 1_U256),
        mulmod5(op::MULMOD, 69_U256, 1_U256, 30_U256 => 9_U256),
        mulmod6(op::MULMOD, 69_U256, 2_U256, 100_U256 => 38_U256),

        exp1(op::EXP, 0_U256, 0_U256 => 1_U256; op_gas(10)),
        exp2(op::EXP, 2_U256, 0_U256 => 1_U256; op_gas(10)),
        exp3(op::EXP, 2_U256, 1_U256 => 2_U256; op_gas(60)),
        exp4(op::EXP, 2_U256, 2_U256 => 4_U256; op_gas(60)),
        exp5(op::EXP, 2_U256, 3_U256 => 8_U256; op_gas(60)),
        exp6(op::EXP, 2_U256, 4_U256 => 16_U256; op_gas(60)),
        exp_overflow(op::EXP, 2_U256, 256_U256 => 0_U256; op_gas(110)),

        signextend1(op::SIGNEXTEND, 0_U256, 0_U256 => 0_U256),
        signextend2(op::SIGNEXTEND, 1_U256, 0_U256 => 0_U256),
        signextend3(op::SIGNEXTEND, 0_U256, -1_U256 => -1_U256),
        signextend4(op::SIGNEXTEND, 1_U256, -1_U256 => -1_U256),
        signextend5(op::SIGNEXTEND, 0_U256, 0x7f_U256 => 0x7f_U256),
        signextend6(op::SIGNEXTEND, 0_U256, 0x80_U256 => -0x80_U256),
        signextend7(op::SIGNEXTEND, 0_U256, 0xff_U256 => U256::MAX),
        signextend8(op::SIGNEXTEND, 1_U256, 0x7fff_U256 => 0x7fff_U256),
        signextend8_extra(op::SIGNEXTEND, 1_U256, 0xff7fff_U256 => 0x7fff_U256),
        signextend9(op::SIGNEXTEND, 1_U256, 0x8000_U256 => -0x8000_U256),
        signextend9_extra(op::SIGNEXTEND, 1_U256, 0x118000_U256 => -0x8000_U256),
        signextend10(op::SIGNEXTEND, 1_U256, 0xffff_U256 => U256::MAX),
    }

    cmp {
        lt1(op::LT, 1_U256, 2_U256 => 1_U256),
        lt2(op::LT, 2_U256, 1_U256 => 0_U256),
        lt3(op::LT, 1_U256, 1_U256 => 0_U256),
        lt4(op::LT, -1_U256, 1_U256 => 0_U256),

        gt1(op::GT, 1_U256, 2_U256 => 0_U256),
        gt2(op::GT, 2_U256, 1_U256 => 1_U256),
        gt3(op::GT, 1_U256, 1_U256 => 0_U256),
        gt4(op::GT, -1_U256, 1_U256 => 1_U256),

        slt1(op::SLT, 1_U256, 2_U256 => 1_U256),
        slt2(op::SLT, 2_U256, 1_U256 => 0_U256),
        slt3(op::SLT, 1_U256, 1_U256 => 0_U256),
        slt4(op::SLT, -1_U256, 1_U256 => 1_U256),

        sgt1(op::SGT, 1_U256, 2_U256 => 0_U256),
        sgt2(op::SGT, 2_U256, 1_U256 => 1_U256),
        sgt3(op::SGT, 1_U256, 1_U256 => 0_U256),
        sgt4(op::SGT, -1_U256, 1_U256 => 0_U256),

        eq1(op::EQ, 1_U256, 2_U256 => 0_U256),
        eq2(op::EQ, 2_U256, 1_U256 => 0_U256),
        eq3(op::EQ, 1_U256, 1_U256 => 1_U256),

        iszero1(op::ISZERO, 0_U256 => 1_U256),
        iszero2(op::ISZERO, 1_U256 => 0_U256),
        iszero3(op::ISZERO, 2_U256 => 0_U256),
    }

    bitwise {
        and1(op::AND, 0_U256, 0_U256 => 0_U256),
        and2(op::AND, 1_U256, 1_U256 => 1_U256),
        and3(op::AND, 1_U256, 2_U256 => 0_U256),
        and4(op::AND, 255_U256, 255_U256 => 255_U256),

        or1(op::OR, 0_U256, 0_U256 => 0_U256),
        or2(op::OR, 1_U256, 2_U256 => 3_U256),
        or3(op::OR, 1_U256, 3_U256 => 3_U256),
        or4(op::OR, 2_U256, 2_U256 => 2_U256),

        xor1(op::XOR, 0_U256, 0_U256 => 0_U256),
        xor2(op::XOR, 1_U256, 2_U256 => 3_U256),
        xor3(op::XOR, 1_U256, 3_U256 => 2_U256),
        xor4(op::XOR, 2_U256, 2_U256 => 0_U256),

        not1(op::NOT, 0_U256 => U256::MAX),
        not2(op::NOT, U256::MAX => 0_U256),
        not3(op::NOT, 1_U256 => U256::MAX.wrapping_sub(1_U256)),

        byte1(op::BYTE, 0_U256, 0x1122334400000000000000000000000000000000000000000000000000000000_U256 => 0x11_U256),
        byte2(op::BYTE, 1_U256, 0x1122334400000000000000000000000000000000000000000000000000000000_U256 => 0x22_U256),
        byte3(op::BYTE, 2_U256, 0x1122334400000000000000000000000000000000000000000000000000000000_U256 => 0x33_U256),
        byte4(op::BYTE, 3_U256, 0x1122334400000000000000000000000000000000000000000000000000000000_U256 => 0x44_U256),
        byte5(op::BYTE, 4_U256, 0x1122334400000000000000000000000000000000000000000000000000000000_U256 => 0x00_U256),
        byte_oob0(op::BYTE, 31_U256, U256::MAX => 0xFF_U256),
        byte_oob1(op::BYTE, 32_U256, U256::MAX => 0_U256),
        byte_oob2(op::BYTE, 33_U256, U256::MAX => 0_U256),

        // shift operand order is reversed for some reason:
        // shift, x
        shl1(op::SHL, 0_U256, 1_U256 => 1_U256),
        shl2(op::SHL, 1_U256, 1_U256 => 2_U256),
        shl3(op::SHL, 2_U256, 1_U256 => 4_U256),
        shl4(op::SHL, 255_U256, -1_U256 => -1_U256 << 255),
        shl5(op::SHL, 256_U256, -1_U256 => 0_U256),

        shr1(op::SHR, 0_U256, 1_U256 => 1_U256),
        shr2(op::SHR, 1_U256, 2_U256 => 1_U256),
        shr3(op::SHR, 2_U256, 4_U256 => 1_U256),
        shr4(op::SHR, 255_U256, -1_U256 => 1_U256),
        shr5(op::SHR, 256_U256, -1_U256 => 0_U256),

        sar1(op::SAR, 0_U256, 1_U256 => 1_U256),
        sar2(op::SAR, 1_U256, 2_U256 => 1_U256),
        sar3(op::SAR, 2_U256, 4_U256 => 1_U256),
        sar4(op::SAR, 1_U256, -1_U256 => -1_U256),
        sar5(op::SAR, 2_U256, -1_U256 => -1_U256),
        sar6(op::SAR, 255_U256, -1_U256 => -1_U256),
        sar7(op::SAR, 256_U256, -1_U256 => -1_U256),
    }

    system {
        gas0(@raw {
            bytecode: &[op::GAS, op::GAS, op::JUMPDEST, op::GAS],
            expected_stack: &[DEF_GAS_LIMIT_U256 - 2_U256, DEF_GAS_LIMIT_U256 - 4_U256, DEF_GAS_LIMIT_U256 - 7_U256],
            expected_gas: 2 + 2 + 1 + 2,
        }),
        keccak256_empty1(@raw {
            bytecode: &[op::PUSH0, op::PUSH0, op::KECCAK256],
            expected_stack: &[KECCAK_EMPTY.into()],
            expected_gas: 2 + 2 + gas::KECCAK256,
        }),
        keccak256_empty2(@raw {
            bytecode: &[op::PUSH0, op::PUSH1, 32, op::KECCAK256],
            expected_stack: &[KECCAK_EMPTY.into()],
            expected_gas: 2 + 3 + gas::KECCAK256,
        }),
        keccak256_1(@raw {
            bytecode: &[op::PUSH1, 32, op::PUSH0, op::KECCAK256],
            expected_stack: &[keccak256([0; 32]).into()],
            expected_memory: &[0; 32],
            expected_gas: 3 + 2 + (gas::keccak256_cost(32).unwrap() + 3),
        }),
        keccak256_2(@raw {
            bytecode: &[op::PUSH2, 0x69, 0x42, op::PUSH0, op::MSTORE, op::PUSH1, 0x20, op::PUSH0, op::KECCAK256],
            expected_stack: &[keccak256(0x6942_U256.to_be_bytes::<32>()).into()],
            expected_memory: &0x6942_U256.to_be_bytes::<32>(),
            expected_gas: 3 + 2 + (3 + 3) + 3 + 2 + gas::keccak256_cost(32).unwrap(),
        }),

        address(@raw {
            bytecode: &[op::ADDRESS, op::ADDRESS],
            expected_stack: &[DEF_ADDR.into_word().into(), DEF_ADDR.into_word().into()],
            expected_gas: 4,
        }),
        origin(@raw {
            bytecode: &[op::ORIGIN, op::ORIGIN],
            expected_stack: &[def_env().tx.caller.into_word().into(), def_env().tx.caller.into_word().into()],
            expected_gas: 4,
        }),
        caller(@raw {
            bytecode: &[op::CALLER, op::CALLER],
            expected_stack: &[DEF_CALLER.into_word().into(), DEF_CALLER.into_word().into()],
            expected_gas: 4,
        }),
        callvalue(@raw {
            bytecode: &[op::CALLVALUE, op::CALLVALUE],
            expected_stack: &[DEF_VALUE, DEF_VALUE],
            expected_gas: 4,
        }),
    }

    calldata {
        calldataload1(@raw {
            bytecode: &[op::PUSH0, op::CALLDATALOAD],
            expected_stack: &[U256::from_be_slice(&DEF_CD[..32])],
            expected_gas: 2 + 3,
        }),
        calldataload2(@raw {
            bytecode: &[op::PUSH1, 63, op::CALLDATALOAD],
            expected_stack: &[0xaa00000000000000000000000000000000000000000000000000000000000000_U256],
            expected_gas: 3 + 3,
        }),
        calldatasize(@raw {
            bytecode: &[op::CALLDATASIZE, op::CALLDATASIZE],
            expected_stack: &[U256::from(DEF_CD.len()), U256::from(DEF_CD.len())],
            expected_gas: 2 + 2,
        }),
        calldatacopy(@raw {
            bytecode: &[op::PUSH1, 32, op::PUSH0, op::PUSH0, op::CALLDATACOPY],
            expected_memory: &DEF_CD[..32],
            expected_gas: 3 + 2 + 2 + (gas::verylowcopy_cost(32).unwrap() + 3),
        }),
    }

    code {
        codesize(@raw {
            bytecode: &[op::CODESIZE, op::CODESIZE],
            expected_stack: &[2_U256, 2_U256],
            expected_gas: 2 + 2,
        }),
        codecopy(@raw {
            bytecode: &[op::PUSH1, 5, op::PUSH0, op::PUSH0, op::CODECOPY],
            expected_memory: &hex!("60055f5f39000000000000000000000000000000000000000000000000000000"),
            expected_gas: 3 + 2 + 2 + (gas::verylowcopy_cost(32).unwrap() + gas::memory_gas(1)),
        }),
    }

    returndata {
        returndataload1(@raw {
            bytecode: &eof(&[op::PUSH0, op::RETURNDATALOAD, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[U256::from_be_slice(&DEF_RD[..32])],
            expected_gas: 2 + 3,
        }),
        returndataload2(@raw {
            bytecode: &eof(&[op::PUSH1, 63, op::RETURNDATALOAD, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[0xbb00000000000000000000000000000000000000000000000000000000000000_U256],
            expected_gas: 3 + 3,
        }),
        returndatasize(@raw {
            bytecode: &[op::RETURNDATASIZE, op::RETURNDATASIZE],
            expected_stack: &[64_U256, 64_U256],
            expected_gas: 2 + 2,
        }),
        returndatacopy(@raw {
            bytecode: &[op::PUSH1, 32, op::PUSH0, op::PUSH0, op::RETURNDATACOPY],
            expected_memory: &DEF_RD[..32],
            expected_gas: 3 + 2 + 2 + (gas::verylowcopy_cost(32).unwrap() + gas::memory_gas(1)),
        }),
    }

    data {
        dataload1(@raw {
            bytecode: &eof(&[op::PUSH0, op::DATALOAD, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[U256::from_be_slice(&DEF_DATA[..32])],
            expected_gas: 2 + 4,
        }),
        dataload2(@raw {
            bytecode: &eof(&[op::PUSH1, 63, op::DATALOAD, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[0xcc00000000000000000000000000000000000000000000000000000000000000_U256],
            expected_gas: 3 + 4,
        }),
        dataloadn1(@raw {
            bytecode: &eof(&[op::DATALOADN, 0x00, 0x00, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[U256::from_be_slice(&DEF_DATA[..32])],
            expected_gas: 3,
        }),
        // TODO: Revm does not allow out-of-bounds `DATALOADN`. Is this right?
        // dataloadn2(@raw {
        //     bytecode: &eof(&[op::DATALOADN, 0x00, 63, op::STOP]),
        //     spec_id: SpecId::PRAGUE_EOF,
        //     expected_stack: &[0xcc00000000000000000000000000000000000000000000000000000000000000_U256],
        //     expected_gas: 3,
        // }),
        datasize(@raw {
            bytecode: &eof(&[op::DATASIZE, op::DATASIZE, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_stack: &[U256::from(DEF_DATA.len()), U256::from(DEF_DATA.len())],
            expected_gas: 4,
        }),
        datacopy(@raw {
            bytecode: &eof(&[op::PUSH1, 32, op::PUSH0, op::PUSH0, op::DATACOPY, op::STOP]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_memory: &DEF_DATA[..32],
            expected_gas: 3 + 2 + 2 + (gas::verylowcopy_cost(32).unwrap() + gas::memory_gas(1)),
        }),
    }

    extcode {
        extcodesize1(op::EXTCODESIZE, DEF_ADDR.into_word().into() => 0_U256;
            op_gas(100)),
        extcodesize2(op::EXTCODESIZE, OTHER_ADDR.into_word().into() => U256::from(def_codemap()[&OTHER_ADDR].len());
            op_gas(100)),
        extcodecopy1(@raw {
            bytecode: &[op::PUSH0, op::PUSH0, op::PUSH0, op::PUSH0, op::EXTCODECOPY],
            expected_memory: &[],
            expected_gas: 2 + 2 + 2 + 2 + 100,
        }),
        extcodecopy2(@raw {
            // bytecode: &[op::PUSH1, 64, op::PUSH0, op::PUSH0, op::PUSH20, OTHER_ADDR, op::EXTCODECOPY],
            bytecode: &hex!("6040 5f 5f 736969696969696969696969696969696969696969 3c"),
            expected_memory: &{
                let mut mem = [0; 64];
                let code = def_codemap()[&OTHER_ADDR].original_bytes();
                mem[..code.len()].copy_from_slice(&code);
                mem
            },
            expected_gas: 3 + 2 + 2 + 3 + (100 + 12),
        }),
        extcodehash1(op::EXTCODEHASH, DEF_ADDR.into_word().into() => KECCAK_EMPTY.into();
            op_gas(100)),
        extcodehash2(op::EXTCODEHASH, OTHER_ADDR.into_word().into() => def_codemap()[&OTHER_ADDR].hash_slow().into();
            op_gas(100)),
    }

    env {
        gas_price(@raw {
            bytecode: &[op::GASPRICE],
            expected_stack: &[def_env().effective_gas_price()],
            expected_gas: 2,
        }),
        // Host determines blockhash
        blockhash0(op::BLOCKHASH, DEF_BN - 0_U256 => DEF_BN - 0_U256),
        blockhash1(op::BLOCKHASH, DEF_BN - 1_U256 => DEF_BN - 1_U256),
        blockhash2(op::BLOCKHASH, DEF_BN - 255_U256 => DEF_BN - 255_U256),
        blockhash3(op::BLOCKHASH, DEF_BN - 256_U256 => DEF_BN - 256_U256),
        blockhash4(op::BLOCKHASH, DEF_BN - 257_U256 => DEF_BN - 257_U256),
        coinbase(@raw {
            bytecode: &[op::COINBASE, op::COINBASE],
            expected_stack: &[def_env().block.coinbase.into_word().into(), def_env().block.coinbase.into_word().into()],
            expected_gas: 4,
        }),
        timestamp(@raw {
            bytecode: &[op::TIMESTAMP, op::TIMESTAMP],
            expected_stack: &[def_env().block.timestamp, def_env().block.timestamp],
            expected_gas: 4,
        }),
        number(@raw {
            bytecode: &[op::NUMBER, op::NUMBER],
            expected_stack: &[def_env().block.number, def_env().block.number],
            expected_gas: 4,
        }),
        difficulty(@raw {
            bytecode: &[op::DIFFICULTY, op::DIFFICULTY],
            spec_id: SpecId::GRAY_GLACIER,
            expected_stack: &[def_env().block.difficulty, def_env().block.difficulty],
            expected_gas: 4,
        }),
        difficulty_prevrandao(@raw {
            bytecode: &[op::DIFFICULTY, op::DIFFICULTY],
            spec_id: SpecId::MERGE,
            expected_stack: &[def_env().block.prevrandao.unwrap().into(), def_env().block.prevrandao.unwrap().into()],
            expected_gas: 4,
        }),
        gaslimit(@raw {
            bytecode: &[op::GASLIMIT, op::GASLIMIT],
            expected_stack: &[def_env().block.gas_limit, def_env().block.gas_limit],
            expected_gas: 4,
        }),
        chainid(@raw {
            bytecode: &[op::CHAINID, op::CHAINID],
            expected_stack: &[U256::from(def_env().cfg.chain_id), U256::from(def_env().cfg.chain_id)],
            expected_gas: 4,
        }),
        selfbalance(@raw {
            bytecode: &[op::SELFBALANCE, op::SELFBALANCE],
            expected_stack: &[0xba_U256, 0xba_U256],
            expected_gas: 10,
        }),
        basefee(@raw {
            bytecode: &[op::BASEFEE, op::BASEFEE],
            expected_stack: &[def_env().block.basefee, def_env().block.basefee],
            expected_gas: 4,
        }),
        blobhash0(@raw {
            bytecode: &[op::PUSH0, op::BLOBHASH],
            expected_stack: &[def_env().tx.blob_hashes[0].into()],
            expected_gas: 2 + 3,
        }),
        blobhash1(@raw {
            bytecode: &[op::PUSH1, 1, op::BLOBHASH],
            expected_stack: &[def_env().tx.blob_hashes[1].into()],
            expected_gas: 3 + 3,
        }),
        blobhash2(@raw {
            bytecode: &[op::PUSH1, 2, op::BLOBHASH],
            expected_stack: &[0_U256],
            expected_gas: 3 + 3,
        }),
        blobbasefee(@raw {
            bytecode: &[op::BLOBBASEFEE, op::BLOBBASEFEE],
            expected_stack: &[U256::from(def_env().block.get_blob_gasprice().unwrap()), U256::from(def_env().block.get_blob_gasprice().unwrap())],
            expected_gas: 4,
        }),
    }

    memory {
        mload1(@raw {
            bytecode: &[op::PUSH0, op::MLOAD],
            expected_stack: &[0_U256],
            expected_memory: &[0; 32],
            expected_gas: 2 + (3 + gas::memory_gas(1)),
        }),
        mload2(@raw {
            bytecode: &[op::PUSH1, 1, op::MLOAD],
            expected_stack: &[0_U256],
            expected_memory: &[0; 64],
            expected_gas: 3 + (3 + gas::memory_gas(2)),
        }),
        mload3(@raw {
            bytecode: &[op::PUSH1, 32, op::MLOAD],
            expected_stack: &[0_U256],
            expected_memory: &[0; 64],
            expected_gas: 3 + (3 + gas::memory_gas(2)),
        }),
        mload4(@raw {
            bytecode: &[op::PUSH1, 33, op::MLOAD],
            expected_stack: &[0_U256],
            expected_memory: &[0; 96],
            expected_gas: 3 + (3 + gas::memory_gas(3)),
        }),
        mload_overflow1(@raw {
            bytecode: &[op::PUSH8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, op::MLOAD],
            expected_return: InstructionResult::MemoryOOG,
            expected_stack: &[U256::from(u64::MAX)],
            expected_gas: 3 + 3,
        }),
        mload_overflow2(@raw {
            bytecode: &[op::PUSH8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff - 1, op::MLOAD],
            expected_return: InstructionResult::MemoryOOG,
            expected_stack: &[U256::from(u64::MAX - 1)],
            expected_gas: 3 + 3,
        }),
        mload_overflow3(@raw {
            bytecode: &[op::PUSH8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff - 31, op::MLOAD],
            expected_return: InstructionResult::MemoryOOG,
            expected_stack: &[U256::from(u64::MAX - 31)],
            expected_gas: 3 + 3,
        }),
        mload_overflow4(@raw {
            bytecode: &[op::PUSH8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff - 32, op::MLOAD],
            expected_return: InstructionResult::MemoryOOG,
            expected_stack: &[U256::from(u64::MAX - 32)],
            expected_gas: 3 + 3,
        }),
        mload_overflow5(@raw {
            bytecode: &[op::PUSH8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff - 33, op::MLOAD],
            expected_return: InstructionResult::MemoryOOG,
            expected_stack: &[U256::from(u64::MAX - 33)],
            expected_gas: 3 + 3,
        }),
        mload_overflow6(@raw {
            bytecode: &[op::ADDRESS, op::MLOAD],
            expected_return: InstructionResult::InvalidOperandOOG,
            expected_stack: &[DEF_ADDR.into_word().into()],
            expected_gas: 5,
        }),
        mstore1(@raw {
            bytecode: &[op::PUSH0, op::PUSH0, op::MSTORE],
            expected_memory: &[0; 32],
            expected_gas: 2 + 2 + (3 + gas::memory_gas(1)),
        }),
        mstore8_1(@raw {
            bytecode: &[op::PUSH0, op::PUSH0, op::MSTORE8],
            expected_memory: &[0; 32],
            expected_gas: 2 + 2 + (3 + gas::memory_gas(1)),
        }),
        mstore8_2(@raw {
            bytecode: &[op::PUSH2, 0x69, 0x69, op::PUSH0, op::MSTORE8],
            expected_memory: &{
                let mut mem = [0; 32];
                mem[0] = 0x69;
                mem
            },
            expected_gas: 3 + 2 + (3 + gas::memory_gas(1)),
        }),
        msize1(@raw {
            bytecode: &[op::MSIZE, op::MSIZE],
            expected_stack: &[0_U256, 0_U256],
            expected_gas: 2 + 2,
        }),
        msize2(@raw {
            bytecode: &[op::MSIZE, op::PUSH0, op::MLOAD, op::POP, op::MSIZE, op::PUSH1, 1, op::MLOAD, op::POP, op::MSIZE],
            expected_stack: &[0_U256, 32_U256, 64_U256],
            expected_memory: &[0; 64],
            expected_gas: 2 + 2 + (3 + gas::memory_gas(1)) + 2 + 2 + 3 + (3 + (gas::memory_gas(2) - gas::memory_gas(1))) + 2 + 2,
        }),
        mcopy1(@raw {
            bytecode: &[op::PUSH1, 32, op::PUSH0, op::PUSH1, 32, op::MCOPY],
            expected_memory: &[0; 64],
            expected_gas: 3 + 2 + 3 + (gas::verylowcopy_cost(32).unwrap() + gas::memory_gas(2)),
        }),
        mcopy2(@raw {
            bytecode: &[op::PUSH2, 0x42, 0x69, op::PUSH0, op::MSTORE,
                        op::PUSH1, 2, op::PUSH1, 30, op::PUSH1, 1, op::MCOPY],
            expected_memory: &{
                let mut mem = [0; 32];
                mem[30] = 0x42;
                mem[31] = 0x69;
                mem[1] = 0x42;
                mem[2] = 0x69;
                mem
            },
            expected_gas: 3 + 2 + (3 + gas::memory_gas(1)) +
                          3 + 3 + 3 + gas::verylowcopy_cost(2).unwrap(),
        }),
    }

    host {
        balance(op::BALANCE, 0_U256 => 0_U256; op_gas(100)),
        sload1(@raw {
            bytecode: &[op::PUSH1, 69, op::SLOAD],
            expected_stack: &[42_U256],
            expected_gas: 3 + 100,
        }),
        sload2(@raw {
            bytecode: &[op::PUSH1, 70, op::SLOAD],
            expected_stack: &[0_U256],
            expected_gas: 3 + 2100,
        }),
        sload3(@raw {
            bytecode: &[op::PUSH1, 0xff, op::SLOAD],
            expected_stack: &[0_U256],
            expected_gas: 3 + 2100,
        }),
        sstore1(@raw {
            bytecode: &[op::PUSH1, 200, op::SLOAD, op::PUSH1, 100, op::PUSH1, 200, op::SSTORE, op::PUSH1, 200, op::SLOAD],
            expected_stack: &[0_U256, 100_U256],
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            assert_host: Some(|host| {
                assert_eq!(host.storage.get(&200_U256), Some(&100_U256));
            }),
        }),
        tload(@raw {
            bytecode: &[op::PUSH1, 69, op::TLOAD],
            expected_stack: &[0_U256],
            expected_gas: 3 + 100,
            assert_host: Some(|host| {
                assert!(host.transient_storage.is_empty());
            }),
        }),
        tstore(@raw {
            bytecode: &[op::PUSH1, 69, op::TLOAD, op::PUSH1, 42, op::PUSH1, 69, op::TSTORE, op::PUSH1, 69, op::TLOAD],
            expected_stack: &[0_U256, 42_U256],
            expected_gas: 3 + 100 + 3 + 3 + 100 + 3 + 100,
            assert_host: Some(|host| {
                assert_eq!(host.transient_storage.get(&69_U256), Some(&42_U256));
            }),
        }),
        log0(@raw {
            bytecode: &[op::PUSH0, op::PUSH0, op::LOG0],
            expected_gas: 2 + 2 + gas::log_cost(0, 0).unwrap(),
            assert_host: Some(|host| {
                assert_eq!(host.log, [primitives::Log {
                    address: DEF_ADDR,
                    data: LogData::new(vec![], Bytes::new()).unwrap(),
                }]);
            }),
        }),
        log0_data(@raw {
            bytecode: &[op::PUSH2, 0x69, 0x42, op::PUSH0, op::MSTORE, op::PUSH1, 32, op::PUSH0, op::LOG0],
            expected_memory: &0x6942_U256.to_be_bytes::<32>(),
            expected_gas: 3 + 2 + (3 + gas::memory_gas(1)) + 3 + 2 + gas::log_cost(0, 32).unwrap(),
            assert_host: Some(|host| {
                assert_eq!(host.log, [primitives::Log {
                    address: DEF_ADDR,
                    data: LogData::new(vec![], Bytes::copy_from_slice(&0x6942_U256.to_be_bytes::<32>())).unwrap(),
                }]);
            }),
        }),
        log1_1(@raw {
            bytecode: &[op::PUSH0, op::PUSH0, op::PUSH0, op::LOG1],
            expected_gas: 2 + 2 + 2 + gas::log_cost(1, 0).unwrap(),
            assert_host: Some(|host| {
                assert_eq!(host.log, [primitives::Log {
                    address: DEF_ADDR,
                    data: LogData::new(vec![B256::ZERO], Bytes::new()).unwrap(),
                }]);
            }),
        }),
        log1_2(@raw {
            bytecode: &hex!(
                "7f000000000000000000000000ffffffffffffffffffffffffffffffffffffffff"
                "7f0000000000000000000000000000000000000000000000000000000000000032" // 50
                "59"
                "a1"
            ),
            expected_memory: &[0; 64],
            expected_gas: 3 + 3 + 2 + (gas::log_cost(1, 50).unwrap() + gas::memory_gas(2)),
            assert_host: Some(|host| {
                assert_eq!(host.log, [primitives::Log {
                    address: DEF_ADDR,
                    data: LogData::new(
                        vec![0xffffffffffffffffffffffffffffffffffffffff_U256.into()],
                        Bytes::copy_from_slice(&[0; 50]),
                    ).unwrap(),
                }]);
            }),
        }),
        eofcreate(@raw {
            bytecode: &eof(&[
                op::PUSH1, 0x69, op::PUSH0, op::MSTORE,
                op::PUSH1, 32, op::PUSH0, op::PUSH1, 0x70, op::PUSH1, 0x42,
                op::EOFCREATE, 0x00,
                op::STOP,
            ]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_return: InstructionResult::CallOrCreate,
            expected_memory: &0x69_U256.to_be_bytes::<32>(),
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_next_action: InterpreterAction::EOFCreate {
                inputs: Box::new(revm_interpreter::EOFCreateInputs {
                    caller: DEF_ADDR,
                    value: 0x42_U256,
                    gas_limit: 66899,
                    kind: revm_interpreter::EOFCreateKind::Opcode {
                        initcode: primitives::Eof::decode(eof_subcontainer()).unwrap(),
                        input: 0x69_U256.to_be_bytes::<32>().into(),
                        created_address: DEF_ADDR.create2_from_code(0x70_U256.to_be_bytes::<32>(), eof_subcontainer()),
                    },
                }),
            },
        }),
        returncontract(@raw {
            bytecode: &eof(&[op::PUSH1, 32, op::PUSH0, op::RETURNCONTRACT, 0x00]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_return: InstructionResult::ReturnContract,
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_memory: &[0; 32],
            expected_next_action: InterpreterAction::Return {
                result: InterpreterResult {
                    gas: {
                        let mut gas = Gas::new(DEF_GAS_LIMIT);
                        let _ = gas.record_cost(8);
                        gas
                    },
                    result: InstructionResult::ReturnContract,
                    output: [&{
                        // ef00010100040200010001040040000080000000
                        let mut sub = eof_subcontainer().to_vec();
                        sub[13] += 32;
                        sub
                    }[..], &[0u8; 32][..]].concat().into(),
                }
            },
            modify_ecx: Some(|ecx| {
                ecx.is_eof_init = true;
            }),
        }),
        create(@raw {
            bytecode: &[op::PUSH1, 0x69, op::PUSH0, op::MSTORE, op::PUSH1, 32, op::PUSH0, op::PUSH1, 0x42, op::CREATE],
            expected_return: InstructionResult::CallOrCreate,
            // NOTE: The address is pushed by the caller.
            expected_stack: &[],
            expected_memory: &0x69_U256.to_be_bytes::<32>(),
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_next_action: InterpreterAction::Create {
                inputs: Box::new(CreateInputs {
                    caller: DEF_ADDR,
                    scheme: primitives::CreateScheme::Create,
                    value: 0x42_U256,
                    init_code: Bytes::copy_from_slice(&0x69_U256.to_be_bytes::<32>()),
                    gas_limit: 66917,
                })
            },
        }),
        create2(@raw {
            bytecode: &[op::PUSH1, 0x69, op::PUSH0, op::MSTORE, op::PUSH1, 100, op::PUSH1, 32, op::PUSH0, op::PUSH1, 0x42, op::CREATE2],
            expected_return: InstructionResult::CallOrCreate,
            // NOTE: The address is pushed by the caller.
            expected_stack: &[],
            expected_memory: &0x69_U256.to_be_bytes::<32>(),
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_next_action: InterpreterAction::Create {
                inputs: Box::new(CreateInputs {
                    caller: DEF_ADDR,
                    scheme: primitives::CreateScheme::Create2 { salt: 100_U256 },
                    value: 0x42_U256,
                    init_code: Bytes::copy_from_slice(&0x69_U256.to_be_bytes::<32>()),
                    gas_limit: 66908,
                })
            },
        }),
        call(@raw {
            bytecode: &[
                op::PUSH1, 1, // ret length
                op::PUSH1, 2, // ret offset
                op::PUSH1, 3, // args length
                op::PUSH1, 4, // args offset
                op::PUSH1, 5, // value
                op::PUSH1, 6, // address
                op::PUSH1, 7, // gas
                op::CALL,
            ],
            expected_return: InstructionResult::CallOrCreate,
            // NOTE: The return is pushed by the caller.
            expected_stack: &[],
            expected_memory: &[0; 32],
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_next_action: InterpreterAction::Call {
                inputs: Box::new(CallInputs {
                    input: Bytes::copy_from_slice(&[0; 3]),
                    return_memory_offset: 2..2+1,
                    gas_limit: gas::CALL_STIPEND + 7,
                    bytecode_address: Address::from_word(6_U256.into()),
                    target_address: Address::from_word(6_U256.into()),
                    caller: DEF_ADDR,
                    value: interpreter::CallValue::Transfer(5_U256),
                    scheme: interpreter::CallScheme::Call,
                    is_static: false,
                    is_eof: false,
                }),
            },
        }),
        callcode(@raw {
            bytecode: &[
                op::PUSH1, 1, // ret length
                op::PUSH1, 2, // ret offset
                op::PUSH1, 3, // args length
                op::PUSH1, 4, // args offset
                op::PUSH1, 5, // value
                op::PUSH1, 6, // address
                op::PUSH1, 7, // gas
                op::CALLCODE,
            ],
            expected_return: InstructionResult::CallOrCreate,
            expected_memory: MEMORY_WHAT_INTERPRETER_SAYS,
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_next_action: ACTION_WHAT_INTERPRETER_SAYS,
        }),
        delegatecall(@raw {
            bytecode: &[
                op::PUSH1, 1, // ret length
                op::PUSH1, 2, // ret offset
                op::PUSH1, 3, // args length
                op::PUSH1, 4, // args offset
                op::PUSH1, 5, // address
                op::PUSH1, 6, // gas
                op::DELEGATECALL,
            ],
            expected_return: InstructionResult::CallOrCreate,
            expected_memory: MEMORY_WHAT_INTERPRETER_SAYS,
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_next_action: ACTION_WHAT_INTERPRETER_SAYS,
        }),
        extcall(@raw {
            bytecode: &eof(&[
                op::PUSH1, 1, // value
                op::PUSH1, 2, // args length
                op::PUSH1, 3, // args offset
                op::PUSH1, 4, // address
                op::EXTCALL,
                op::STOP,
            ]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_return: InstructionResult::CallOrCreate,
            expected_memory: MEMORY_WHAT_INTERPRETER_SAYS,
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_next_action: ACTION_WHAT_INTERPRETER_SAYS,
        }),
        extdelegatecall(@raw {
            bytecode: &eof(&[
                op::PUSH1, 1, // args length
                op::PUSH1, 2, // args offset
                op::PUSH1, 3, // address
                op::EXTDELEGATECALL,
                op::STOP,
            ]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_return: InstructionResult::CallOrCreate,
            expected_memory: MEMORY_WHAT_INTERPRETER_SAYS,
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_next_action: ACTION_WHAT_INTERPRETER_SAYS,
        }),
        staticcall(@raw {
            bytecode: &[
                op::PUSH1, 1, // ret length
                op::PUSH1, 2, // ret offset
                op::PUSH1, 3, // args length
                op::PUSH1, 4, // args offset
                op::PUSH1, 5, // address
                op::PUSH1, 6, // gas
                op::STATICCALL,
            ],
            expected_return: InstructionResult::CallOrCreate,
            expected_memory: MEMORY_WHAT_INTERPRETER_SAYS,
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_next_action: ACTION_WHAT_INTERPRETER_SAYS,
        }),
        extstaticcall(@raw {
            bytecode: &eof(&[
                op::PUSH1, 1, // args length
                op::PUSH1, 2, // args offset
                op::PUSH1, 3, // address
                op::EXTSTATICCALL,
                op::STOP,
            ]),
            spec_id: SpecId::PRAGUE_EOF,
            expected_return: InstructionResult::CallOrCreate,
            expected_memory: MEMORY_WHAT_INTERPRETER_SAYS,
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_next_action: ACTION_WHAT_INTERPRETER_SAYS,
        }),
        ret(@raw {
            bytecode: &[op::PUSH1, 0x69, op::PUSH0, op::MSTORE, op::PUSH1, 32, op::PUSH0, op::RETURN],
            expected_return: InstructionResult::Return,
            expected_memory: &0x69_U256.to_be_bytes::<32>(),
            expected_gas: 3 + 2 + (3 + gas::memory_gas(1)) + 3 + 2,
            expected_next_action: InterpreterAction::Return {
                result: InterpreterResult {
                    result: InstructionResult::Return,
                    output: Bytes::copy_from_slice(&0x69_U256.to_be_bytes::<32>()),
                    gas: {
                        let mut gas = Gas::new(DEF_GAS_LIMIT);
                        assert!(gas.record_cost(3 + 2 + (3 + gas::memory_gas(1)) + 3 + 2));
                        gas
                    },
                },
            },
        }),
        revert(@raw {
            bytecode: &[op::PUSH1, 0x69, op::PUSH0, op::MSTORE, op::PUSH1, 32, op::PUSH0, op::REVERT],
            expected_return: InstructionResult::Revert,
            expected_memory: &0x69_U256.to_be_bytes::<32>(),
            expected_gas: 3 + 2 + (3 + gas::memory_gas(1)) + 3 + 2,
            expected_next_action: InterpreterAction::Return {
                result: InterpreterResult {
                    result: InstructionResult::Revert,
                    output: Bytes::copy_from_slice(&0x69_U256.to_be_bytes::<32>()),
                    gas: {
                        let mut gas = Gas::new(DEF_GAS_LIMIT);
                        assert!(gas.record_cost(3 + 2 + (3 + gas::memory_gas(1)) + 3 + 2));
                        gas
                    },
                },
            },
        }),
        selfdestruct(@raw {
            bytecode: &[op::PUSH1, 0x69, op::SELFDESTRUCT, op::INVALID],
            expected_return: InstructionResult::SelfDestruct,
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            assert_host: Some(|host| {
                assert_eq!(host.selfdestructs, [(DEF_ADDR, Address::with_last_byte(0x69))]);
            }),
        }),
    }

    regressions {
        // Mismatched costs in < BERLIN.
        // GeneralStateTests/stSolidityTest/TestKeywords.json
        st_solidity_keywords(@raw {
            bytecode: &hex!("7c01000000000000000000000000000000000000000000000000000000006000350463380e439681146037578063c040622614604757005b603d6084565b8060005260206000f35b604d6057565b8060005260206000f35b6000605f6084565b600060006101000a81548160ff0219169083021790555060ff60016000540416905090565b6000808160011560cd575b600a82121560a157600190910190608f565b81600a1460ac5760c9565b50600a5b60008160ff16111560c85760019182900391900360b0565b5b60d5565b6000925060ed565b8160001460e05760e8565b6001925060ed565b600092505b50509056"),
            spec_id: SpecId::ISTANBUL,
            modify_ecx: Some(|ecx| {
                ecx.contract.call_value = 1_U256;
                ecx.contract.input = Bytes::from(&hex!("c0406226"));
            }),
            expected_return: InstructionResult::Return,
            expected_stack: STACK_WHAT_INTERPRETER_SAYS,
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_memory: MEMORY_WHAT_INTERPRETER_SAYS,
            expected_next_action: InterpreterAction::Return {
                result: InterpreterResult {
                    result: InstructionResult::Return,
                    output: Bytes::copy_from_slice(&1_U256.to_be_bytes::<32>()),
                    gas: Gas::new(GAS_WHAT_INTERPRETER_SAYS),
                },
            },
            assert_host: Some(|host| {
                assert_eq!(host.storage.get(&0_U256), Some(&1_U256));
            }),
        }),
    }
}

#[track_caller]
fn eof(code: &[u8]) -> Bytes {
    eof_sections(&[code])
}

#[track_caller]
fn eof_sections(code: &[&[u8]]) -> Bytes {
    let eof = eof_sections_unchecked(code);
    match revm_interpreter::analysis::validate_eof(&eof) {
        Ok(()) => {}
        Err(EofError::Decode(e)) => panic!("{e}"),
        Err(EofError::Validation(e)) => match e {
            EofValidationError::UnknownOpcode
                if code.iter().any(|code| code.contains(&TEST_SUSPEND)) => {}
            EofValidationError::InvalidTypesSection => {}
            EofValidationError::MaxStackMismatch => {}
            e => panic!("validation error: {e:?}"),
        },
    }
    eof.raw
}

// We have to expose this because validation fails at invalid type sections
fn eof_sections_unchecked(code: &[&[u8]]) -> primitives::Eof {
    eof_body(code, vec![eof_subcontainer()]).into_eof()
}

fn eof_body(code: &[&[u8]], containers: Vec<Bytes>) -> primitives::eof::EofBody {
    revm_primitives::eof::EofBody {
        types_section: {
            let mut types =
                vec![primitives::eof::TypesSection { inputs: 0, outputs: 0x80, max_stack_size: 0 }];
            for _ in 1..code.len() {
                types.push(primitives::eof::TypesSection {
                    inputs: 0,
                    outputs: 0,
                    max_stack_size: 0,
                });
            }
            types
        },
        code_section: code.iter().copied().map(Bytes::copy_from_slice).collect(),
        container_section: containers,
        data_section: Bytes::from_static(DEF_DATA),
        is_data_filled: false,
    }
}

fn eof_subcontainer() -> Bytes {
    eof_body(&[&[op::STOP]], vec![]).into_eof().raw
}

fn bytecode_unop(op: u8, a: U256) -> [u8; 34] {
    let mut code = [0; 34];
    let mut i = 0;
    build_push32!(code[i], a);
    code[i] = op;
    code
}

fn bytecode_binop(op: u8, a: U256, b: U256) -> [u8; 67] {
    let mut code = [0; 67];
    let mut i = 0;
    build_push32!(code[i], b);
    build_push32!(code[i], a);
    code[i] = op;
    code
}

fn bytecode_ternop(op: u8, a: U256, b: U256, c: U256) -> [u8; 100] {
    let mut code = [0; 100];
    let mut i = 0;
    build_push32!(code[i], c);
    build_push32!(code[i], b);
    build_push32!(code[i], a);
    code[i] = op;
    code
}

#[rustfmt::skip]
#[allow(clippy::erasing_op, clippy::identity_op)]
fn rjumpv_code<const VALUE: u8>() -> Bytes {
    eof(&[
        op::PUSH1, VALUE,
        op::RJUMPV, 0x02,
        0x00, 3 + 0 * 3,
        0x00, 3 + 1 * 3,
        0x00, 3 + 2 * 3,
        /* _ => */ op::PUSH1, 10, op::STOP,
        /* 0 => */ op::PUSH1, 20, op::STOP,
        /* 1 => */ op::PUSH1, 30, op::STOP,
        /* 2 => */ op::PUSH1, 40, op::STOP,
    ])
}

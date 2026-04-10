//! Regression test: dedup must not poison DSE via lost leader marks.
//!
//! When a JUMPI's fall-through block gets deduped (marked DEAD_CODE), the leader mark on the
//! first dead instruction must propagate to the next alive instruction so that `rebuild_cfg`
//! starts a new block there. Without this, the JUMPI block absorbs the next alive instruction
//! (e.g. INVALID) as its terminator, which—if diverging—causes DSE to treat all exit stack
//! positions as dead, incorrectly NOOP-ing live PUSHes needed by the JUMPI's taken branch.
//!
//! The bytecode below has two identical `PUSH1 0x09; JUMP` fall-through blocks after separate
//! JUMPIs. Dedup merges the duplicate into the canonical copy, leaving dead code between the
//! second JUMPI and an alive INVALID instruction. The `PUSH1 0x2a` in the JUMPI block is
//! live-out (consumed at the JUMPI target), so DSE must not kill it.
//!
//! `inspect_stack` must be **off** for this test because `inspect_stack=true` disables the
//! diverging-terminator optimisation in DSE, preventing the bug from manifesting.

use super::{DEF_SPEC, with_evm_context};
use crate::{Backend, EvmCompiler};
use revm_bytecode::opcode as op;
use revm_interpreter::{InstructionResult, InterpreterAction, InterpreterResult};

/// Bytecode layout:
/// ```text
/// 00: 5b           JUMPDEST
/// 01: 60 00        PUSH1 0x00        ; cond = 0
/// 03: 60 09        PUSH1 0x09        ; JUMPI target
/// 05: 57           JUMPI             ; not taken
/// 06: 60 09        PUSH1 0x09        ; ┐ canonical fall-through
/// 08: 56           JUMP              ; ┘ (PUSH1 0x09; JUMP)
///
/// 09: 5b           JUMPDEST          ; JUMPI#2 block
/// 0a: 60 2a        PUSH1 0x2a        ; ← value DSE must preserve
/// 0c: 36           CALLDATASIZE      ; condition (≠ 0 with default calldata)
/// 0d: 60 14        PUSH1 0x14        ; JUMPI target = dest2
/// 0f: 57           JUMPI             ; taken
/// 10: 60 09        PUSH1 0x09        ; ┐ duplicate fall-through (deduped)
/// 12: 56           JUMP              ; ┘
///
/// 13: fe           INVALID           ; alive after dead block
///
/// 14: 5b           JUMPDEST          ; dest2: uses 0x2a
/// 15: 60 00        PUSH1 0x00
/// 17: 52           MSTORE            ; mem[0] = 0x2a
/// 18: 60 20        PUSH1 0x20
/// 1a: 60 00        PUSH1 0x00
/// 1c: f3           RETURN            ; return 32 bytes
/// ```
const BYTECODE: &[u8] = &[
    // Block 0: JUMPDEST, setup JUMPI#1 (cond=0, not taken)
    op::JUMPDEST,
    op::PUSH1,
    0x00, // cond = 0
    op::PUSH1,
    0x09, // target
    op::JUMPI,
    // Canonical fall-through: PUSH1 0x09; JUMP
    op::PUSH1,
    0x09,
    op::JUMP,
    // Block 2: JUMPDEST, JUMPI#2 (taken when CALLDATASIZE ≠ 0)
    op::JUMPDEST,
    op::PUSH1,
    0x2a,             // live-out value
    op::CALLDATASIZE, // condition
    op::PUSH1,
    0x14, // dest2
    op::JUMPI,
    // Duplicate fall-through: PUSH1 0x09; JUMP (same bytes as canonical)
    op::PUSH1,
    0x09,
    op::JUMP,
    // INVALID — alive instruction after dead deduped block
    op::INVALID,
    // dest2: consume the 0x2a value
    op::JUMPDEST,
    op::PUSH1,
    0x00,
    op::MSTORE,
    op::PUSH1,
    0x20,
    op::PUSH1,
    0x00,
    op::RETURN,
];

fn run_dedup_leader_test<B: Backend>(compiler: &mut EvmCompiler<B>) {
    unsafe { compiler.clear() }.unwrap();
    // inspect_stack must be OFF so that DSE's diverging-terminator logic is active.
    compiler.inspect_stack(false);
    let f = unsafe { compiler.jit("dedup_leader_propagation", BYTECODE, DEF_SPEC) }.unwrap();

    with_evm_context(BYTECODE, DEF_SPEC, |ecx, _stack, _stack_len| {
        let r = unsafe { f.call(None, None, ecx) };
        assert_eq!(r, InstructionResult::Return, "JIT must return Return, got {r:?}");

        let action = ecx.next_action.as_ref().expect("expected Return action");
        let InterpreterAction::Return(InterpreterResult { output, .. }) = action else {
            panic!("expected Return action, got {action:?}");
        };
        assert_eq!(output.len(), 32, "expected 32-byte return data");
        assert_eq!(
            output[31], 0x2a,
            "PUSH1 0x2a was incorrectly killed by DSE; return_data[31] = {:#04x}, expected 0x2a",
            output[31]
        );
    });
}

matrix_tests!(dedup_leader_propagation = run_dedup_leader_test);

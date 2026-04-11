//! Regression test: dedup of a reachable JUMPDEST block must preserve leader marks.
//!
//! Same class of bug as `dedup_leader_propagation.rs`, but triggered by a deduped
//! reachable JUMPDEST block rather than a JUMPI fall-through.
//!
//! When `rebuild_cfg` skips dead instructions before checking `is_reachable_jumpdest`,
//! a deduped JUMPDEST block never gets its leader mark set. `pending_leader` never fires
//! and the preceding live block absorbs the next alive instruction past the dead region.
//! If that instruction is a diverging terminator (INVALID), DSE treats all exit stack
//! positions as dead and incorrectly NOOPs live PUSHes.
//!
//! The key difference from `dedup_leader_propagation.rs`: that test's deduped block is
//! preceded by JUMPI, whose `is_branching()` already sets `is_leader[i+1]` on the first
//! dead instruction. Here, the deduped block is preceded by a non-branching instruction
//! (PUSH1), so the leader mark must come from `is_reachable_jumpdest` — which the buggy
//! code skips for dead instructions.
//!
//! Real-world trigger: tx 0x3a0ab5...31856 (block 5330710).
//!
//! `inspect_stack` must be **off** so DSE's diverging-terminator optimisation is active.

use super::{DEF_SPEC, with_evm_context};
use crate::{Backend, EvmCompiler};
use revm_bytecode::opcode as op;
use revm_interpreter::{InstructionResult, InterpreterAction, InterpreterResult};

/// Bytecode layout:
/// ```text
/// 00: JUMPDEST          ; entry
/// 01: PUSH1 0x00        ; dummy value (replaced at alt_entry)
/// 03: CALLDATASIZE      ; ≠ 0 → taken
/// 04: PUSH1 0x0e        ; → alt_entry
/// 06: JUMPI             ; taken
/// 07: PUSH1 0x14        ; → dup_ret (makes it reachable)
/// 09: JUMP
///
/// 0a: JUMPDEST          ; canonical_ret
/// 0b: PUSH1 0x19        ; → dest
/// 0d: JUMP
///
/// 0e: JUMPDEST          ; alt_entry
/// 0f: DUP1              ; dup entry's 0x00 (raises max_growth so DSE bitvec fits)
/// 10: POP               ; drop dup
/// 11: POP               ; drop entry's 0x00
/// 12: PUSH1 0x2a        ; ← live value DSE must NOT kill
///
/// 14: JUMPDEST          ; dup_ret (identical to canonical_ret → deduped)
/// 15: PUSH1 0x19        ; → dest
/// 17: JUMP
///
/// 18: INVALID           ; alive after dead dup_ret
///
/// 19: JUMPDEST          ; dest: uses 0x2a
/// 1a: PUSH1 0x00
/// 1c: MSTORE            ; mem[0] = 0x2a
/// 1d: PUSH1 0x20
/// 1f: PUSH1 0x00
/// 21: RETURN            ; return 32 bytes
/// ```
const BYTECODE: &[u8] = &[
    // 0x00: entry
    op::JUMPDEST,
    op::PUSH1,
    0x00,
    op::CALLDATASIZE,
    op::PUSH1,
    0x0e, // → alt_entry
    op::JUMPI,
    // 0x07: not-taken → dup_ret (makes it reachable for analysis)
    op::PUSH1,
    0x14, // → dup_ret
    op::JUMP,
    // 0x0a: canonical_ret
    op::JUMPDEST,
    op::PUSH1,
    0x19, // → dest
    op::JUMP,
    // 0x0e: alt_entry
    op::JUMPDEST,
    op::DUP1,
    op::POP,
    op::POP,
    op::PUSH1,
    0x2a, // ← DSE must NOT kill
    // 0x14: dup_ret (byte-identical to canonical_ret → deduped)
    op::JUMPDEST,
    op::PUSH1,
    0x19, // → dest
    op::JUMP,
    // 0x18: alive after dead dup_ret
    op::INVALID,
    // 0x19: dest
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

fn run_dedup_jumpdest_leader_test<B: Backend>(compiler: &mut EvmCompiler<B>) {
    unsafe { compiler.clear() }.unwrap();
    // inspect_stack must be OFF so that DSE's diverging-terminator logic is active.
    compiler.inspect_stack(false);
    let f = unsafe { compiler.jit("dedup_jumpdest_leader", BYTECODE, DEF_SPEC) }.unwrap();

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

matrix_tests!(dedup_jumpdest_leader = run_dedup_jumpdest_leader_test);

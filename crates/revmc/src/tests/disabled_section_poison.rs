//! Regression test: disabled opcodes must not poison stack sections.
//!
//! When a disabled opcode (e.g. TSTORE before Cancun) follows executable instructions in the same
//! section, its stack I/O requirements must not be folded into the section-head underflow check.
//! Otherwise the JIT returns `StackUnderflow` where the interpreter returns `NotActivated`.

use super::with_evm_context;
use crate::{Backend, EvmCompiler, SpecId};
use revm_bytecode::opcode as op;
use revm_interpreter::InstructionResult;

fn run_disabled_section_test<B: Backend>(
    compiler: &mut EvmCompiler<B>,
    name: &str,
    code: &[u8],
    spec_id: SpecId,
) {
    unsafe { compiler.clear() }.unwrap();
    let f = unsafe { compiler.jit(name, code, spec_id) }.unwrap();

    with_evm_context(code, spec_id, |ecx, stack, stack_len| {
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(
            r,
            InstructionResult::NotActivated,
            "{name}: JIT must return NotActivated, not {r:?}"
        );
    });
}

// CALLDATASIZE(0→1) ; TSTORE(2→0, disabled before Cancun).
// The interpreter executes CALLDATASIZE then halts at TSTORE with NotActivated.
// Before the fix, the JIT folded TSTORE's stack requirements into the section head
// and returned StackUnderflow at CALLDATASIZE.
matrix_tests!(
    calldatasize_tstore_shanghai = |jit| run_disabled_section_test(
        jit,
        "calldatasize_tstore_shanghai",
        &[op::CALLDATASIZE, op::TSTORE],
        SpecId::SHANGHAI,
    )
);

// PUSH0(0→1) ; TLOAD(1→1, disabled before Cancun).
// Same pattern — a valid prefix followed by a disabled opcode.
matrix_tests!(
    push0_tload_shanghai = |jit| run_disabled_section_test(
        jit,
        "push0_tload_shanghai",
        &[op::PUSH0, op::TLOAD],
        SpecId::SHANGHAI,
    )
);

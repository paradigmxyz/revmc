// Tests for resume_at persistence in call_with_interpreter (49274dbb).
//
// When JIT code encounters a CALL instruction, it suspends execution and
// returns `InterpreterAction::NewFrame`. After the callee completes, the
// caller must resume from the instruction *after* the CALL — not from PC=0.
//
// Without the fix, `call_with_interpreter` did not persist `ecx.resume_at`
// back into the interpreter's bytecode PC, causing re-execution from the
// beginning on every re-entry.

use super::{
    DEF_ADDR, DEF_CALLER, DEF_CD, DEF_GAS_LIMIT, DEF_SPEC, DEF_VALUE, TestHost,
    insert_call_outcome_test,
};
use crate::{Backend, EvmCompiler};
use revm_bytecode::opcode as op;
use revm_interpreter::{
    CallInput, FrameInput, Gas, InputsImpl, InstructionResult, Interpreter, InterpreterAction,
    InterpreterResult, SharedMemory, interpreter::ExtBytecode,
};
use revm_primitives::{Bytes, U256};

matrix_tests!(call_then_push = |compiler| run_call_then_push(compiler));
matrix_tests!(call_then_return = |compiler| run_call_then_return(compiler));

/// Contract: PUSH args → CALL → PUSH1 0x42 → STOP
///
/// After the CALL returns, execution must continue with PUSH1 0x42 (not restart
/// from PUSH args). We verify by checking the stack after the second call to
/// `call_with_interpreter`.
fn run_call_then_push<B: Backend>(compiler: &mut EvmCompiler<B>) {
    #[rustfmt::skip]
    let bytecode: &[u8] = &[
        // Set up CALL arguments (7 stack items)
        op::PUSH1, 0,    // ret length
        op::PUSH1, 0,    // ret offset
        op::PUSH1, 0,    // args length
        op::PUSH1, 0,    // args offset
        op::PUSH1, 0,    // value = 0
        op::PUSH1, 0x69, // address
        op::GAS,         // gas (all remaining)
        op::CALL,        // suspends here with NewFrame
        // -- after CALL returns, execution resumes here --
        op::PUSH1, 0x42, // push marker value
        op::STOP,
    ];

    unsafe { compiler.clear() }.unwrap();
    compiler.inspect_stack_length(true);
    let f = unsafe { compiler.jit("resume_call", bytecode, DEF_SPEC) }.unwrap();

    // First call: should suspend at CALL with NewFrame
    let mut host = TestHost::new();
    let input = InputsImpl {
        target_address: DEF_ADDR,
        bytecode_address: None,
        caller_address: DEF_CALLER,
        input: CallInput::Bytes(Bytes::from_static(DEF_CD)),
        call_value: DEF_VALUE,
    };
    let bytecode_obj = revm_bytecode::Bytecode::new_raw(Bytes::copy_from_slice(bytecode));
    let ext_bytecode = ExtBytecode::new(bytecode_obj);
    let mut interpreter =
        Interpreter::new(SharedMemory::new(), ext_bytecode, input, false, DEF_SPEC, DEF_GAS_LIMIT);

    let action = unsafe { f.call_with_interpreter(&mut interpreter, &mut host) };

    // Should get NewFrame(Call(...))
    let return_memory_offset = match &action {
        InterpreterAction::NewFrame(FrameInput::Call(call_inputs)) => {
            Some(call_inputs.return_memory_offset.clone())
        }
        other => panic!("expected NewFrame(Call), got {other:?}"),
    };

    // Simulate the callee completing successfully with no output
    let call_result = InterpreterResult {
        result: InstructionResult::Stop,
        output: Bytes::new(),
        gas: Gas::new(0),
    };
    insert_call_outcome_test(&mut interpreter, call_result, return_memory_offset);

    // Second call: should resume after CALL, execute PUSH1 0x42, STOP
    let action = unsafe { f.call_with_interpreter(&mut interpreter, &mut host) };

    match &action {
        InterpreterAction::Return(result) => {
            assert_eq!(
                result.result,
                InstructionResult::Stop,
                "expected Stop after resume, got {:?}",
                result.result
            );
            // Stack should have: [call_success_indicator(1), 0x42]
            // The CALL pushed success=1, then PUSH1 0x42
            assert_eq!(interpreter.stack.len(), 2, "stack should have 2 items");
            assert_eq!(
                interpreter.stack.data()[1],
                U256::from(0x42),
                "top of stack should be 0x42 (the marker value pushed after CALL)"
            );
        }
        other => panic!("expected Return after resume, got {other:?}"),
    }
}

/// Contract: PUSH args → CALL → PUSH1 32 → PUSH0 → RETURN
///
/// After the CALL returns, execution should RETURN 32 zero bytes.
/// Without the resume_at fix, it would re-enter from the beginning and try
/// to CALL again, never reaching the RETURN.
fn run_call_then_return<B: Backend>(compiler: &mut EvmCompiler<B>) {
    #[rustfmt::skip]
    let bytecode: &[u8] = &[
        // CALL arguments
        op::PUSH1, 0,    // ret length
        op::PUSH1, 0,    // ret offset
        op::PUSH1, 0,    // args length
        op::PUSH1, 0,    // args offset
        op::PUSH1, 0,    // value = 0
        op::PUSH1, 0x69, // address
        op::GAS,         // gas
        op::CALL,        // suspends
        // -- resume point --
        op::POP,          // pop call success indicator
        op::PUSH1, 32,   // return size
        op::PUSH0,       // return offset
        op::RETURN,       // should return 32 zero bytes
    ];

    unsafe { compiler.clear() }.unwrap();
    compiler.inspect_stack_length(true);
    let f = unsafe { compiler.jit("resume_return", bytecode, DEF_SPEC) }.unwrap();

    let mut host = TestHost::new();
    let input = InputsImpl {
        target_address: DEF_ADDR,
        bytecode_address: None,
        caller_address: DEF_CALLER,
        input: CallInput::Bytes(Bytes::from_static(DEF_CD)),
        call_value: DEF_VALUE,
    };
    let bytecode_obj = revm_bytecode::Bytecode::new_raw(Bytes::copy_from_slice(bytecode));
    let ext_bytecode = ExtBytecode::new(bytecode_obj);
    let mut interpreter =
        Interpreter::new(SharedMemory::new(), ext_bytecode, input, false, DEF_SPEC, DEF_GAS_LIMIT);

    // First call: suspends at CALL
    let action = unsafe { f.call_with_interpreter(&mut interpreter, &mut host) };
    let return_memory_offset = match &action {
        InterpreterAction::NewFrame(FrameInput::Call(call_inputs)) => {
            Some(call_inputs.return_memory_offset.clone())
        }
        other => panic!("expected NewFrame(Call), got {other:?}"),
    };

    // Simulate callee returning
    let call_result = InterpreterResult {
        result: InstructionResult::Stop,
        output: Bytes::new(),
        gas: Gas::new(0),
    };
    insert_call_outcome_test(&mut interpreter, call_result, return_memory_offset);

    // Second call: should resume, POP, PUSH1 32, PUSH0, RETURN
    let action = unsafe { f.call_with_interpreter(&mut interpreter, &mut host) };

    match &action {
        InterpreterAction::Return(result) => {
            assert_eq!(result.result, InstructionResult::Return);
            assert_eq!(result.output.len(), 32, "expected 32-byte return output");
        }
        other => panic!("expected Return after resume, got {other:?}"),
    }
}

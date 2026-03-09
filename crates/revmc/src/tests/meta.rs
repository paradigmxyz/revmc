use super::with_evm_context;
use crate::{Backend, EvmCompiler, SpecId};
use revm_interpreter::InstructionResult;
use revm_primitives::U256;

// Also tests multiple functions in the same module.
matrix_tests!(
    translate_then_compile = |compiler| {
        let bytecode: &[u8] = &[];
        let spec_id = SpecId::CANCUN;
        compiler.gas_metering(false);
        let gas_id = compiler.translate("test1", bytecode, spec_id).unwrap();
        compiler.gas_metering(true);
        let no_gas_id = compiler.translate("test2", bytecode, spec_id).unwrap();
        let gas_fn = unsafe { compiler.jit_function(gas_id) }.unwrap();
        let no_gas_fn = unsafe { compiler.jit_function(no_gas_id) }.unwrap();
        with_evm_context(bytecode, spec_id, |ecx, stack, stack_len| {
            let r = unsafe { gas_fn.call(Some(stack), Some(stack_len), ecx) };
            assert_eq!(r, InstructionResult::Stop);
            let r = unsafe { no_gas_fn.call(Some(stack), Some(stack_len), ecx) };
            assert_eq!(r, InstructionResult::Stop);
        });
    }
);

// Tests that calling `clear_ir` between two compilations works: the first function remains
// callable after the IR is cleared, and the second function compiles and runs correctly.
matrix_tests!(
    clear_ir_between_compiles = |compiler| {
        use revm_bytecode::opcode as op;

        let spec_id = SpecId::CANCUN;
        compiler.inspect_stack_length(true);

        // First function: PUSH1 42, STOP.
        let bytecode1: &[u8] = &[op::PUSH1, 42];
        let f1 = unsafe { compiler.jit("clear_ir_1", bytecode1, spec_id) }.unwrap();

        compiler.clear_ir().unwrap();

        // Second function: PUSH1 1, PUSH1 2, ADD, STOP.
        let bytecode2: &[u8] = &[op::PUSH1, 1, op::PUSH1, 2, op::ADD];
        let f2 = unsafe { compiler.jit("clear_ir_2", bytecode2, spec_id) }.unwrap();

        // First function still works after clear_ir + second compilation.
        with_evm_context(bytecode1, spec_id, |ecx, stack, stack_len| {
            let r = unsafe { f1.call(Some(stack), Some(stack_len), ecx) };
            assert_eq!(r, InstructionResult::Stop);
            assert_eq!(*stack_len, 1);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(42));
        });

        // Second function works.
        with_evm_context(bytecode2, spec_id, |ecx, stack, stack_len| {
            let r = unsafe { f2.call(Some(stack), Some(stack_len), ecx) };
            assert_eq!(r, InstructionResult::Stop);
            assert_eq!(*stack_len, 1);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(3));
        });
    }
);

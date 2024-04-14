use super::with_evm_context;
use crate::{Backend, JitEvm};
use revm_interpreter::InstructionResult;
use revm_primitives::SpecId;

matrix_tests!(translate_then_compile);

fn translate_then_compile<B: Backend>(jit: &mut JitEvm<B>) {
    let name = Some("test");
    let bytecode: &[u8] = &[];
    let spec_id = SpecId::CANCUN;
    jit.set_disable_gas(false);
    let gas_id = jit.translate(name, bytecode, spec_id).unwrap();
    jit.set_disable_gas(true);
    let no_gas_id = jit.translate(name, bytecode, spec_id).unwrap();
    let gas_fn = jit.jit_function(gas_id).unwrap();
    let no_gas_fn = jit.jit_function(no_gas_id).unwrap();
    with_evm_context(bytecode, |ecx, stack, stack_len| {
        let r = unsafe { gas_fn.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        let r = unsafe { no_gas_fn.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
    });
}

use super::with_evm_context;
use crate::{Backend, EvmCompiler};
use revm_interpreter::InstructionResult;
use revm_primitives::SpecId;

matrix_tests!(translate_then_compile);

fn translate_then_compile<B: Backend>(compiler: &mut EvmCompiler<B>) {
    let name = Some("test");
    let bytecode: &[u8] = &[];
    let spec_id = SpecId::CANCUN;
    compiler.gas_metering(false);
    let gas_id = compiler.translate(name, bytecode, spec_id).unwrap();
    compiler.gas_metering(true);
    let no_gas_id = compiler.translate(name, bytecode, spec_id).unwrap();
    let gas_fn = unsafe { compiler.jit_function(gas_id) }.unwrap();
    let no_gas_fn = unsafe { compiler.jit_function(no_gas_id) }.unwrap();
    with_evm_context(bytecode, |ecx, stack, stack_len| {
        let r = unsafe { gas_fn.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        let r = unsafe { no_gas_fn.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
    });
}

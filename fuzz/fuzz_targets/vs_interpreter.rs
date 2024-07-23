#![no_main]

use libfuzzer_sys::fuzz_target;
use revmc::{
    interpreter::OPCODE_INFO_JUMPTABLE,
    primitives::SpecId,
    tests::{run_test_case, TestCase},
    EvmCompiler, EvmLlvmBackend, OpcodesIter, OptimizationLevel,
};
use std::path::PathBuf;

fuzz_target!(|test_case: TestCase<'_>| {
    if should_skip(test_case.bytecode, test_case.spec_id) {
        return;
    }

    let context = revmc::llvm::inkwell::context::Context::create();
    let backend = EvmLlvmBackend::new(&context, false, OptimizationLevel::None).unwrap();
    let mut compiler = EvmCompiler::new(backend);
    if let Ok(dump_location) = std::env::var("COMPILER_DUMP") {
        compiler.set_dump_to(Some(PathBuf::from(dump_location)));
    }
    run_test_case(&test_case, &mut compiler);
});

fn should_skip(bytecode: &[u8], spec_id: SpecId) -> bool {
    OpcodesIter::new(bytecode, spec_id).any(|op| {
        let Some(info) = OPCODE_INFO_JUMPTABLE[op.opcode as usize] else { return true };
        // Skip if the immediate is incomplete.
        // TODO: What is the expected behavior here?
        if info.immediate_size() > 0 && op.immediate.is_none() {
            return true;
        }
        false
    })
}

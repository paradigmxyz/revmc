#![no_main]

use libfuzzer_sys::fuzz_target;
use revmc::{
    revm_bytecode::opcode::OPCODE_INFO,
    tests::{run_test_case, TestCase},
    EvmCompiler, EvmLlvmBackend, OpcodesIter, OptimizationLevel, SpecId,
};
use std::path::PathBuf;

fuzz_target!(|test_case: TestCase<'_>| {
    if should_skip(test_case.bytecode, test_case.spec_id) {
        return;
    }

    let backend = EvmLlvmBackend::new(false).unwrap();
    let mut compiler = EvmCompiler::new(backend);
    compiler.set_opt_level(OptimizationLevel::None);
    if let Ok(dump_location) = std::env::var("COMPILER_DUMP") {
        compiler.set_dump_to(Some(PathBuf::from(dump_location)));
    }
    run_test_case(&test_case, &mut compiler);
});

fn should_skip(bytecode: &[u8], spec_id: SpecId) -> bool {
    OpcodesIter::new(bytecode, spec_id).any(|op| {
        let Some(info) = OPCODE_INFO[op.opcode as usize] else { return true };
        // Skip if the immediate is incomplete.
        // TODO: What is the expected behavior here?
        if info.immediate_size() > 0 && op.immediate.is_none() {
            return true;
        }
        false
    })
}

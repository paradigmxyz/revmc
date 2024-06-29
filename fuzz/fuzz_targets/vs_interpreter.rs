#![no_main]

use libfuzzer_sys::fuzz_target;
use revmc::{
    interpreter::OPCODE_INFO_JUMPTABLE,
    tests::{run_test_case, TestCase},
    EvmCompiler, EvmLlvmBackend, OpcodesIter, OptimizationLevel,
};
use std::path::PathBuf;

fuzz_target!(|test_case: TestCase<'_>| {
    if should_skip(test_case.bytecode) {
        return;
    }

    let mut test_case = test_case;
    // EOF is not yet implemented.
    if test_case.spec_id > revmc::primitives::SpecId::CANCUN {
        test_case.spec_id = revmc::primitives::SpecId::CANCUN;
    }

    let context = revmc::llvm::inkwell::context::Context::create();
    let backend = EvmLlvmBackend::new(&context, false, OptimizationLevel::None).unwrap();
    let mut compiler = EvmCompiler::new(backend);
    if let Ok(dump_location) = std::env::var("COMPILER_DUMP") {
        compiler.set_dump_to(Some(PathBuf::from(dump_location)));
    }
    run_test_case(&test_case, &mut compiler);
});

fn should_skip(bytecode: &[u8]) -> bool {
    OpcodesIter::new(bytecode).any(|op| {
        let Some(info) = OPCODE_INFO_JUMPTABLE[op.opcode as usize] else { return true };
        // Skip all EOF opcodes since they might have different error codes in the interpreter.
        if is_eof(op.opcode) {
            return true;
        }
        // Skip if the immediate is incomplete.
        // TODO: What is the expected behavior here?
        if info.immediate_size() > 0 && op.immediate.is_none() {
            return true;
        }
        false
    })
}

// https://github.com/ipsilon/eof/blob/53e0e987e10bedee36d6c6ad0a6d1cfe806905e2/spec/eof.md?plain=1#L159
fn is_eof(op: u8) -> bool {
    use revmc::interpreter::opcode::*;
    matches!(
        op,
        RJUMP
            | RJUMPI
            | RJUMPV
            | CALLF
            | RETF
            | JUMPF
            | EOFCREATE
            | RETURNCONTRACT
            | DATALOAD
            | DATALOADN
            | DATASIZE
            | DATACOPY
            | DUPN
            | SWAPN
            | EXCHANGE
            | RETURNDATALOAD
            | EXTCALL
            | EXTDELEGATECALL
            | EXTSTATICCALL
    )
}

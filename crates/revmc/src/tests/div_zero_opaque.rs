//! Regression test: DIV/MOD/SMOD with zero divisor loaded from memory.
//!
//! When the divisor is opaque to LLVM (e.g. loaded via MLOAD), `udiv/urem/srem
//! i256` with divisor==0 is UB in LLVM IR. The old `binop!(@if_not_zero ...)`
//! used `select` which evaluates both operands unconditionally, allowing LLVM to
//! exploit the UB and remove the zero check entirely. The fix uses `lazy_select`
//! (conditional branch) so the division is never executed when b==0.

use super::{with_evm_context, DEF_SPEC};
use crate::{Backend, EvmCompiler};
use revm_bytecode::opcode as op;
use revm_interpreter::InstructionResult;
use revm_primitives::U256;

/// Build bytecode: MSTORE(a, 0), MSTORE(b, 32), MLOAD(32), MLOAD(0), `<op>`
///
/// The operands pass through MLOAD (an opaque builtin), preventing LLVM from
/// constant-folding or exploiting division-by-zero UB at compile time.
fn bytecode_binop_opaque(opcode: u8, a: U256, b: U256) -> Vec<u8> {
    let mut code = Vec::with_capacity(128);

    // Store a at memory[0]
    code.push(op::PUSH32);
    code.extend_from_slice(&a.to_be_bytes::<32>());
    code.push(op::PUSH1);
    code.push(0x00);
    code.push(op::MSTORE);

    // Store b at memory[32]
    code.push(op::PUSH32);
    code.extend_from_slice(&b.to_be_bytes::<32>());
    code.push(op::PUSH1);
    code.push(0x20);
    code.push(op::MSTORE);

    // Load b from memory (opaque to LLVM)
    code.push(op::PUSH1);
    code.push(0x20);
    code.push(op::MLOAD);

    // Load a from memory (opaque to LLVM)
    code.push(op::PUSH1);
    code.push(0x00);
    code.push(op::MLOAD);

    // Execute the target opcode: OP(a, b)
    code.push(opcode);

    code
}

fn run_opaque_zero_div_test<B: Backend>(
    compiler: &mut EvmCompiler<B>,
    name: &str,
    opcode: u8,
    a: U256,
) {
    let code = bytecode_binop_opaque(opcode, a, U256::ZERO);

    unsafe { compiler.clear() }.unwrap();
    compiler.inspect_stack_length(true);
    let f = unsafe { compiler.jit(name, &code, DEF_SPEC) }.unwrap();

    with_evm_context(&code, |ecx, stack, stack_len| {
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop, "{name}: unexpected return");
        assert_eq!(*stack_len, 1, "{name}: expected 1 stack element");
        let actual = stack.as_slice()[0].to_u256();
        assert_eq!(
            actual,
            U256::ZERO,
            "{name}: JIT produced {actual} but EVM spec mandates 0 for division by zero"
        );
    });
}

// EVM spec: x DIV 0 = 0, x MOD 0 = 0, x SMOD 0 = 0, x SDIV 0 = 0
matrix_tests!(div_zero = |jit| run_opaque_zero_div_test(jit, "div_zero", op::DIV, U256::from(32)));
matrix_tests!(mod_zero = |jit| run_opaque_zero_div_test(jit, "mod_zero", op::MOD, U256::from(32)));
matrix_tests!(
    smod_zero = |jit| run_opaque_zero_div_test(jit, "smod_zero", op::SMOD, U256::from(5))
);
matrix_tests!(
    sdiv_zero = |jit| run_opaque_zero_div_test(jit, "sdiv_zero", op::SDIV, U256::from(5))
);

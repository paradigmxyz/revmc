use super::{eof, eof_sections_unchecked, with_evm_context, DEF_SPEC};
use crate::{Backend, EvmCompiler, TEST_SUSPEND};
use revm_interpreter::{opcode as op, InstructionResult};
use revm_primitives::{SpecId, U256};

matrix_tests!(legacy = |compiler| run(compiler, TEST, DEF_SPEC));
matrix_tests!(eof_one_section = |compiler| run(compiler, &eof(TEST), SpecId::PRAGUE_EOF));
matrix_tests!(
    eof_two_sections = |compiler| run(
        compiler,
        &eof_sections_unchecked(&[&[op::JUMPF, 0x00, 0x01], TEST]).raw,
        SpecId::PRAGUE_EOF
    )
);

#[rustfmt::skip]
const TEST: &[u8] = &[
    // 0
    op::PUSH1, 0x42,
    TEST_SUSPEND,

    // 1
    op::PUSH1, 0x69,
    TEST_SUSPEND,
    
    // 2
    op::ADD,
    TEST_SUSPEND,

    // 3
    op::STOP,
];

fn run<B: Backend>(compiler: &mut EvmCompiler<B>, code: &[u8], spec_id: SpecId) {
    let f = unsafe { compiler.jit("resume", code, spec_id) }.unwrap();

    with_evm_context(code, |ecx, stack, stack_len| {
        let is_eof = ecx.contract.bytecode.is_eof();
        assert_eq!(ecx.resume_at, 0);

        // op::PUSH1, 0x42,
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::CallOrCreate);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42));
        assert_eq!(ecx.resume_at, 1);

        // op::PUSH1, 0x69,
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::CallOrCreate);
        assert_eq!(*stack_len, 2);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42));
        assert_eq!(stack.as_slice()[1].to_u256(), U256::from(0x69));
        assert_eq!(ecx.resume_at, 2);

        // op::ADD,
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::CallOrCreate);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69));
        assert_eq!(ecx.resume_at, 3);

        // stop
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69));
        assert_eq!(ecx.resume_at, 3);

        // Does not stack overflow EOF because of removed checks. This cannot happen in practice.
        if !is_eof {
            // op::ADD,
            ecx.resume_at = 2;
            let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
            assert_eq!(r, InstructionResult::StackUnderflow);
            assert_eq!(*stack_len, 1);
            assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69));
            assert_eq!(ecx.resume_at, 2);
        }

        stack.as_mut_slice()[*stack_len] = U256::from(2).into();
        *stack_len += 1;

        // op::ADD,
        ecx.resume_at = 2;
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::CallOrCreate);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2));
        assert_eq!(ecx.resume_at, 3);

        // op::PUSH1, 0x69,
        ecx.resume_at = 1;
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::CallOrCreate);
        assert_eq!(*stack_len, 2);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2));
        assert_eq!(stack.as_slice()[1].to_u256(), U256::from(0x69));
        assert_eq!(ecx.resume_at, 2);

        // op::ADD,
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::CallOrCreate);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2 + 0x69));
        assert_eq!(ecx.resume_at, 3);

        // stop
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2 + 0x69));
        assert_eq!(ecx.resume_at, 3);

        // stop
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2 + 0x69));
        assert_eq!(ecx.resume_at, 3);
    });
}

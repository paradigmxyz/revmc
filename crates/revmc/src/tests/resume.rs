use super::{DEF_SPEC, with_evm_context};
use crate::{Backend, EvmCompiler, SpecId, TEST_SUSPEND};
use revm_bytecode::opcode as op;
use revm_interpreter::InstructionResult;
use revm_primitives::U256;

matrix_tests!(legacy = |compiler| run(compiler, TEST, DEF_SPEC));

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

    with_evm_context(code, spec_id, |ecx, stack, stack_len| {
        assert_eq!(ecx.resume_at, 0);

        // Suspension returns `InstructionResult::Stop` with `resume_at` set to the next
        // resume point. The `resume_at` assertions below verify that suspension occurred
        // (as opposed to reaching an actual STOP opcode).

        // op::PUSH1, 0x42,
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42));
        let resume_1 = ecx.resume_at;
        if resume_1 < 100 {
            assert_eq!(resume_1, 1);
        }

        // op::PUSH1, 0x69,
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 2);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42));
        assert_eq!(stack.as_slice()[1].to_u256(), U256::from(0x69));
        let resume_2 = ecx.resume_at;
        if resume_2 < 100 {
            assert_eq!(resume_2, 2);
        }

        // op::ADD,
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69));
        let resume_3 = ecx.resume_at;
        if resume_3 < 100 {
            assert_eq!(resume_3, 3);
        }

        // stop
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69));
        assert_eq!(ecx.resume_at, resume_3);

        // op::ADD,
        ecx.resume_at = resume_2;
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::StackUnderflow);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69));
        assert_eq!(ecx.resume_at, resume_2);

        stack.as_mut_slice()[*stack_len] = U256::from(2).into();
        *stack_len += 1;

        // op::ADD,
        ecx.resume_at = resume_2;
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2));
        assert_eq!(ecx.resume_at, resume_3);

        // op::PUSH1, 0x69,
        ecx.resume_at = resume_1;
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 2);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2));
        assert_eq!(stack.as_slice()[1].to_u256(), U256::from(0x69));
        assert_eq!(ecx.resume_at, resume_2);

        // op::ADD,
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2 + 0x69));
        assert_eq!(ecx.resume_at, resume_3);

        // stop
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2 + 0x69));
        assert_eq!(ecx.resume_at, resume_3);

        // stop
        let r = unsafe { f.call(Some(stack), Some(stack_len), ecx) };
        assert_eq!(r, InstructionResult::Stop);
        assert_eq!(*stack_len, 1);
        assert_eq!(stack.as_slice()[0].to_u256(), U256::from(0x42 + 0x69 + 2 + 0x69));
        assert_eq!(ecx.resume_at, resume_3);
    });
}

#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]

#[doc(inline)]
pub use revm_jit_core::*;

#[cfg(feature = "llvm")]
#[doc(no_inline)]
pub use llvm::JitEvmLlvmBackend;
#[cfg(feature = "llvm")]
#[doc(inline)]
pub use revm_jit_llvm as llvm;

#[cfg(feature = "cranelift")]
#[doc(no_inline)]
pub use cranelift::JitEvmCraneliftBackend;
#[cfg(feature = "cranelift")]
#[doc(inline)]
pub use revm_jit_cranelift as cranelift;

mod compiler;
pub use compiler::JitEvm;

#[cfg(test)]
#[allow(dead_code, unused_imports)]
mod tests {
    use super::*;
    use revm_interpreter::opcode as op;
    use revm_jit_core::{Backend, ContextStack, Ret};
    use revm_primitives::{ruint::uint, U256};

    #[cfg(feature = "llvm")]
    #[test]
    fn test_llvm() {
        let context = revm_jit_llvm::inkwell::context::Context::create();
        run_tests_with_backend(|| JitEvmLlvmBackend::new(&context).unwrap());
    }

    struct TestCase<'a> {
        name: &'a str,
        bytecode: &'a [u8],

        expected_return: Ret,
        expected_stack: &'a [U256],
    }

    static TEST_CASES: &[TestCase<'static>] = &[
        TestCase {
            name: "stop",
            bytecode: &[op::STOP],
            expected_return: Ret::Stop,
            expected_stack: &[],
        },
        TestCase {
            name: "add",
            #[rustfmt::skip]
            bytecode: &[
                op::PUSH1, 0x01,
                op::PUSH1, 0x02,
                op::ADD,
                op::STOP,
            ],
            expected_return: Ret::Stop,
            expected_stack: &[uint!(3_U256)],
        },
    ];

    // TODO: Have to create a new backend per call for now
    fn run_tests_with_backend<B: Backend>(make_backend: impl Fn() -> B) {
        for &TestCase { name, bytecode, expected_return, expected_stack } in TEST_CASES {
            let mut jit = JitEvm::new(make_backend());
            jit.no_optimize();

            println!("Running test case: {name}");
            let f = jit.compile(bytecode).unwrap();

            let mut stack = ContextStack::new();
            let actual_return = unsafe { f(&mut stack) };
            assert_eq!(actual_return, expected_return);

            for (i, (chunk, expected)) in
                stack.as_slice().chunks_exact(32).zip(expected_stack).enumerate()
            {
                let bytes: [u8; 32] = chunk.try_into().unwrap();
                let actual = if cfg!(target_endian = "big") {
                    U256::from_be_bytes(bytes)
                } else {
                    U256::from_le_bytes(bytes)
                };
                assert_eq!(actual, *expected, "stack item {i} does not match");
            }
        }
    }
}

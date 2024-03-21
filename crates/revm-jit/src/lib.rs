#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

use revm_primitives::U256;

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

#[doc(no_inline)]
pub use revm_interpreter as interpreter;
#[doc(no_inline)]
pub use revm_primitives as primitives;

mod compiler;
pub use compiler::JitEvm;

mod bytecode;
pub use bytecode::{format_bytecode, imm_len, RawBytecodeIter, RawOpcode};

#[allow(dead_code)]
const MINUS_1: U256 = U256::MAX;
#[allow(dead_code)]
const I256_MIN: U256 = U256::from_limbs([
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x8000000000000000,
]);
#[allow(dead_code)]
const I256_MAX: U256 = U256::from_limbs([
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0x7FFFFFFFFFFFFFFF,
]);

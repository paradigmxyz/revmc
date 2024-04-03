#![doc = include_str!("../README.md")]
#![allow(missing_docs, clippy::missing_safety_doc)]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![no_std]

// We use `no_std` to reduce generated bitcode size.
#[allow(unused_imports)]
#[macro_use]
extern crate alloc;

use revm_interpreter::{gas as rgas, InstructionResult};
use revm_jit_context::{EvmContext, EvmWord};

#[macro_use]
mod macros;

mod utils;
pub use utils::*;

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_callback_mload(
    ecx: &mut EvmContext<'_>,
    [offset_ptr]: &mut [EvmWord; 1],
) -> InstructionResult {
    gas!(ecx, rgas::VERYLOW);
    let offset = try_into_usize!(offset_ptr.as_u256());
    resize_memory!(ecx, offset, 32);
    *offset_ptr = ecx.memory.get_u256(offset).into();
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_callback_mstore(
    ecx: &mut EvmContext<'_>,
    rev![offset, value]: &mut [EvmWord; 2],
) -> InstructionResult {
    gas!(ecx, rgas::VERYLOW);
    let offset = try_into_usize!(offset.as_u256());
    resize_memory!(ecx, offset, 32);
    ecx.memory.set(offset, &value.to_be_bytes());
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_callback_mstore8(
    ecx: &mut EvmContext<'_>,
    rev![offset, value]: &mut [EvmWord; 2],
) -> InstructionResult {
    gas!(ecx, rgas::VERYLOW);
    let offset = try_into_usize!(offset.as_u256());
    resize_memory!(ecx, offset, 1);
    ecx.memory.set_byte(offset, value.to_u256().byte(0));
    InstructionResult::Continue
}

#[no_mangle]
#[inline]
pub unsafe extern "C" fn __revm_jit_callback_msize(ecx: &mut EvmContext<'_>) -> usize {
    ecx.memory.len()
}

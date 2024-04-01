#![doc = include_str!("../README.md")]
#![allow(missing_docs, clippy::missing_safety_doc)]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

use revm_jit_context::EvmWord;

#[allow(unused_macros)]
#[macro_use]
mod macros;

#[no_mangle]
#[inline]
pub extern "C" fn __revm_jit_callback_addmod(sp: &mut [EvmWord; 3]) {
    read_words!(sp, a, b, c);
    *c = a.to_u256().add_mod(b.to_u256(), c.to_u256()).into();
}

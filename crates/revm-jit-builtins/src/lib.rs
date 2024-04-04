#![doc = include_str!("../README.md")]
#![allow(missing_docs, clippy::missing_safety_doc)]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![no_std]

#[allow(unused_imports)]
#[macro_use]
extern crate alloc;

#[cfg(feature = "ir")]
#[macro_use]
extern crate tracing;

use alloc::{boxed::Box, vec::Vec};
use revm_interpreter::{
    as_usize_saturated, gas as rgas, CreateInputs, InstructionResult, InterpreterAction,
    InterpreterResult, SStoreResult,
};
use revm_jit_context::{EvmContext, EvmWord};
use revm_primitives::{
    spec_to_generic, Bytes, CreateScheme, Log, LogData, SpecId, BLOCK_HASH_HISTORY, KECCAK_EMPTY,
    MAX_INITCODE_SIZE,
};

#[cfg(feature = "ir")]
mod ir;
#[cfg(feature = "ir")]
pub use ir::*;

#[macro_use]
mod macros;

mod utils;
use utils::*;

// NOTE: All functions MUST be `extern "C"` and their parameters must match `Builtin` enum.
//
// The `sp` parameter always points to the last popped stack element.
// If results are expected to be pushed back onto the stack, they must be written to the read
// pointers in **reverse order**, meaning the last pointer is the first return value.

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_panic(ptr: *const u8, len: usize) -> ! {
    let msg = core::str::from_utf8_unchecked(core::slice::from_raw_parts(ptr, len));
    panic!("{msg}");
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_addmod(rev![a, b, c]: &mut [EvmWord; 3]) {
    *c = a.to_u256().add_mod(b.to_u256(), c.to_u256()).into();
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_mulmod(rev![a, b, c]: &mut [EvmWord; 3]) {
    *c = a.to_u256().mul_mod(b.to_u256(), c.to_u256()).into();
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_exp(
    ecx: &mut EvmContext<'_>,
    rev![base, exponent_ptr]: &mut [EvmWord; 2],
    spec_id: SpecId,
) -> InstructionResult {
    let exponent = exponent_ptr.to_u256();
    // TODO: SpecId
    gas_opt!(ecx, spec_to_generic!(spec_id, rgas::exp_cost::<SPEC>(exponent)));
    *exponent_ptr = base.to_u256().pow(exponent).into();
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_keccak256(
    ecx: &mut EvmContext<'_>,
    rev![offset, len_ptr]: &mut [EvmWord; 2],
) -> InstructionResult {
    if *len_ptr == EvmWord::ZERO {
        *len_ptr = EvmWord::from_be_bytes(KECCAK_EMPTY.0);
        gas!(ecx, rgas::KECCAK256);
        return InstructionResult::Continue;
    }
    let len = try_into_usize!(len_ptr.as_u256());
    gas_opt!(ecx, rgas::keccak256_cost(len as u64));
    let offset = try_into_usize!(offset.as_u256());

    resize_memory!(ecx, offset, len);

    let data = ecx.memory.slice(offset, len);
    *len_ptr = EvmWord::from_be_bytes(revm_primitives::keccak256(data).0);

    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_balance(
    ecx: &mut EvmContext<'_>,
    address: &mut EvmWord,
    spec_id: SpecId,
) -> InstructionResult {
    let (balance, is_cold) = try_host!(ecx.host.balance(address.to_address()));
    *address = balance.into();
    let gas = if spec_id.is_enabled_in(SpecId::ISTANBUL) {
        spec_to_generic!(spec_id, rgas::account_access_gas::<SPEC>(is_cold))
    } else if spec_id.is_enabled_in(SpecId::TANGERINE) {
        400
    } else {
        20
    };
    gas!(ecx, gas);
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_calldatacopy(
    ecx: &mut EvmContext<'_>,
    sp: &mut [EvmWord; 3],
) -> InstructionResult {
    let input = decouple_lt(&ecx.contract.input[..]);
    copy_operation(ecx, sp, input)
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_codecopy(
    ecx: &mut EvmContext<'_>,
    sp: &mut [EvmWord; 3],
) -> InstructionResult {
    let code = decouple_lt(ecx.contract.bytecode.original_bytecode_slice());
    copy_operation(ecx, sp, code)
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_extcodesize(
    ecx: &mut EvmContext<'_>,
    address: &mut EvmWord,
    spec_id: SpecId,
) -> InstructionResult {
    let (code, is_cold) = try_host!(ecx.host.code(address.to_address()));
    *address = code.len().into();
    let gas = if spec_id.is_enabled_in(SpecId::BERLIN) {
        if is_cold {
            rgas::COLD_ACCOUNT_ACCESS_COST
        } else {
            rgas::WARM_STORAGE_READ_COST
        }
    } else if spec_id.is_enabled_in(SpecId::TANGERINE) {
        700
    } else {
        20
    };
    gas!(ecx, gas);
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_extcodecopy(
    ecx: &mut EvmContext<'_>,
    rev![address, memory_offset, code_offset, len]: &mut [EvmWord; 4],
    spec_id: SpecId,
) -> InstructionResult {
    let (code, is_cold) = try_host!(ecx.host.code(address.to_address()));
    let len = tri!(usize::try_from(len));
    // TODO: SpecId
    gas_opt!(ecx, spec_to_generic!(spec_id, rgas::extcodecopy_cost::<SPEC>(len as u64, is_cold)));
    if len != 0 {
        let memory_offset = try_into_usize!(memory_offset.as_u256());
        let code_offset = code_offset.to_u256();
        let code_offset = as_usize_saturated!(code_offset).min(code.len());
        resize_memory!(ecx, memory_offset, len);
        ecx.memory.set_data(memory_offset, code_offset, len, code.bytes());
    }
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_returndatacopy(
    ecx: &mut EvmContext<'_>,
    rev![memory_offset, offset, len]: &mut [EvmWord; 3],
) -> InstructionResult {
    let len = tri!(usize::try_from(len));
    gas_opt!(ecx, rgas::verylowcopy_cost(len as u64));
    let data_offset = offset.to_u256();
    let data_offset = as_usize_saturated!(data_offset);
    let (data_end, overflow) = data_offset.overflowing_add(len);
    if overflow || data_end > ecx.return_data.len() {
        return InstructionResult::OutOfOffset;
    }
    if len != 0 {
        let memory_offset = try_into_usize!(memory_offset.as_u256());
        resize_memory!(ecx, memory_offset, len);
        ecx.memory.set(memory_offset, &ecx.return_data[data_offset..data_end]);
    }
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_extcodehash(
    ecx: &mut EvmContext<'_>,
    address: &mut EvmWord,
    spec_id: SpecId,
) -> InstructionResult {
    let (hash, is_cold) = try_host!(ecx.host.code_hash(address.to_address()));
    *address = EvmWord::from_be_bytes(hash.0);
    let gas = if spec_id.is_enabled_in(SpecId::BERLIN) {
        if is_cold {
            rgas::COLD_ACCOUNT_ACCESS_COST
        } else {
            rgas::WARM_STORAGE_READ_COST
        }
    } else if spec_id.is_enabled_in(SpecId::ISTANBUL) {
        700
    } else {
        400
    };
    gas!(ecx, gas);
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_blockhash(
    ecx: &mut EvmContext<'_>,
    number_ptr: &mut EvmWord,
) -> InstructionResult {
    let number = number_ptr.to_u256();
    if let Some(diff) = ecx.host.env().block.number.checked_sub(number) {
        let diff = as_usize_saturated!(diff);
        // blockhash should push zero if number is same as current block number.
        if diff != 0 && diff <= BLOCK_HASH_HISTORY {
            let hash = try_host!(ecx.host.block_hash(number));
            *number_ptr = EvmWord::from_be_bytes(hash.0);
            return InstructionResult::Continue;
        }
    }
    *number_ptr = EvmWord::ZERO;
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_self_balance(
    ecx: &mut EvmContext<'_>,
    slot: &mut EvmWord,
) -> InstructionResult {
    let (balance, _) = try_host!(ecx.host.balance(ecx.contract.address));
    *slot = balance.into();
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_blob_hash(
    ecx: &mut EvmContext<'_>,
    index_ptr: &mut EvmWord,
) {
    let index = index_ptr.to_u256();
    *index_ptr = EvmWord::from_be_bytes(
        ecx.host
            .env()
            .tx
            .blob_hashes
            .get(as_usize_saturated!(index))
            .copied()
            .unwrap_or_default()
            .0,
    );
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_blob_base_fee(
    ecx: &mut EvmContext<'_>,
    slot: &mut EvmWord,
) {
    *slot = ecx.host.env().block.get_blob_gasprice().unwrap_or_default().into();
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_mload(
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
pub unsafe extern "C" fn __revm_jit_builtin_mstore(
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
pub unsafe extern "C" fn __revm_jit_builtin_mstore8(
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
pub unsafe extern "C" fn __revm_jit_builtin_sload(
    ecx: &mut EvmContext<'_>,
    index: &mut EvmWord,
    spec_id: SpecId,
) -> InstructionResult {
    let address = ecx.contract.address;
    let (res, is_cold) = try_opt!(ecx.host.sload(address, index.to_u256()));
    gas!(ecx, spec_to_generic!(spec_id, rgas::sload_cost::<SPEC>(is_cold)));
    *index = res.into();
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_sstore(
    ecx: &mut EvmContext<'_>,
    rev![index, value]: &mut [EvmWord; 2],
    spec_id: SpecId,
) -> InstructionResult {
    let SStoreResult { original_value: original, present_value: old, new_value: new, is_cold } =
        try_opt!(ecx.host.sstore(ecx.contract.address, index.to_u256(), value.to_u256()));

    // TODO: SpecId
    gas_opt!(
        ecx,
        spec_to_generic!(
            spec_id,
            rgas::sstore_cost::<SPEC>(original, old, new, ecx.gas.remaining(), is_cold)
        )
    );
    ecx.gas
        .record_refund(spec_to_generic!(spec_id, rgas::sstore_refund::<SPEC>(original, old, new)));
    InstructionResult::Continue
}

#[inline]
#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_msize(ecx: &mut EvmContext<'_>) -> usize {
    ecx.memory.len()
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_tstore(
    ecx: &mut EvmContext<'_>,
    rev![key, value]: &mut [EvmWord; 2],
) {
    ecx.host.tstore(ecx.contract.address, key.to_u256(), value.to_u256());
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_tload(ecx: &mut EvmContext<'_>, key: &mut EvmWord) {
    *key = ecx.host.tload(ecx.contract.address, key.to_u256()).into();
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_log(
    ecx: &mut EvmContext<'_>,
    sp: *mut EvmWord,
    n: u8,
) -> InstructionResult {
    debug_assert!(n <= 4, "invalid log topic count: {n}");
    let sp = sp.add(n as usize);
    read_words!(sp, offset, len);
    let len = tri!(usize::try_from(len));
    gas_opt!(ecx, rgas::log_cost(n, len as u64));
    let data = if len != 0 {
        let offset = try_into_usize!(offset.as_u256());
        resize_memory!(ecx, offset, len);
        Bytes::copy_from_slice(ecx.memory.slice(offset, len))
    } else {
        Bytes::new()
    };

    let mut topics = Vec::with_capacity(n as usize);
    for i in 0..n {
        topics.push(sp.sub(i as usize).read().to_be_bytes().into());
    }

    ecx.host.log(Log {
        address: ecx.contract.address,
        data: LogData::new(topics, data).expect("too many topics"),
    });
    InstructionResult::Continue
}

// NOTE: Return `InstructionResult::Continue` here to indicate success, not the final result of
// the execution.
#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_create(
    ecx: &mut EvmContext<'_>,
    rev![value, code_offset, len, salt]: &mut [EvmWord; 4],
    spec_id: SpecId,
    is_create2: bool,
) -> InstructionResult {
    let len = tri!(usize::try_from(len));
    let code = if len != 0 {
        if spec_id.is_enabled_in(SpecId::SHANGHAI) {
            // Limit is set as double of max contract bytecode size
            let max_initcode_size = ecx
                .host
                .env()
                .cfg
                .limit_contract_code_size
                .map(|limit| limit.saturating_mul(2))
                .unwrap_or(MAX_INITCODE_SIZE);
            if len > max_initcode_size {
                return InstructionResult::CreateInitCodeSizeLimit;
            }
            gas!(ecx, rgas::initcode_cost(len as u64));
        }

        let code_offset = try_into_usize!(code_offset.as_u256());
        resize_memory!(ecx, code_offset, len);
        Bytes::copy_from_slice(ecx.memory.slice(code_offset, len))
    } else {
        Bytes::new()
    };

    gas_opt!(ecx, if is_create2 { rgas::create2_cost(len) } else { Some(rgas::CREATE) });

    let scheme = if is_create2 {
        CreateScheme::Create2 { salt: salt.to_u256() }
    } else {
        CreateScheme::Create
    };

    let mut gas_limit = ecx.gas.remaining();
    if spec_id.is_enabled_in(SpecId::TANGERINE) {
        gas_limit -= gas_limit / 64;
    }
    gas!(ecx, gas_limit);

    *ecx.next_action = InterpreterAction::Create {
        inputs: Box::new(CreateInputs {
            caller: ecx.contract.address,
            scheme,
            value: value.to_u256(),
            init_code: code,
            gas_limit,
        }),
    };

    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_do_return(
    ecx: &mut EvmContext<'_>,
    rev![offset, len]: &mut [EvmWord; 2],
    result: InstructionResult,
) -> InstructionResult {
    let len = try_into_usize!(len.as_u256());
    let output = if len != 0 {
        let offset = try_into_usize!(offset.as_u256());
        resize_memory!(ecx, offset, len);
        ecx.memory.slice(offset, len).to_vec().into()
    } else {
        Bytes::new()
    };
    *ecx.next_action =
        InterpreterAction::Return { result: InterpreterResult { output, gas: *ecx.gas, result } };
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_selfdestruct(
    ecx: &mut EvmContext<'_>,
    target: &mut EvmWord,
    spec_id: SpecId,
) -> InstructionResult {
    let res = try_host!(ecx.host.selfdestruct(ecx.contract.address, target.to_address()));

    // EIP-3529: Reduction in refunds
    if !spec_id.is_enabled_in(SpecId::LONDON) && !res.previously_destroyed {
        ecx.gas.record_refund(rgas::SELFDESTRUCT);
    }
    // TODO: SpecId
    gas!(ecx, spec_to_generic!(spec_id, rgas::selfdestruct_cost::<SPEC>(res)));

    InstructionResult::Continue
}

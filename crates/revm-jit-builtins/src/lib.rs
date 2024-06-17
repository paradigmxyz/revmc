#![doc = include_str!("../README.md")]
#![allow(missing_docs, clippy::missing_safety_doc)]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[macro_use]
#[cfg(feature = "ir")]
extern crate tracing;

use alloc::{boxed::Box, vec::Vec};
use revm_interpreter::{
    as_u64_saturated, as_usize_saturated, CallInputs, CallScheme, CallValue, CreateInputs,
    InstructionResult, InterpreterAction, InterpreterResult, LoadAccountResult, SStoreResult,
};
use revm_jit_context::{EvmContext, EvmWord};
use revm_primitives::{
    Bytes, CreateScheme, Log, LogData, SpecId, KECCAK_EMPTY, MAX_INITCODE_SIZE, U256,
};

pub mod gas;

#[cfg(feature = "ir")]
mod ir;
#[cfg(feature = "ir")]
pub use ir::*;

#[macro_use]
mod macros;

mod utils;
use utils::*;

/// The kind of a `*CALL*` instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CallKind {
    /// `CALL`.
    Call,
    /// `CALLCODE`.
    CallCode,
    /// `DELEGATECALL`.
    DelegateCall,
    /// `STATICCALL`.
    StaticCall,
}

impl From<CallKind> for CallScheme {
    fn from(kind: CallKind) -> Self {
        match kind {
            CallKind::Call => CallScheme::Call,
            CallKind::CallCode => CallScheme::CallCode,
            CallKind::DelegateCall => CallScheme::DelegateCall,
            CallKind::StaticCall => CallScheme::StaticCall,
        }
    }
}

/// The kind of a `CREATE*` instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CreateKind {
    /// `CREATE`.
    Create,
    /// `CREATE2`.
    Create2,
}

// NOTE: All functions MUST be `extern "C"` and their parameters must match `Builtin` enum.
//
// The `sp` parameter always points to the last popped stack element.
// If results are expected to be pushed back onto the stack, they must be written to the read
// pointers in **reverse order**, meaning the last pointer is the first return value.

#[no_mangle]
pub unsafe extern "C-unwind" fn __revm_jit_builtin_panic(data: *const u8, len: usize) -> ! {
    let msg = core::str::from_utf8_unchecked(core::slice::from_raw_parts(data, len));
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
    gas_opt!(ecx, gas::dyn_exp_cost(spec_id, exponent));
    *exponent_ptr = base.to_u256().pow(exponent).into();
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_keccak256(
    ecx: &mut EvmContext<'_>,
    rev![offset, len_ptr]: &mut [EvmWord; 2],
) -> InstructionResult {
    let len = try_into_usize!(len_ptr);
    *len_ptr = EvmWord::from_be_bytes(if len == 0 {
        KECCAK_EMPTY.0
    } else {
        gas_opt!(ecx, gas::dyn_keccak256_cost(len as u64));
        let offset = try_into_usize!(offset);
        resize_memory!(ecx, offset, len);
        let data = ecx.memory.slice(offset, len);
        revm_primitives::keccak256(data).0
    });
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
    let gas = if spec_id.is_enabled_in(SpecId::BERLIN) {
        gas::warm_cold_cost(is_cold)
    } else if spec_id.is_enabled_in(SpecId::ISTANBUL) {
        // EIP-1884: Repricing for trie-size-dependent opcodes
        700
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
pub unsafe extern "C" fn __revm_jit_builtin_codesize(ecx: &mut EvmContext<'_>) -> usize {
    assume!(!ecx.contract.bytecode.is_eof());
    ecx.contract.bytecode.len()
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_codecopy(
    ecx: &mut EvmContext<'_>,
    sp: &mut [EvmWord; 3],
) -> InstructionResult {
    assume!(!ecx.contract.bytecode.is_eof());
    let code = decouple_lt(ecx.contract.bytecode.original_byte_slice());
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
            gas::COLD_ACCOUNT_ACCESS_COST
        } else {
            gas::WARM_STORAGE_READ_COST
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
    let len = try_into_usize!(len);
    gas_opt!(ecx, gas::extcodecopy_cost(spec_id, len as u64, is_cold));
    if len != 0 {
        let memory_offset = try_into_usize!(memory_offset);
        let code_offset = code_offset.to_u256();
        let code_offset = as_usize_saturated!(code_offset).min(code.len());
        resize_memory!(ecx, memory_offset, len);
        ecx.memory.set_data(memory_offset, code_offset, len, &code);
    }
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_returndatacopy(
    ecx: &mut EvmContext<'_>,
    rev![memory_offset, offset, len]: &mut [EvmWord; 3],
) -> InstructionResult {
    let len = try_into_usize!(len);
    gas_opt!(ecx, gas::dyn_verylowcopy_cost(len as u64));
    let data_offset = offset.to_u256();
    let data_offset = as_usize_saturated!(data_offset);
    let (data_end, overflow) = data_offset.overflowing_add(len);
    if overflow || data_end > ecx.return_data.len() {
        return InstructionResult::OutOfOffset;
    }
    if len != 0 {
        let memory_offset = try_into_usize!(memory_offset);
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
            gas::COLD_ACCOUNT_ACCESS_COST
        } else {
            gas::WARM_STORAGE_READ_COST
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
    let hash = try_host!(ecx.host.block_hash(number_ptr.to_u256()));
    *number_ptr = EvmWord::from_be_bytes(hash.0);
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_difficulty(
    ecx: &mut EvmContext<'_>,
    slot: &mut EvmWord,
    spec_id: SpecId,
) {
    *slot = if spec_id.is_enabled_in(SpecId::MERGE) {
        EvmWord::from_be_bytes(ecx.host.env().block.prevrandao.unwrap().0)
    } else {
        ecx.host.env().block.difficulty.into()
    };
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_self_balance(
    ecx: &mut EvmContext<'_>,
    slot: &mut EvmWord,
) -> InstructionResult {
    let (balance, _) = try_host!(ecx.host.balance(ecx.contract.target_address));
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
    offset_ptr: &mut EvmWord,
) -> InstructionResult {
    let offset = try_into_usize!(offset_ptr);
    resize_memory!(ecx, offset, 32);
    *offset_ptr = EvmWord::from_be_bytes(ecx.memory.get_word(offset).0);
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_mstore(
    ecx: &mut EvmContext<'_>,
    rev![offset, value]: &mut [EvmWord; 2],
) -> InstructionResult {
    let offset = try_into_usize!(offset);
    resize_memory!(ecx, offset, 32);
    ecx.memory.set(offset, &value.to_be_bytes());
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_mstore8(
    ecx: &mut EvmContext<'_>,
    rev![offset, value]: &mut [EvmWord; 2],
) -> InstructionResult {
    let offset = try_into_usize!(offset);
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
    let address = ecx.contract.target_address;
    let (res, is_cold) = try_opt!(ecx.host.sload(address, index.to_u256()));
    gas!(ecx, gas::sload_cost(spec_id, is_cold));
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
        try_opt!(ecx.host.sstore(ecx.contract.target_address, index.to_u256(), value.to_u256()));

    gas_opt!(ecx, gas::sstore_cost(spec_id, original, old, new, ecx.gas.remaining(), is_cold));
    ecx.gas.record_refund(gas::sstore_refund(spec_id, original, old, new));
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_msize(ecx: &mut EvmContext<'_>) -> usize {
    ecx.memory.len()
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_tstore(
    ecx: &mut EvmContext<'_>,
    rev![key, value]: &mut [EvmWord; 2],
) {
    ecx.host.tstore(ecx.contract.target_address, key.to_u256(), value.to_u256());
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_tload(ecx: &mut EvmContext<'_>, key: &mut EvmWord) {
    *key = ecx.host.tload(ecx.contract.target_address, key.to_u256()).into();
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_mcopy(
    ecx: &mut EvmContext<'_>,
    rev![dst, src, len]: &mut [EvmWord; 3],
) -> InstructionResult {
    let len = try_into_usize!(len);
    gas_opt!(ecx, gas::dyn_verylowcopy_cost(len as u64));
    if len != 0 {
        let dst = try_into_usize!(dst);
        let src = try_into_usize!(src);
        resize_memory!(ecx, dst.max(src), len);
        ecx.memory.copy(dst, src, len);
    }
    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_log(
    ecx: &mut EvmContext<'_>,
    sp: *mut EvmWord,
    n: u8,
) -> InstructionResult {
    assume!(n <= 4, "invalid log topic count: {n}");
    let sp = sp.add(n as usize);
    read_words!(sp, offset, len);
    let len = try_into_usize!(len);
    gas_opt!(ecx, gas::dyn_log_cost(len as u64));
    let data = if len != 0 {
        let offset = try_into_usize!(offset);
        resize_memory!(ecx, offset, len);
        Bytes::copy_from_slice(ecx.memory.slice(offset, len))
    } else {
        Bytes::new()
    };

    let mut topics = Vec::with_capacity(n as usize);
    for i in 1..=n {
        topics.push(sp.sub(i as usize).read().to_be_bytes().into());
    }

    ecx.host.log(Log {
        address: ecx.contract.target_address,
        data: LogData::new(topics, data).expect("too many topics"),
    });
    InstructionResult::Continue
}

// NOTE: Return `InstructionResult::Continue` here to indicate success, not the final result of
// the execution.
#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_create(
    ecx: &mut EvmContext<'_>,
    sp: *mut EvmWord,
    spec_id: SpecId,
    create_kind: CreateKind,
) -> InstructionResult {
    let len = match create_kind {
        CreateKind::Create => 3,
        CreateKind::Create2 => 4,
    };
    let mut sp = sp.add(len);
    pop!(sp; value, code_offset, len);

    let len = try_into_usize!(len);
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
            gas!(ecx, gas::initcode_cost(len as u64));
        }

        let code_offset = try_into_usize!(code_offset);
        resize_memory!(ecx, code_offset, len);
        Bytes::copy_from_slice(ecx.memory.slice(code_offset, len))
    } else {
        Bytes::new()
    };

    let is_create2 = create_kind == CreateKind::Create2;
    gas_opt!(ecx, if is_create2 { gas::create2_cost(len as u64) } else { Some(gas::CREATE) });

    let scheme = if is_create2 {
        pop!(sp; salt);
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
            caller: ecx.contract.target_address,
            scheme,
            value: value.to_u256(),
            init_code: code,
            gas_limit,
        }),
    };

    InstructionResult::Continue
}

#[no_mangle]
pub unsafe extern "C" fn __revm_jit_builtin_call(
    ecx: &mut EvmContext<'_>,
    sp: *mut EvmWord,
    spec_id: SpecId,
    call_kind: CallKind,
) -> InstructionResult {
    let len = match call_kind {
        CallKind::Call | CallKind::CallCode => 7,
        CallKind::DelegateCall | CallKind::StaticCall => 6,
    };
    let mut sp = sp.add(len);

    pop!(sp; local_gas_limit, to);
    let local_gas_limit = local_gas_limit.to_u256();
    let to = to.to_address();

    // max gas limit is not possible in real ethereum situation.
    // But for tests we would not like to fail on this.
    // Gas limit for subcall is taken as min of this value and current gas limit.
    let local_gas_limit = as_u64_saturated!(local_gas_limit);

    let value = match call_kind {
        CallKind::Call | CallKind::CallCode => {
            pop!(sp; value);
            let value = value.to_u256();
            if matches!(call_kind, CallKind::Call) && ecx.is_static && value != U256::ZERO {
                return InstructionResult::CallNotAllowedInsideStatic;
            }
            value
        }
        CallKind::DelegateCall | CallKind::StaticCall => U256::ZERO,
    };

    pop!(sp; in_offset, in_len, out_offset, out_len);

    let in_len = try_into_usize!(in_len);
    let input = if in_len != 0 {
        let in_offset = try_into_usize!(in_offset);
        resize_memory!(ecx, in_offset, in_len);
        Bytes::copy_from_slice(ecx.memory.slice(in_offset, in_len))
    } else {
        Bytes::new()
    };

    let out_len = try_into_usize!(out_len);
    let out_offset = if out_len != 0 {
        let out_offset = try_into_usize!(out_offset);
        resize_memory!(ecx, out_offset, out_len);
        out_offset
    } else {
        usize::MAX // unrealistic value so we are sure it is not used
    };

    // Load account and calculate gas cost.
    let LoadAccountResult { is_cold, mut is_empty } = try_host!(ecx.host.load_account(to));
    if !matches!(call_kind, CallKind::Call) {
        is_empty = false;
    }

    gas!(ecx, gas::call_cost(spec_id, value != U256::ZERO, is_cold, is_empty));

    // EIP-150: Gas cost changes for IO-heavy operations
    let mut gas_limit = if spec_id.is_enabled_in(SpecId::TANGERINE) {
        let gas = ecx.gas.remaining();
        // take l64 part of gas_limit
        (gas - gas / 64).min(local_gas_limit)
    } else {
        local_gas_limit
    };

    gas!(ecx, gas_limit);

    // Add call stipend if there is value to be transferred.
    if matches!(call_kind, CallKind::Call | CallKind::CallCode) && value != U256::ZERO {
        gas_limit = gas_limit.saturating_add(gas::CALL_STIPEND);
    }

    *ecx.next_action = InterpreterAction::Call {
        inputs: Box::new(CallInputs {
            input,
            return_memory_offset: out_offset..out_offset + out_len,
            gas_limit,
            bytecode_address: to,
            target_address: if matches!(call_kind, CallKind::DelegateCall | CallKind::CallCode) {
                ecx.contract.target_address
            } else {
                to
            },
            caller: if call_kind == CallKind::DelegateCall {
                ecx.contract.caller
            } else {
                ecx.contract.target_address
            },
            value: if matches!(call_kind, CallKind::DelegateCall) {
                CallValue::Apparent(ecx.contract.call_value)
            } else {
                CallValue::Transfer(value)
            },
            scheme: call_kind.into(),
            is_static: ecx.is_static || matches!(call_kind, CallKind::StaticCall),
            // TODO(EOF)
            is_eof: false,
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
    let len = try_into_usize!(len);
    let output = if len != 0 {
        let offset = try_into_usize!(offset);
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
    let res = try_host!(ecx.host.selfdestruct(ecx.contract.target_address, target.to_address()));

    // EIP-3529: Reduction in refunds
    if !spec_id.is_enabled_in(SpecId::LONDON) && !res.previously_destroyed {
        ecx.gas.record_refund(gas::SELFDESTRUCT);
    }
    gas!(ecx, gas::selfdestruct_cost(spec_id, res));

    InstructionResult::Continue
}

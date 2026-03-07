#![doc = include_str!("../README.md")]
#![allow(missing_docs, clippy::missing_safety_doc)]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[macro_use]
#[cfg(feature = "ir")]
extern crate tracing;

use alloc::{boxed::Box, vec::Vec};
use revm_interpreter::{
    as_u64_saturated, as_usize_saturated,
    interpreter_types::{InputsTr, MemoryTr},
    CallInput, CallInputs, CallScheme, CallValue, CreateInputs, CreateScheme, InstructionResult,
    InterpreterAction, InterpreterResult,
};
use revm_primitives::{hardfork::SpecId, Bytes, Log, LogData, KECCAK_EMPTY, U256};
use revmc_context::{EvmContext, EvmWord};

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
            CallKind::Call => Self::Call,
            CallKind::CallCode => Self::CallCode,
            CallKind::DelegateCall => Self::DelegateCall,
            CallKind::StaticCall => Self::StaticCall,
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
pub unsafe extern "C-unwind" fn __revmc_builtin_panic(data: *const u8, len: usize) -> ! {
    let msg = core::str::from_utf8_unchecked(core::slice::from_raw_parts(data, len));
    panic!("{msg}");
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_udiv(rev![a, b]: &mut [EvmWord; 2]) {
    let divisor = b.to_u256();
    *b = if divisor.is_zero() { U256::ZERO } else { a.to_u256().wrapping_div(divisor) }.into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_urem(rev![a, b]: &mut [EvmWord; 2]) {
    let divisor = b.to_u256();
    *b = if divisor.is_zero() { U256::ZERO } else { a.to_u256().wrapping_rem(divisor) }.into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_addmod(rev![a, b, c]: &mut [EvmWord; 3]) {
    *c = a.to_u256().add_mod(b.to_u256(), c.to_u256()).into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_mulmod(rev![a, b, c]: &mut [EvmWord; 3]) {
    *c = a.to_u256().mul_mod(b.to_u256(), c.to_u256()).into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_exp(
    ecx: &mut EvmContext<'_>,
    rev![base, exponent_ptr]: &mut [EvmWord; 2],
) -> InstructionResult {
    let exponent = exponent_ptr.to_u256();
    gas!(ecx, ecx.host.gas_params().exp_cost(exponent));
    *exponent_ptr = base.to_u256().pow(exponent).into();
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_keccak256(
    ecx: &mut EvmContext<'_>,
    rev![offset, len_ptr]: &mut [EvmWord; 2],
) -> InstructionResult {
    let len = try_into_usize!(len_ptr);
    *len_ptr = EvmWord::from_be_bytes(if len == 0 {
        KECCAK_EMPTY.0
    } else {
        gas!(ecx, ecx.host.gas_params().keccak256_cost(len));
        let offset = try_into_usize!(offset);
        ensure_memory!(ecx, offset, len);
        let data = ecx.memory.slice(offset..offset + len);
        revm_primitives::keccak256(&*data).0
    });
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_balance(
    ecx: &mut EvmContext<'_>,
    address: &mut EvmWord,
    spec_id: SpecId,
) -> InstructionResult {
    let addr = address.to_address();
    // Berlin+: use cold-load-skip optimization (mirrors revm's berlin_load_account! macro).
    // Pre-Berlin: no warm/cold model, use load_account_info_skip_cold_load with skip=false.
    if spec_id.is_enabled_in(SpecId::BERLIN) {
        let account = berlin_load_account!(ecx, addr, false);
        *address = account.balance.into();
    } else {
        let Ok(account) = ecx.host.load_account_info_skip_cold_load(addr, false, false) else {
            return InstructionResult::FatalExternalError;
        };
        *address = account.balance.into();
    }
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_origin(ecx: &mut EvmContext<'_>, slot: &mut EvmWord) {
    // In the Host trait, `caller()` returns the transaction origin
    let addr = ecx.host.caller();
    let mut word = [0u8; 32];
    word[12..32].copy_from_slice(addr.as_slice());
    *slot = EvmWord::from_be_bytes(word);
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_calldataload(
    ecx: &mut EvmContext<'_>,
    offset: &mut EvmWord,
) {
    let offset_usize = as_usize_saturated!(offset.to_u256());
    let mut word = [0u8; 32];

    match ecx.input.input() {
        CallInput::Bytes(bytes) => {
            let len = bytes.len().saturating_sub(offset_usize).min(32);
            if len > 0 && offset_usize < bytes.len() {
                word[..len].copy_from_slice(&bytes[offset_usize..offset_usize + len]);
            }
        }
        CallInput::SharedBuffer(range) => {
            let input_slice = ecx.memory.global_slice(range.clone());
            let input_len = input_slice.len();
            if offset_usize < input_len {
                let count = 32.min(input_len - offset_usize);
                word[..count].copy_from_slice(&input_slice[offset_usize..offset_usize + count]);
            }
        }
    }

    *offset = EvmWord::from_be_bytes(word);
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_calldatasize(ecx: &mut EvmContext<'_>) -> usize {
    match ecx.input.input() {
        CallInput::Bytes(bytes) => bytes.len(),
        CallInput::SharedBuffer(range) => range.len(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_calldatacopy(
    ecx: &mut EvmContext<'_>,
    rev![memory_offset, data_offset, len]: &mut [EvmWord; 3],
) -> InstructionResult {
    let len = try_into_usize!(len);
    if len != 0 {
        gas!(ecx, ecx.host.gas_params().copy_cost(len));
        let memory_offset = try_into_usize!(memory_offset);
        ensure_memory!(ecx, memory_offset, len);
        let data_offset = as_usize_saturated!(data_offset.to_u256());

        match ecx.input.input() {
            CallInput::Bytes(bytes) => {
                ecx.memory.set_data(memory_offset, data_offset, len, bytes.as_ref());
            }
            CallInput::SharedBuffer(range) => {
                ecx.memory.set_data_from_global(memory_offset, data_offset, len, range.clone());
            }
        }
    }
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_codecopy(
    ecx: &mut EvmContext<'_>,
    sp: &mut [EvmWord; 3],
) -> InstructionResult {
    let bytecode = unsafe { &*ecx.bytecode };
    copy_operation(ecx, sp, bytecode)
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_gas_price(ecx: &mut EvmContext<'_>, slot: &mut EvmWord) {
    *slot = ecx.host.effective_gas_price().into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_extcodesize(
    ecx: &mut EvmContext<'_>,
    address: &mut EvmWord,
    spec_id: SpecId,
) -> InstructionResult {
    let addr = address.to_address();
    if spec_id.is_enabled_in(SpecId::BERLIN) {
        let account = berlin_load_account!(ecx, addr, true);
        *address = U256::from(account.code.as_ref().unwrap().len()).into();
    } else {
        let Ok(account) = ecx.host.load_account_info_skip_cold_load(addr, true, false) else {
            return InstructionResult::FatalExternalError;
        };
        *address = U256::from(account.code.as_ref().unwrap().len()).into();
    }
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_extcodecopy(
    ecx: &mut EvmContext<'_>,
    rev![address, memory_offset, code_offset, len]: &mut [EvmWord; 4],
    spec_id: SpecId,
) -> InstructionResult {
    let addr = address.to_address();
    let len = try_into_usize!(len);
    gas!(ecx, ecx.host.gas_params().extcodecopy(len));

    let mut memory_offset_usize = 0;
    if len != 0 {
        memory_offset_usize = try_into_usize!(memory_offset);
        ensure_memory!(ecx, memory_offset_usize, len);
    }

    let code = if spec_id.is_enabled_in(SpecId::BERLIN) {
        let account = berlin_load_account!(ecx, addr, true);
        account.code.as_ref().unwrap().original_bytes()
    } else {
        let Some(code) = ecx.host.load_account_code(addr) else {
            return InstructionResult::FatalExternalError;
        };
        code.data
    };

    let code_offset_usize = core::cmp::min(as_usize_saturated!(code_offset.to_u256()), code.len());
    ecx.memory.set_data(memory_offset_usize, code_offset_usize, len, &code);
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_returndatacopy(
    ecx: &mut EvmContext<'_>,
    rev![memory_offset, offset, len]: &mut [EvmWord; 3],
) -> InstructionResult {
    let len = try_into_usize!(len);
    gas!(ecx, ecx.host.gas_params().copy_cost(len));
    let data_offset = offset.to_u256();
    let data_offset = as_usize_saturated!(data_offset);
    let (data_end, overflow) = data_offset.overflowing_add(len);
    if overflow || data_end > ecx.return_data.len() {
        return InstructionResult::OutOfOffset;
    }
    if len != 0 {
        let memory_offset = try_into_usize!(memory_offset);
        ensure_memory!(ecx, memory_offset, len);
        ecx.memory.set(memory_offset, &ecx.return_data[data_offset..data_end]);
    }
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_extcodehash(
    ecx: &mut EvmContext<'_>,
    address: &mut EvmWord,
    spec_id: SpecId,
) -> InstructionResult {
    let addr = address.to_address();
    let account = if spec_id.is_enabled_in(SpecId::BERLIN) {
        berlin_load_account!(ecx, addr, false)
    } else {
        let Ok(account) = ecx.host.load_account_info_skip_cold_load(addr, false, false) else {
            return InstructionResult::FatalExternalError;
        };
        account
    };
    let code_hash = if account.is_empty {
        revm_primitives::B256::ZERO
    } else {
        account.code_hash
    };
    *address = EvmWord::from_be_bytes(code_hash.0);
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_blockhash(
    ecx: &mut EvmContext<'_>,
    number_ptr: &mut EvmWord,
) -> InstructionResult {
    let requested_number = number_ptr.to_u256();
    let block_number = ecx.host.block_number();

    // Check if requested block is in the future
    let Some(diff) = block_number.checked_sub(requested_number) else {
        *number_ptr = EvmWord::ZERO;
        return InstructionResult::Stop;
    };

    let diff = as_u64_saturated!(diff);

    // Current block returns 0
    if diff == 0 {
        *number_ptr = EvmWord::ZERO;
        return InstructionResult::Stop;
    }

    // BLOCK_HASH_HISTORY is 256
    const BLOCK_HASH_HISTORY: u64 = 256;

    if diff <= BLOCK_HASH_HISTORY {
        let hash = try_host!(ecx.host.block_hash(as_u64_saturated!(requested_number)));
        *number_ptr = EvmWord::from_be_bytes(hash.0);
    } else {
        // Too old, return 0
        *number_ptr = EvmWord::ZERO;
    }

    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_coinbase(ecx: &mut EvmContext<'_>, slot: &mut EvmWord) {
    // In the Host trait, `beneficiary()` returns the coinbase address
    let addr = ecx.host.beneficiary();
    let mut word = [0u8; 32];
    word[12..32].copy_from_slice(addr.as_slice());
    *slot = EvmWord::from_be_bytes(word);
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_timestamp(ecx: &mut EvmContext<'_>, slot: &mut EvmWord) {
    *slot = ecx.host.timestamp().into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_number(ecx: &mut EvmContext<'_>, slot: &mut EvmWord) {
    *slot = ecx.host.block_number().into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_gaslimit(ecx: &mut EvmContext<'_>, slot: &mut EvmWord) {
    *slot = ecx.host.gas_limit().into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_chainid(ecx: &mut EvmContext<'_>, slot: &mut EvmWord) {
    *slot = ecx.host.chain_id().into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_basefee(ecx: &mut EvmContext<'_>, slot: &mut EvmWord) {
    *slot = ecx.host.basefee().into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_difficulty(
    ecx: &mut EvmContext<'_>,
    slot: &mut EvmWord,
    spec_id: SpecId,
) {
    *slot = if spec_id.is_enabled_in(SpecId::MERGE) {
        ecx.host.prevrandao().unwrap_or_default().into()
    } else {
        ecx.host.difficulty().into()
    };
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_self_balance(
    ecx: &mut EvmContext<'_>,
    slot: &mut EvmWord,
) -> InstructionResult {
    let state = try_host!(ecx.host.balance(ecx.input.target_address));
    *slot = state.data.into();
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_blob_hash(
    ecx: &mut EvmContext<'_>,
    index_ptr: &mut EvmWord,
) {
    let index = index_ptr.to_u256();
    let index_usize = as_usize_saturated!(index);
    *index_ptr = ecx.host.blob_hash(index_usize).unwrap_or_default().into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_blob_base_fee(
    ecx: &mut EvmContext<'_>,
    slot: &mut EvmWord,
) {
    *slot = ecx.host.blob_gasprice().into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_sload(
    ecx: &mut EvmContext<'_>,
    index: &mut EvmWord,
    spec_id: SpecId,
) -> InstructionResult {
    let address = ecx.input.target_address;
    let key = index.to_u256();
    if spec_id.is_enabled_in(SpecId::BERLIN) {
        let storage = berlin_sload!(ecx, address, key);
        *index = storage.data.into();
    } else {
        // Pre-Berlin: no cold-load-skip optimization, charge static gas manually
        // since revmc marks SLOAD as DYNAMIC with zero static gas.
        let Some(storage) = ecx.host.sload(address, key) else {
            return InstructionResult::FatalExternalError;
        };
        gas!(ecx, gas::sload_cost(spec_id, storage.is_cold));
        *index = storage.data.into();
    }
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_sstore(
    ecx: &mut EvmContext<'_>,
    rev![index, value]: &mut [EvmWord; 2],
    spec_id: SpecId,
) -> InstructionResult {
    ensure_non_staticcall!(ecx);

    let target = ecx.input.target_address;
    let is_istanbul = spec_id.is_enabled_in(SpecId::ISTANBUL);

    // EIP-2200: If gasleft is less than or equal to gas stipend, fail with OOG.
    if is_istanbul && ecx.gas.remaining() <= ecx.host.gas_params().call_stipend() {
        return InstructionResult::ReentrancySentryOOG;
    }

    gas!(ecx, ecx.host.gas_params().sstore_static_gas());

    let state_load = if spec_id.is_enabled_in(SpecId::BERLIN) {
        berlin_sstore!(ecx, target, index.to_u256(), value.to_u256())
    } else {
        let Some(load) = ecx.host.sstore(target, index.to_u256(), value.to_u256()) else {
            return InstructionResult::FatalExternalError;
        };
        load
    };

    let gp = ecx.host.gas_params();
    gas!(ecx, gp.sstore_dynamic_gas(is_istanbul, &state_load.data, state_load.is_cold));
    ecx.gas.record_refund(gp.sstore_refund(is_istanbul, &state_load.data));
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_msize(ecx: &mut EvmContext<'_>) -> usize {
    ecx.memory.len()
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_tstore(
    ecx: &mut EvmContext<'_>,
    rev![key, value]: &mut [EvmWord; 2],
) -> InstructionResult {
    ensure_non_staticcall!(ecx);
    ecx.host.tstore(ecx.input.target_address, key.to_u256(), value.to_u256());
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_tload(ecx: &mut EvmContext<'_>, key: &mut EvmWord) {
    *key = ecx.host.tload(ecx.input.target_address, key.to_u256()).into();
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_mcopy(
    ecx: &mut EvmContext<'_>,
    rev![dst, src, len]: &mut [EvmWord; 3],
) -> InstructionResult {
    let len = try_into_usize!(len);
    gas!(ecx, ecx.host.gas_params().mcopy_cost(len));
    if len != 0 {
        let dst = try_into_usize!(dst);
        let src = try_into_usize!(src);
        ensure_memory!(ecx, dst.max(src), len);
        ecx.memory.copy(dst, src, len);
    }
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_log(
    ecx: &mut EvmContext<'_>,
    sp: *mut EvmWord,
    n: u8,
) -> InstructionResult {
    ensure_non_staticcall!(ecx);
    assume!(n <= 4, "invalid log topic count: {n}");
    let sp = sp.add(n as usize);
    read_words!(sp, offset, len);
    let len = try_into_usize!(len);
    gas_opt!(ecx, gas::dyn_log_cost(len as u64));
    let data = if len != 0 {
        let offset = try_into_usize!(offset);
        ensure_memory!(ecx, offset, len);
        Bytes::copy_from_slice(&ecx.memory.slice(offset..offset + len))
    } else {
        Bytes::new()
    };

    let mut topics = Vec::with_capacity(n as usize);
    for i in 1..=n {
        topics.push(sp.sub(i as usize).read().to_be_bytes().into());
    }

    ecx.host.log(Log {
        address: ecx.input.target_address,
        data: LogData::new(topics, data).expect("too many topics"),
    });
    InstructionResult::Stop
}

// NOTE: Return `InstructionResult::Stop` here to indicate success, not the final result of
// the execution.

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_create(
    ecx: &mut EvmContext<'_>,
    sp: *mut EvmWord,
    spec_id: SpecId,
    create_kind: CreateKind,
) -> InstructionResult {
    ensure_non_staticcall!(ecx);

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
            let max_initcode_size = ecx.host.max_initcode_size();
            if len > max_initcode_size {
                return InstructionResult::CreateInitCodeSizeLimit;
            }
            gas!(ecx, ecx.host.gas_params().initcode_cost(len));
        }

        let code_offset = try_into_usize!(code_offset);
        ensure_memory!(ecx, code_offset, len);
        Bytes::copy_from_slice(&ecx.memory.slice(code_offset..code_offset + len))
    } else {
        Bytes::new()
    };

    let is_create2 = create_kind == CreateKind::Create2;
    let gp = ecx.host.gas_params();
    gas!(ecx, if is_create2 { gp.create2_cost(len) } else { gp.create_cost() });

    let scheme = if is_create2 {
        pop!(sp; salt);
        CreateScheme::Create2 { salt: salt.to_u256() }
    } else {
        CreateScheme::Create
    };

    let mut gas_limit = ecx.gas.remaining();
    if spec_id.is_enabled_in(SpecId::TANGERINE) {
        gas_limit = ecx.host.gas_params().call_stipend_reduction(gas_limit);
    }
    gas!(ecx, gas_limit);

    *ecx.next_action =
        Some(InterpreterAction::NewFrame(revm_interpreter::FrameInput::Create(Box::new(
            CreateInputs::new(ecx.input.target_address, scheme, value.to_u256(), code, gas_limit),
        ))));

    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_call(
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
            if call_kind == CallKind::Call && ecx.is_static && value != U256::ZERO {
                return InstructionResult::CallNotAllowedInsideStatic;
            }
            value
        }
        CallKind::DelegateCall | CallKind::StaticCall => U256::ZERO,
    };
    let transfers_value = value != U256::ZERO;

    pop!(sp; in_offset, in_len, out_offset, out_len);

    let in_len = try_into_usize!(in_len);
    let input = if in_len != 0 {
        let in_offset = try_into_usize!(in_offset);
        ensure_memory!(ecx, in_offset, in_len);
        Bytes::copy_from_slice(&ecx.memory.slice(in_offset..in_offset + in_len))
    } else {
        Bytes::new()
    };

    let out_len = try_into_usize!(out_len);
    let out_offset = if out_len != 0 {
        let out_offset = try_into_usize!(out_offset);
        ensure_memory!(ecx, out_offset, out_len);
        out_offset
    } else {
        usize::MAX // unrealistic value so we are sure it is not used
    };

    // Charge the CALL base access cost up-front. In the interpreter this is charged as static
    // opcode gas before entering call helpers; revmc marks CALL as dynamic, so the builtin must
    // do it.
    gas!(ecx, ecx.host.gas_params().warm_storage_read_cost());

    if transfers_value {
        gas!(ecx, ecx.host.gas_params().transfer_value_cost());
    }

    // Match interpreter call path: load delegated account and pass resolved bytecode/hash
    // through CallInputs::known_bytecode (covers EIP-7702 delegation and EOF execution).
    let (dynamic_gas, bytecode, code_hash) =
        match revm_interpreter::instructions::contract::load_account_delegated(
            ecx.host,
            spec_id,
            ecx.gas.remaining(),
            to,
            transfers_value,
            call_kind == CallKind::Call,
        ) {
            Ok(out) => out,
            Err(revm_context_interface::host::LoadError::ColdLoadSkipped) => {
                return InstructionResult::OutOfGas;
            }
            Err(revm_context_interface::host::LoadError::DBError) => {
                return InstructionResult::FatalExternalError;
            }
        };

    gas!(ecx, dynamic_gas);

    // EIP-150: Gas cost changes for IO-heavy operations
    let mut gas_limit = if spec_id.is_enabled_in(SpecId::TANGERINE) {
        let gas = ecx.gas.remaining();
        ecx.host.gas_params().call_stipend_reduction(gas).min(local_gas_limit)
    } else {
        local_gas_limit
    };

    gas!(ecx, gas_limit);

    // Add call stipend if there is value to be transferred.
    if matches!(call_kind, CallKind::Call | CallKind::CallCode) && transfers_value {
        gas_limit = gas_limit.saturating_add(ecx.host.gas_params().call_stipend());
    }

    *ecx.next_action = Some(InterpreterAction::NewFrame(revm_interpreter::FrameInput::Call(
        Box::new(CallInputs {
            input: CallInput::Bytes(input),
            return_memory_offset: out_offset..out_offset + out_len,
            gas_limit,
            bytecode_address: to,
            known_bytecode: Some((code_hash, bytecode)),
            target_address: if matches!(call_kind, CallKind::DelegateCall | CallKind::CallCode) {
                ecx.input.target_address
            } else {
                to
            },
            caller: if call_kind == CallKind::DelegateCall {
                ecx.input.caller_address
            } else {
                ecx.input.target_address
            },
            value: if call_kind == CallKind::DelegateCall {
                CallValue::Apparent(ecx.input.call_value)
            } else {
                CallValue::Transfer(value)
            },
            scheme: call_kind.into(),
            is_static: ecx.is_static || call_kind == CallKind::StaticCall,
        }),
    )));

    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_do_return(
    ecx: &mut EvmContext<'_>,
    rev![offset, len]: &mut [EvmWord; 2],
    result: InstructionResult,
) -> InstructionResult {
    let len = try_into_usize!(len);
    let output = if len != 0 {
        let offset = try_into_usize!(offset);
        ensure_memory!(ecx, offset, len);
        ecx.memory.slice(offset..offset + len).to_vec().into()
    } else {
        Bytes::new()
    };
    *ecx.next_action =
        Some(InterpreterAction::Return(InterpreterResult { output, gas: *ecx.gas, result }));
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_selfdestruct(
    ecx: &mut EvmContext<'_>,
    target: &mut EvmWord,
    spec_id: SpecId,
) -> InstructionResult {
    ensure_non_staticcall!(ecx);

    let cold_load_gas = ecx.host.gas_params().selfdestruct_cold_cost();
    let skip_cold_load = ecx.gas.remaining() < cold_load_gas;
    let res = match ecx.host.selfdestruct(ecx.input.target_address, target.to_address(), skip_cold_load) {
        Ok(r) => r,
        Err(revm_context_interface::host::LoadError::ColdLoadSkipped) => {
            return InstructionResult::OutOfGas;
        }
        Err(revm_context_interface::host::LoadError::DBError) => {
            return InstructionResult::FatalExternalError;
        }
    };

    // EIP-161: State trie clearing (invariant-preserving alternative)
    let should_charge_topup = if spec_id.is_enabled_in(SpecId::SPURIOUS_DRAGON) {
        res.data.had_value && !res.data.target_exists
    } else {
        !res.data.target_exists
    };

    gas!(ecx, ecx.host.gas_params().selfdestruct_cost(should_charge_topup, res.is_cold));

    if !res.data.previously_destroyed {
        ecx.gas.record_refund(ecx.host.gas_params().selfdestruct_refund());
    }

    InstructionResult::SelfDestruct
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_resize_memory(
    ecx: &mut EvmContext<'_>,
    new_size: usize,
) -> InstructionResult {
    resize_memory(ecx, new_size)
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_mload(
    ecx: &mut EvmContext<'_>,
    rev![offset_ptr]: &mut [EvmWord; 1],
) -> InstructionResult {
    let offset = try_into_usize!(offset_ptr);
    ensure_memory!(ecx, offset, 32);
    let slice = ecx.memory.slice(offset..offset + 32);
    let mut word = [0u8; 32];
    word.copy_from_slice(&slice);
    *offset_ptr = EvmWord::from_be_bytes(word);
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_mstore(
    ecx: &mut EvmContext<'_>,
    rev![offset, value]: &mut [EvmWord; 2],
) -> InstructionResult {
    let offset = try_into_usize!(offset);
    ensure_memory!(ecx, offset, 32);
    ecx.memory.set(offset, &value.to_be_bytes());
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_mstore8(
    ecx: &mut EvmContext<'_>,
    rev![offset, value]: &mut [EvmWord; 2],
) -> InstructionResult {
    let offset = try_into_usize!(offset);
    ensure_memory!(ecx, offset, 1);
    let byte = value.to_be_bytes()[31];
    ecx.memory.set(offset, &[byte]);
    InstructionResult::Stop
}

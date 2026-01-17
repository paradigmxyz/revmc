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
    as_u64_saturated, as_usize_saturated, 
    interpreter_types::{InputsTr, MemoryTr},
    CallInput, CallInputs, CallScheme, CallValue, CreateInputs,
    CreateScheme, InstructionResult, InterpreterAction, InterpreterResult,
};
use revm_primitives::{
    hardfork::SpecId, Address, Bytes, Log, LogData, KECCAK_EMPTY, U256,
};
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

/// The result of a `EXT*CALL` instruction if the gas limit is less than `MIN_CALLEE_GAS`.
// NOTE: This is just a random value that cannot happen normally.
pub const EXTCALL_LIGHT_FAILURE: InstructionResult = InstructionResult::PrecompileError;

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

/// The kind of a `EXT*CALL` instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ExtCallKind {
    /// `EXTCALL`.
    Call,
    /// `EXTDELEGATECALL`.
    DelegateCall,
    /// `EXTSTATICCALL`.
    StaticCall,
}

impl From<ExtCallKind> for CallScheme {
    fn from(kind: ExtCallKind) -> Self {
        // ExtCall variants map to regular call schemes in revm v34
        match kind {
            ExtCallKind::Call => Self::Call,
            ExtCallKind::DelegateCall => Self::DelegateCall,
            ExtCallKind::StaticCall => Self::StaticCall,
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
    spec_id: SpecId,
) -> InstructionResult {
    let exponent = exponent_ptr.to_u256();
    gas_opt!(ecx, gas::dyn_exp_cost(spec_id, exponent));
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
        gas_opt!(ecx, gas::dyn_keccak256_cost(len as u64));
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
    let state = try_host!(ecx.host.balance(address.to_address()));
    *address = state.data.into();
    let gas = if spec_id.is_enabled_in(SpecId::BERLIN) {
        gas::warm_cold_cost(state.is_cold)
    } else if spec_id.is_enabled_in(SpecId::ISTANBUL) {
        // EIP-1884: Repricing for trie-size-dependent opcodes
        700
    } else if spec_id.is_enabled_in(SpecId::TANGERINE) {
        400
    } else {
        20
    };
    gas!(ecx, gas);
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
pub unsafe extern "C" fn __revmc_builtin_calldataload(ecx: &mut EvmContext<'_>, offset: &mut EvmWord) {
    let offset_usize = as_usize_saturated!(offset.to_u256());
    match ecx.input.input() {
        CallInput::Bytes(bytes) => {
            let mut word = [0u8; 32];
            let len = bytes.len().saturating_sub(offset_usize).min(32);
            if len > 0 && offset_usize < bytes.len() {
                word[..len].copy_from_slice(&bytes[offset_usize..offset_usize + len]);
            }
            *offset = EvmWord::from_be_bytes(word);
        }
        CallInput::SharedBuffer(_range) => {
            // SharedBuffer requires access to LocalContext which we don't have
            *offset = EvmWord::ZERO;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_calldatasize(ecx: &mut EvmContext<'_>) -> usize {
    match ecx.input.input() {
        CallInput::Bytes(bytes) => bytes.len(),
        CallInput::SharedBuffer(range) => range.len(),
    }
}

// TODO: CallInput is now an enum (Bytes or SharedBuffer), needs proper handling
#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_calldatacopy(
    ecx: &mut EvmContext<'_>,
    sp: &mut [EvmWord; 3],
) -> InstructionResult {
    match ecx.input.input() {
        CallInput::Bytes(bytes) => {
            let data = decouple_lt(&bytes[..]);
            copy_operation(ecx, sp, data)
        }
        CallInput::SharedBuffer(_range) => {
            // SharedBuffer requires access to LocalContext which we don't have
            // For now, treat as empty
            copy_operation(ecx, sp, &[])
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_codesize(ecx: &mut EvmContext<'_>) -> usize {
    ecx.bytecode_len
}

// TODO: CODECOPY needs access to bytecode slice which is not available in EvmContext
// due to borrow conflicts. For now, return empty data.
#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_codecopy(
    ecx: &mut EvmContext<'_>,
    sp: &mut [EvmWord; 3],
) -> InstructionResult {
    // Bytecode slice is not available - this is a limitation of the current context design
    copy_operation(ecx, sp, &[])
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
    let state_load = try_opt!(ecx.host.load_account_code(address.to_address()));
    *address = U256::from(state_load.data.len()).into();
    let gas = if spec_id.is_enabled_in(SpecId::BERLIN) {
        gas::warm_cold_cost(state_load.is_cold)
    } else if spec_id.is_enabled_in(SpecId::TANGERINE) {
        700
    } else {
        20
    };
    gas!(ecx, gas);
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_extcodecopy(
    ecx: &mut EvmContext<'_>,
    rev![address, memory_offset, code_offset, len]: &mut [EvmWord; 4],
    spec_id: SpecId,
) -> InstructionResult {
    let state_load = try_opt!(ecx.host.load_account_code(address.to_address()));

    let len = try_into_usize!(len);
    gas_opt!(ecx, gas::extcodecopy_cost(spec_id, len as u64, state_load.is_cold));
    if len != 0 {
        let memory_offset = try_into_usize!(memory_offset);
        let code_offset = code_offset.to_u256();
        let code_offset = as_usize_saturated!(code_offset).min(state_load.data.len());
        ensure_memory!(ecx, memory_offset, len);
        ecx.memory.set_data(memory_offset, code_offset, len, &state_load.data);
    }
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_returndatacopy(
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
    let state_load = try_opt!(ecx.host.load_account_code_hash(address.to_address()));
    *address = EvmWord::from_be_bytes(state_load.data.0);
    let gas = if spec_id.is_enabled_in(SpecId::BERLIN) {
        gas::warm_cold_cost(state_load.is_cold)
    } else if spec_id.is_enabled_in(SpecId::ISTANBUL) {
        700
    } else {
        400
    };
    gas!(ecx, gas);
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_blockhash(
    ecx: &mut EvmContext<'_>,
    number_ptr: &mut EvmWord,
) -> InstructionResult {
    let hash = try_host!(ecx.host.block_hash(as_u64_saturated!(number_ptr.to_u256())));
    *number_ptr = EvmWord::from_be_bytes(hash.0);
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
    let state = try_opt!(ecx.host.sload(address, index.to_u256()));
    gas!(ecx, gas::sload_cost(spec_id, state.is_cold));
    *index = state.data.into();
    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_sstore(
    ecx: &mut EvmContext<'_>,
    rev![index, value]: &mut [EvmWord; 2],
    spec_id: SpecId,
) -> InstructionResult {
    ensure_non_staticcall!(ecx);

    let state =
        try_opt!(ecx.host.sstore(ecx.input.target_address, index.to_u256(), value.to_u256()));

    gas_opt!(ecx, gas::sstore_cost(spec_id, &state.data, ecx.gas.remaining(), state.is_cold));
    ecx.gas.record_refund(gas::sstore_refund(spec_id, &state.data));
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
    gas_opt!(ecx, gas::dyn_verylowcopy_cost(len as u64));
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
        Bytes::copy_from_slice(&*ecx.memory.slice(offset..offset + len))
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

// TODO: EOF data access needs to be updated for revm v34
pub unsafe extern "C" fn __revmc_builtin_data_load(_ecx: &mut EvmContext<'_>, slot: &mut EvmWord) {
    // EOF bytecode data access is not currently supported
    *slot = EvmWord::ZERO;
}

// TODO: EOF data access needs to be updated for revm v34
pub unsafe extern "C" fn __revmc_builtin_data_copy(
    _ecx: &mut EvmContext<'_>,
    _sp: &mut [EvmWord; 3],
) -> InstructionResult {
    // EOF bytecode data access is not currently supported
    InstructionResult::NotActivated
}

pub unsafe extern "C" fn __revmc_builtin_returndataload(
    ecx: &mut EvmContext<'_>,
    slot: &mut EvmWord,
) {
    let offset = as_usize_saturated!(slot.to_u256());
    let mut output = [0u8; 32];
    if let Some(available) = ecx.return_data.len().checked_sub(offset) {
        let copy_len = available.min(32);
        output[..copy_len].copy_from_slice(&ecx.return_data[offset..offset + copy_len]);
    }
    *slot = EvmWord::from_be_bytes(output);
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
            gas!(ecx, gas::initcode_cost(len as u64));
        }

        let code_offset = try_into_usize!(code_offset);
        ensure_memory!(ecx, code_offset, len);
        Bytes::copy_from_slice(&*ecx.memory.slice(code_offset..code_offset + len))
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

    *ecx.next_action = Some(InterpreterAction::NewFrame(
        revm_interpreter::FrameInput::Create(Box::new(CreateInputs::new(
            ecx.input.target_address,
            scheme,
            value.to_u256(),
            code,
            gas_limit,
        ))),
    ));

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
        Bytes::copy_from_slice(&*ecx.memory.slice(in_offset..in_offset + in_len))
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

    // Load account and calculate gas cost.
    let mut account_load = try_host!(ecx.host.load_account_delegated(to));

    if call_kind != CallKind::Call {
        account_load.is_empty = false;
    }

    gas!(ecx, gas::call_cost(spec_id, transfers_value, account_load));

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
    if matches!(call_kind, CallKind::Call | CallKind::CallCode) && transfers_value {
        gas_limit = gas_limit.saturating_add(gas::CALL_STIPEND);
    }

    *ecx.next_action = Some(InterpreterAction::NewFrame(
        revm_interpreter::FrameInput::Call(Box::new(CallInputs {
            input: CallInput::Bytes(input),
            return_memory_offset: out_offset..out_offset + out_len,
            gas_limit,
            bytecode_address: to,
            known_bytecode: None,
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
        })),
    ));

    InstructionResult::Stop
}

#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_ext_call(
    ecx: &mut EvmContext<'_>,
    sp: *mut EvmWord,
    call_kind: ExtCallKind,
    spec_id: SpecId,
) -> InstructionResult {
    let (target_address, in_offset, in_len, value) = if call_kind == ExtCallKind::Call {
        let rev![target_address, in_offset, in_len, value] = &mut *sp.cast::<[EvmWord; 4]>();
        (target_address, in_offset, in_len, value.to_u256())
    } else {
        let rev![target_address, in_offset, in_len] = &mut *sp.cast::<[EvmWord; 3]>();
        (target_address, in_offset, in_len, U256::ZERO)
    };

    let target_address_bytes = target_address.to_be_bytes();
    let (pad, target_address) = target_address_bytes.split_last_chunk::<20>().unwrap();
    if !pad.iter().all(|i| *i == 0) {
        // Invalid EXTCALL target - address has non-zero padding bytes
        return InstructionResult::Revert;
    }
    let target_address = Address::new(*target_address);

    let in_len = try_into_usize!(in_len);
    let input = if in_len != 0 {
        let in_offset = try_into_usize!(in_offset);
        ensure_memory!(ecx, in_offset, in_len);
        Bytes::copy_from_slice(&*ecx.memory.slice(in_offset..in_offset + in_len))
    } else {
        Bytes::new()
    };

    let transfers_value = value != U256::ZERO;
    if ecx.is_static && transfers_value {
        return InstructionResult::CallNotAllowedInsideStatic;
    }

    let Some(account_load) = ecx.host.load_account_delegated(target_address) else {
        return InstructionResult::FatalExternalError;
    };
    let call_cost = gas::call_cost(spec_id, transfers_value, account_load);
    gas!(ecx, call_cost);

    let gas_reduce = core::cmp::max(ecx.gas.remaining() / 64, 5000);
    let gas_limit = ecx.gas.remaining().saturating_sub(gas_reduce);
    if gas_limit < gas::MIN_CALLEE_GAS {
        ecx.return_data = &[];
        return EXTCALL_LIGHT_FAILURE;
    }
    gas!(ecx, gas_limit);

    // Call host to interact with target contract
    *ecx.next_action = Some(InterpreterAction::NewFrame(
        revm_interpreter::FrameInput::Call(Box::new(CallInputs {
            input: CallInput::Bytes(input),
            gas_limit,
            target_address: if call_kind == ExtCallKind::DelegateCall {
                ecx.input.target_address
            } else {
                target_address
            },
            caller: if call_kind == ExtCallKind::DelegateCall {
                ecx.input.caller_address
            } else {
                ecx.input.target_address
            },
            bytecode_address: target_address,
            known_bytecode: None,
            value: if call_kind == ExtCallKind::DelegateCall {
                CallValue::Apparent(ecx.input.call_value)
            } else {
                CallValue::Transfer(value)
            },
            scheme: call_kind.into(),
            is_static: ecx.is_static || call_kind == ExtCallKind::StaticCall,
            return_memory_offset: 0..0,
        })),
    ));
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

    let res = match ecx.host.selfdestruct(ecx.input.target_address, target.to_address(), false) {
        Ok(r) => r,
        Err(_) => return InstructionResult::FatalExternalError,
    };

    // EIP-3529: Reduction in refunds
    if !spec_id.is_enabled_in(SpecId::LONDON) && !res.data.previously_destroyed {
        ecx.gas.record_refund(gas::SELFDESTRUCT);
    }
    gas!(ecx, gas::selfdestruct_cost(spec_id, res));

    InstructionResult::Stop
}



#[no_mangle]
pub unsafe extern "C" fn __revmc_builtin_resize_memory(
    ecx: &mut EvmContext<'_>,
    new_size: usize,
) -> InstructionResult {
    resize_memory(ecx, new_size)
}

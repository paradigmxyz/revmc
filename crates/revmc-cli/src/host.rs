use revm::state::AccountInfo;
use revm_interpreter::{Host, SStoreResult, SelfDestructResult, StateLoad, host::LoadError};
use revm_primitives::{Address, B256, Log, StorageKey, StorageValue, U256};
use revmc::{
    context_interface::{cfg::GasParams, journaled_state::AccountInfoLoad},
    primitives::hardfork::SpecId,
};
use std::collections::HashMap;

/// Minimal [`Host`] with in-memory storage, suitable for benchmarking bytecode
/// that uses SLOAD/SSTORE without needing a full EVM database.
#[allow(missing_debug_implementations)]
pub struct BenchHost {
    gas_params: GasParams,
    storage: HashMap<(Address, StorageKey), StorageValue>,
}

impl BenchHost {
    pub fn new(spec_id: SpecId) -> Self {
        Self { gas_params: GasParams::new_spec(spec_id), storage: HashMap::new() }
    }
}

impl Host for BenchHost {
    fn basefee(&self) -> U256 {
        U256::ZERO
    }
    fn blob_gasprice(&self) -> U256 {
        U256::ZERO
    }
    fn gas_limit(&self) -> U256 {
        U256::MAX
    }
    fn difficulty(&self) -> U256 {
        U256::ZERO
    }
    fn prevrandao(&self) -> Option<U256> {
        None
    }
    fn block_number(&self) -> U256 {
        U256::ZERO
    }
    fn timestamp(&self) -> U256 {
        U256::ZERO
    }
    fn beneficiary(&self) -> Address {
        Address::ZERO
    }
    fn slot_num(&self) -> U256 {
        U256::ZERO
    }
    fn chain_id(&self) -> U256 {
        U256::from(1)
    }
    fn effective_gas_price(&self) -> U256 {
        U256::ZERO
    }
    fn caller(&self) -> Address {
        Address::ZERO
    }
    fn blob_hash(&self, _number: usize) -> Option<U256> {
        None
    }
    fn max_initcode_size(&self) -> usize {
        usize::MAX
    }
    fn gas_params(&self) -> &GasParams {
        &self.gas_params
    }
    fn block_hash(&mut self, _number: u64) -> Option<B256> {
        Some(B256::ZERO)
    }
    fn log(&mut self, _log: Log) {}
    fn tstore(&mut self, _address: Address, _key: StorageKey, _value: StorageValue) {}
    fn tload(&mut self, _address: Address, _key: StorageKey) -> StorageValue {
        StorageValue::ZERO
    }

    fn selfdestruct(
        &mut self,
        _address: Address,
        _target: Address,
        _skip_cold_load: bool,
    ) -> Result<StateLoad<SelfDestructResult>, LoadError> {
        Ok(StateLoad::new(Default::default(), false))
    }

    fn load_account_info_skip_cold_load(
        &mut self,
        _address: Address,
        _load_code: bool,
        _skip_cold_load: bool,
    ) -> Result<AccountInfoLoad<'_>, LoadError> {
        static ACCOUNT: AccountInfo = AccountInfo {
            balance: U256::ZERO,
            nonce: 0,
            code_hash: B256::ZERO,
            code: None,
            account_id: None,
        };
        Ok(AccountInfoLoad::new(&ACCOUNT, false, false))
    }

    fn sload_skip_cold_load(
        &mut self,
        address: Address,
        key: StorageKey,
        _skip_cold_load: bool,
    ) -> Result<StateLoad<StorageValue>, LoadError> {
        let value = self.storage.get(&(address, key)).copied().unwrap_or(StorageValue::ZERO);
        Ok(StateLoad::new(value, false))
    }

    fn sstore_skip_cold_load(
        &mut self,
        address: Address,
        key: StorageKey,
        value: StorageValue,
        _skip_cold_load: bool,
    ) -> Result<StateLoad<SStoreResult>, LoadError> {
        let old = self.storage.insert((address, key), value).unwrap_or(StorageValue::ZERO);
        Ok(StateLoad::new(
            SStoreResult { original_value: old, present_value: old, new_value: value },
            false,
        ))
    }
}

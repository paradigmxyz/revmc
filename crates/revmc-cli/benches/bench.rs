#![allow(missing_docs)]

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use revm_bytecode::Bytecode;
use revm_interpreter::{
    InputsImpl, SharedMemory,
    host::LoadError,
    instruction_table,
    interpreter::{EthInterpreter, ExtBytecode},
};
use revm_primitives::{Address, B256, Log, StorageKey, StorageValue, U256};
use revmc::{
    EvmCompiler, EvmCompilerFn, EvmContext, EvmLlvmBackend, EvmStack, OptimizationLevel,
    primitives::hardfork::SpecId,
};
use revmc_cli::Bench;
use std::{collections::HashMap, time::Duration};

const SPEC_ID: SpecId = SpecId::OSAKA;

fn bench(c: &mut Criterion) {
    for bench in &revmc_cli::get_benches() {
        run_bench(c, bench);
    }
}

fn run_bench(c: &mut Criterion, bench: &Bench) {
    let Bench { name, bytecode, calldata, stack_input, native } = bench;

    let mut g = c.benchmark_group(*name);
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(1));
    g.measurement_time(Duration::from_secs(5));

    let gas_limit = u64::MAX / 2;
    let calldata: revmc::primitives::Bytes = calldata.clone().into();
    let bytecode_raw = Bytecode::new_raw(revmc::primitives::Bytes::copy_from_slice(bytecode));

    // ── Compile-time ────────────────────────────────────────────────────
    // Skip snailtracer compile-time benchmarks because they're too slow.

    if *name != "snailtracer" {
        g.bench_function("compile/translate", |b| {
            b.iter_batched(
                || new_compiler(OptimizationLevel::None),
                |mut compiler| {
                    compiler.translate(name, bytecode.as_slice(), SPEC_ID).unwrap();
                },
                BatchSize::PerIteration,
            )
        });

        g.bench_function("compile/jit", |b| {
            b.iter_batched(
                || {
                    let mut compiler = new_compiler(OptimizationLevel::Aggressive);
                    let id =
                        compiler.translate(name, bytecode.as_slice(), SPEC_ID).expect("translate");
                    (compiler, id)
                },
                |(mut compiler, id)| unsafe {
                    compiler.jit_function(id).unwrap();
                },
                BatchSize::PerIteration,
            )
        });
    }

    // ── Runtime ─────────────────────────────────────────────────────────

    let mut host = BenchHost::new(SPEC_ID);
    let table = instruction_table::<EthInterpreter, BenchHost>();

    let opt_level = revmc::OptimizationLevel::Aggressive;
    let backend = EvmLlvmBackend::new(false, opt_level).unwrap();
    let mut compiler = EvmCompiler::new(backend);
    compiler.inspect_stack_length(!stack_input.is_empty());
    compiler.gas_metering(true);

    if let Some(native) = *native {
        g.bench_function("native", |b| b.iter(native));
    }

    let mut stack = EvmStack::new();
    let mut call_jit = |f: EvmCompilerFn| {
        for (i, input) in stack_input.iter().enumerate() {
            stack.as_mut_slice()[i] = (*input).into();
        }
        let mut stack_len = stack_input.len();

        let ext_bytecode = ExtBytecode::new(bytecode_raw.clone());
        let input = InputsImpl {
            input: revm_interpreter::CallInput::Bytes(calldata.clone()),
            ..Default::default()
        };
        let mut interpreter = revm_interpreter::Interpreter::new(
            SharedMemory::new(),
            ext_bytecode,
            input,
            false,
            SPEC_ID,
            gas_limit,
        );
        let mut ecx = EvmContext::from_interpreter(&mut interpreter, &mut host);

        unsafe { f.call(Some(&mut stack), Some(&mut stack_len), &mut ecx) }
    };

    let jit_matrix = [("default", (true, true)), ("no_gas", (false, true))];
    let jit_ids = jit_matrix.map(|(name, (gas, stack))| {
        compiler.gas_metering(gas);
        unsafe { compiler.stack_bound_checks(stack) };
        (name, compiler.translate(name, bytecode_raw.original_byte_slice(), SPEC_ID).expect(name))
    });
    for &(name, fn_id) in &jit_ids {
        let jit = unsafe { compiler.jit_function(fn_id) }.expect(name);
        g.bench_function(format!("revmc/{name}"), |b| b.iter(|| call_jit(jit)));
    }

    g.bench_function("revm-interpreter", |b| {
        b.iter(|| {
            let ext_bytecode = ExtBytecode::new(bytecode_raw.clone());
            let input = InputsImpl {
                input: revm_interpreter::CallInput::Bytes(calldata.clone()),
                ..Default::default()
            };
            let mut interpreter = revm_interpreter::Interpreter::new(
                SharedMemory::new(),
                ext_bytecode,
                input,
                false,
                SPEC_ID,
                gas_limit,
            );

            interpreter.stack.data_mut().extend_from_slice(stack_input);

            let action = interpreter.run_plain(&table, &mut host);
            let result =
                action.instruction_result().unwrap_or(revm_interpreter::InstructionResult::Stop);
            assert!(result.is_ok(), "Interpreter failed with {result:?}");
            assert!(action.is_return(), "Interpreter bad action: {action:?}");
            action
        })
    });

    g.finish();
}

fn new_compiler(opt_level: OptimizationLevel) -> EvmCompiler<EvmLlvmBackend> {
    let backend = EvmLlvmBackend::new(false, opt_level).unwrap();
    EvmCompiler::new(backend)
}

// ── Minimal Host with storage support ───────────────────────────────────────

use revm_context_interface::cfg::GasParams;
use revm_interpreter::{Host, SStoreResult, SelfDestructResult, StateLoad};

struct BenchHost {
    gas_params: GasParams,
    storage: HashMap<(Address, StorageKey), StorageValue>,
}

impl BenchHost {
    fn new(spec_id: SpecId) -> Self {
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
    ) -> Result<revm_context_interface::journaled_state::AccountInfoLoad<'_>, LoadError> {
        use revm::state::AccountInfo;
        use revm_context_interface::journaled_state::AccountInfoLoad;
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

criterion_group!(benches, bench);
criterion_main!(benches);

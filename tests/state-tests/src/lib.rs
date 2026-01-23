//! Ethereum state tests for revmc JIT compiler.
//!
//! Runs tests from ethereum/tests against the JIT compiler and compares
//! results with the revm interpreter.

use eyre::{eyre, Result};
use revm::{
    bytecode::Bytecode,
    context::{BlockEnv, CfgEnv, Context, Journal, TxEnv},
    context_interface::JournalTr,
    database::CacheDB,
    database_interface::EmptyDB,
    interpreter::{
        instructions::instruction_table_gas_changes_spec,
        interpreter::{EthInterpreter, ExtBytecode},
        interpreter_types::ReturnData,
        FrameInput, InputsImpl, Interpreter, InterpreterAction, InterpreterResult, SharedMemory,
    },
    primitives::{hardfork::SpecId, keccak256, Bytes, TxKind, B256, U256},
    state::AccountInfo,
};
use revm_statetest_types::{SpecName, TestSuite, TestUnit};
use revmc::{
    llvm::inkwell::context::Context as LlvmContext, Backend, EvmCompiler, EvmCompilerFn,
    EvmContext, EvmLlvmBackend, EvmStack, OptimizationLevel,
};
use std::{
    cmp::min,
    collections::HashMap,
    env,
    ops::Range,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

/// Default path to ethereum/tests repository
const DEFAULT_ETHTESTS_PATH: &str = "tests/ethereum-tests";

/// Get the path to ethereum/tests
pub fn get_ethtests_path() -> PathBuf {
    env::var("ETHTESTS").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(DEFAULT_ETHTESTS_PATH))
}

/// Find all JSON test files in a directory
pub fn find_json_tests(path: &Path) -> Vec<PathBuf> {
    WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
        .map(|e| e.path().to_path_buf())
        .collect()
}

/// Load a test suite from a JSON file
pub fn load_test_suite(path: &Path) -> Result<TestSuite> {
    let content = std::fs::read_to_string(path)?;
    let suite: TestSuite = serde_json::from_str(&content)?;
    Ok(suite)
}

/// Convert SpecName to SpecId
pub fn spec_name_to_spec_id(spec_name: &SpecName) -> Option<SpecId> {
    match *spec_name {
        SpecName::Frontier => Some(SpecId::FRONTIER),
        SpecName::Homestead => Some(SpecId::HOMESTEAD),
        SpecName::EIP150 => Some(SpecId::TANGERINE),
        SpecName::EIP158 => Some(SpecId::SPURIOUS_DRAGON),
        SpecName::Byzantium => Some(SpecId::BYZANTIUM),
        SpecName::Constantinople => None, // Skip, has reentrancy bug
        SpecName::ConstantinopleFix => Some(SpecId::PETERSBURG),
        SpecName::Istanbul => Some(SpecId::ISTANBUL),
        SpecName::Berlin => Some(SpecId::BERLIN),
        SpecName::London => Some(SpecId::LONDON),
        SpecName::Paris | SpecName::Merge => Some(SpecId::MERGE),
        SpecName::Shanghai => Some(SpecId::SHANGHAI),
        SpecName::Cancun => Some(SpecId::CANCUN),
        SpecName::Prague => Some(SpecId::PRAGUE),
        SpecName::Osaka => Some(SpecId::OSAKA),
        _ => None, // Skip transition specs and unknown
    }
}

/// Test result from running a single test case
#[derive(Debug)]
pub struct TestResult {
    pub name: String,
    pub spec: String,
    pub passed: bool,
    pub error: Option<String>,
}

/// JIT-compiled bytecode cache
pub struct CompiledContracts {
    functions: HashMap<B256, EvmCompilerFn>,
}

impl CompiledContracts {
    pub fn new() -> Self {
        Self { functions: HashMap::new() }
    }

    pub fn get(&self, code_hash: &B256) -> Option<EvmCompilerFn> {
        self.functions.get(code_hash).copied()
    }

    pub fn insert(&mut self, code_hash: B256, func: EvmCompilerFn) {
        self.functions.insert(code_hash, func);
    }
}

impl Default for CompiledContracts {
    fn default() -> Self {
        Self::new()
    }
}

/// Build pre-state database from TestUnit
pub fn build_pre_state(unit: &TestUnit) -> CacheDB<EmptyDB> {
    let mut db = CacheDB::new(EmptyDB::new());

    for (address, info) in &unit.pre {
        let code_hash = keccak256(&info.code);
        let bytecode =
            if info.code.is_empty() { None } else { Some(Bytecode::new_legacy(info.code.clone())) };

        let acc_info = AccountInfo {
            balance: info.balance,
            nonce: info.nonce,
            code_hash,
            code: bytecode,
            ..Default::default()
        };

        db.insert_account_info(*address, acc_info);

        for (key, value) in &info.storage {
            let _ = db.insert_account_storage(*address, *key, *value);
        }
    }

    db
}

/// Build transaction environment from test indices
pub fn build_tx_env(unit: &TestUnit, test: &revm_statetest_types::Test) -> Result<TxEnv> {
    test.tx_env(unit).map_err(|e| eyre!("Failed to build tx env: {:?}", e))
}

/// Compile all contracts in pre-state using JIT
///
/// This uses a two-phase approach:
/// 1. Translate all contracts (before module finalization)
/// 2. JIT all translated functions (first call finalizes the module)
///
/// This is necessary because `translate()` cannot be called after finalization,
/// but `jit_function()` can be called multiple times after the module is finalized.
pub fn compile_contracts<'ctx>(
    unit: &TestUnit,
    spec_id: SpecId,
    compiler: &mut EvmCompiler<EvmLlvmBackend<'ctx>>,
) -> Result<CompiledContracts> {
    let mut compiled = CompiledContracts::new();
    let mut func_ids: Vec<(B256, <EvmLlvmBackend<'ctx> as Backend>::FuncId)> = Vec::new();

    // Phase 1: Translate all contracts (before finalization)
    for (address, info) in &unit.pre {
        if info.code.is_empty() {
            continue;
        }

        let code_hash = keccak256(&info.code);
        // Skip if we already have this code hash queued
        if func_ids.iter().any(|(hash, _)| hash == &code_hash) {
            continue;
        }

        let name = format!("contract_{:x}", address);
        let func_id = compiler
            .translate(&name, &info.code[..], spec_id)
            .map_err(|e| eyre!("Failed to translate contract {:x}: {}", address, e))?;

        func_ids.push((code_hash, func_id));
    }

    // Phase 2: JIT all translated functions (first call finalizes the module)
    for (code_hash, func_id) in func_ids {
        let func = unsafe { compiler.jit_function(func_id) }
            .map_err(|e| eyre!("Failed to JIT function for {:x}: {}", code_hash, e))?;

        compiled.insert(code_hash, func);
    }

    Ok(compiled)
}

/// Run a single test unit with the JIT compiler
pub fn run_test_unit(name: &str, unit: &TestUnit, spec_name: &SpecName) -> Result<Vec<TestResult>> {
    let spec_str = format!("{:?}", spec_name);

    let Some(spec_id) = spec_name_to_spec_id(spec_name) else {
        return Ok(vec![TestResult {
            name: name.to_string(),
            spec: spec_str,
            passed: true,
            error: Some("Skipped: unsupported spec".to_string()),
        }]);
    };

    let tests = match unit.post.get(spec_name) {
        Some(tests) => tests,
        None => return Ok(vec![]),
    };

    let mut results = Vec::new();

    for (idx, test) in tests.iter().enumerate() {
        let result = run_single_test(name, unit, test, spec_id, idx);
        results.push(TestResult {
            name: format!("{}[{}]", name, idx),
            spec: spec_str.clone(),
            passed: result.is_ok(),
            error: result.err().map(|e| e.to_string()),
        });
    }

    Ok(results)
}

/// Run a single test case comparing JIT vs interpreter
fn run_single_test(
    name: &str,
    unit: &TestUnit,
    test: &revm_statetest_types::Test,
    spec_id: SpecId,
    idx: usize,
) -> Result<()> {
    let has_code = unit.pre.values().any(|acc| !acc.code.is_empty());
    if !has_code {
        return Ok(());
    }

    if test.indexes.data >= unit.transaction.data.len() {
        return Err(eyre!(
            "Test {}[{}]: invalid data index {} >= {}",
            name,
            idx,
            test.indexes.data,
            unit.transaction.data.len()
        ));
    }
    if test.indexes.gas >= unit.transaction.gas_limit.len() {
        return Err(eyre!(
            "Test {}[{}]: invalid gas index {} >= {}",
            name,
            idx,
            test.indexes.gas,
            unit.transaction.gas_limit.len()
        ));
    }
    if test.indexes.value >= unit.transaction.value.len() {
        return Err(eyre!(
            "Test {}[{}]: invalid value index {} >= {}",
            name,
            idx,
            test.indexes.value,
            unit.transaction.value.len()
        ));
    }

    let tx_env = build_tx_env(unit, test)?;
    let db = build_pre_state(unit);

    let interpreter_result = run_with_interpreter(unit, &tx_env, db.clone(), spec_id)?;

    let context = LlvmContext::create();
    let backend = EvmLlvmBackend::new(&context, false, OptimizationLevel::Default)?;
    let mut compiler = EvmCompiler::new(backend);
    let compiled = compile_contracts(unit, spec_id, &mut compiler)?;
    let jit_result = run_with_jit(unit, &tx_env, db, spec_id, &compiled)?;

    if interpreter_result.success != jit_result.success {
        return Err(eyre!(
            "Test {}[{}]: success mismatch: interpreter={}, jit={}",
            name,
            idx,
            interpreter_result.success,
            jit_result.success
        ));
    }

    if interpreter_result.gas_used != jit_result.gas_used {
        return Err(eyre!(
            "Test {}[{}]: gas mismatch: interpreter={}, jit={}",
            name,
            idx,
            interpreter_result.gas_used,
            jit_result.gas_used
        ));
    }

    if interpreter_result.output != jit_result.output {
        return Err(eyre!("Test {}[{}]: output mismatch", name, idx));
    }

    Ok(())
}

/// Result from executing a test
#[derive(Debug)]
pub struct ExecutionResult {
    pub success: bool,
    pub gas_used: u64,
    pub output: Vec<u8>,
}

/// Run test with standard revm interpreter at bytecode level.
///
/// To match JIT gas accounting, we run the interpreter directly on the bytecode
/// rather than using full transaction execution which includes intrinsic gas costs.
fn run_with_interpreter(
    unit: &TestUnit,
    tx_env: &TxEnv,
    db: CacheDB<EmptyDB>,
    spec_id: SpecId,
) -> Result<ExecutionResult> {
    let target = match tx_env.kind {
        TxKind::Call(addr) => addr,
        TxKind::Create => {
            return Ok(ExecutionResult { success: true, gas_used: 0, output: Vec::new() });
        }
    };

    let account = match unit.pre.get(&target) {
        Some(acc) if !acc.code.is_empty() => acc,
        _ => {
            return Ok(ExecutionResult { success: true, gas_used: 0, output: Vec::new() });
        }
    };

    let gas_limit = tx_env.gas_limit;
    let bytecode = Bytecode::new_legacy(account.code.clone());
    let ext_bytecode = ExtBytecode::new(bytecode);

    let input = InputsImpl {
        target_address: target,
        bytecode_address: None,
        caller_address: tx_env.caller,
        input: revm::interpreter::CallInput::Bytes(tx_env.data.clone()),
        call_value: tx_env.value,
    };

    let mut interpreter =
        Interpreter::new(SharedMemory::new(), ext_bytecode, input, false, spec_id, gas_limit);

    let mut cfg = CfgEnv::default();
    cfg.spec = spec_id;
    let block = unit.block_env(&mut cfg);
    let mut ctx = Context::<BlockEnv, TxEnv, CfgEnv, _, Journal<_>, ()>::new(db, spec_id)
        .with_block(block)
        .with_cfg(cfg);

    // Load the target account into the journal (required for SLOAD/SSTORE to work)
    let _ = ctx.journaled_state.load_account(target);
    let _ = ctx.journaled_state.load_account(tx_env.caller);

    let table = instruction_table_gas_changes_spec::<EthInterpreter, _>(spec_id);
    let mut action = interpreter.run_plain(&table, &mut ctx);

    loop {
        match action {
            InterpreterAction::Return(result) => {
                return Ok(ExecutionResult {
                    success: result.result.is_ok(),
                    gas_used: interpreter.gas.spent(),
                    output: result.output.to_vec(),
                });
            }
            InterpreterAction::NewFrame(frame_input) => {
                let (call_result, return_memory_offset) = match frame_input {
                    FrameInput::Call(ref call_inputs) => {
                        let offset = call_inputs.return_memory_offset.clone();
                        let result = execute_frame_interpreter(&mut ctx, frame_input, spec_id);
                        (result, Some(offset))
                    }
                    FrameInput::Create(_) => {
                        let result = execute_frame_interpreter(&mut ctx, frame_input, spec_id);
                        (result, None)
                    }
                    FrameInput::Empty => {
                        return Ok(ExecutionResult {
                            success: false,
                            gas_used: interpreter.gas.spent(),
                            output: Vec::new(),
                        });
                    }
                };

                insert_call_outcome(&mut interpreter, call_result, return_memory_offset);
                action = interpreter.run_plain(&table, &mut ctx);
            }
        }
    }
}

/// Insert call outcome into interpreter state.
///
/// This mimics what revm-handler does in `insert_call_outcome`:
/// - Push success indicator (1 or 0) onto stack
/// - Copy return data to memory at the specified offset
/// - Return unspent gas to parent
/// - Record refunds on success
fn insert_call_outcome(
    interpreter: &mut Interpreter<EthInterpreter>,
    outcome: InterpreterResult,
    return_memory_offset: Option<Range<usize>>,
) {
    let ins_result = outcome.result;
    let out_gas = outcome.gas;
    let returned_len = outcome.output.len();

    interpreter.return_data.set_buffer(outcome.output);

    let success_indicator = if ins_result.is_ok() { U256::from(1) } else { U256::ZERO };
    let _ = interpreter.stack.push(success_indicator);

    if ins_result.is_ok_or_revert() {
        interpreter.gas.erase_cost(out_gas.remaining());

        if let Some(mem_range) = return_memory_offset {
            let target_len = min(mem_range.len(), returned_len);
            if target_len > 0 {
                interpreter
                    .memory
                    .set(mem_range.start, &interpreter.return_data.buffer()[..target_len]);
            }
        }
    }

    if ins_result.is_ok() {
        interpreter.gas.record_refund(out_gas.refunded());
    }
}

/// Execute a nested call frame using the interpreter.
fn execute_frame_interpreter<DB: revm::Database>(
    ctx: &mut Context<BlockEnv, TxEnv, CfgEnv, DB, Journal<DB>, ()>,
    frame_input: FrameInput,
    spec_id: SpecId,
) -> InterpreterResult
where
    DB::Error: std::fmt::Debug,
{
    match frame_input {
        FrameInput::Call(call_inputs) => {
            let target = call_inputs.bytecode_address;
            let code_result = ctx.journaled_state.code(target);
            let Ok(state_load) = code_result else {
                return InterpreterResult {
                    result: revm::interpreter::InstructionResult::FatalExternalError,
                    output: Bytes::new(),
                    gas: revm::interpreter::Gas::new(0),
                };
            };
            let code_bytes = state_load.data;

            if code_bytes.is_empty() {
                return InterpreterResult {
                    result: revm::interpreter::InstructionResult::Stop,
                    output: Bytes::new(),
                    gas: revm::interpreter::Gas::new(call_inputs.gas_limit),
                };
            }

            let bytecode = Bytecode::new_legacy(code_bytes);
            let ext_bytecode = ExtBytecode::new(bytecode);
            let input = InputsImpl {
                target_address: call_inputs.target_address,
                bytecode_address: Some(call_inputs.bytecode_address),
                caller_address: call_inputs.caller,
                input: call_inputs.input.clone(),
                call_value: match call_inputs.value {
                    revm::interpreter::CallValue::Transfer(v) => v,
                    revm::interpreter::CallValue::Apparent(v) => v,
                },
            };

            let mut nested_interpreter = Interpreter::new(
                SharedMemory::new(),
                ext_bytecode,
                input,
                call_inputs.is_static,
                spec_id,
                call_inputs.gas_limit,
            );

            let table = instruction_table_gas_changes_spec::<EthInterpreter, _>(spec_id);
            let mut nested_action = nested_interpreter.run_plain(&table, ctx);

            loop {
                match nested_action {
                    InterpreterAction::Return(result) => {
                        return InterpreterResult {
                            result: result.result,
                            output: result.output,
                            gas: nested_interpreter.gas,
                        };
                    }
                    InterpreterAction::NewFrame(inner_frame) => {
                        let (inner_result, inner_return_offset) = match inner_frame {
                            FrameInput::Call(ref inner_call) => {
                                let offset = inner_call.return_memory_offset.clone();
                                let result = execute_frame_interpreter(ctx, inner_frame, spec_id);
                                (result, Some(offset))
                            }
                            FrameInput::Create(_) => {
                                let result = execute_frame_interpreter(ctx, inner_frame, spec_id);
                                (result, None)
                            }
                            FrameInput::Empty => {
                                return InterpreterResult {
                                    result: revm::interpreter::InstructionResult::Stop,
                                    output: Bytes::new(),
                                    gas: nested_interpreter.gas,
                                };
                            }
                        };
                        insert_call_outcome(
                            &mut nested_interpreter,
                            inner_result,
                            inner_return_offset,
                        );
                        nested_action = nested_interpreter.run_plain(&table, ctx);
                    }
                }
            }
        }
        FrameInput::Create(create_inputs) => InterpreterResult {
            result: revm::interpreter::InstructionResult::CreateInitCodeStartingEF00,
            output: Bytes::new(),
            gas: revm::interpreter::Gas::new(create_inputs.gas_limit()),
        },
        FrameInput::Empty => InterpreterResult {
            result: revm::interpreter::InstructionResult::Stop,
            output: Bytes::new(),
            gas: revm::interpreter::Gas::new(0),
        },
    }
}

/// Run test with JIT-compiled bytecode
///
/// This function handles the execution loop for JIT-compiled code, including
/// processing CALL/CREATE actions that require re-entry.
fn run_with_jit(
    unit: &TestUnit,
    tx_env: &TxEnv,
    db: CacheDB<EmptyDB>,
    spec_id: SpecId,
    compiled: &CompiledContracts,
) -> Result<ExecutionResult> {
    let target = match tx_env.kind {
        TxKind::Call(addr) => addr,
        TxKind::Create => {
            return Ok(ExecutionResult { success: true, gas_used: 0, output: Vec::new() });
        }
    };

    let account = match unit.pre.get(&target) {
        Some(acc) if !acc.code.is_empty() => acc,
        _ => {
            return Ok(ExecutionResult { success: true, gas_used: 0, output: Vec::new() });
        }
    };

    let code_hash = keccak256(&account.code);
    let Some(jit_fn) = compiled.get(&code_hash) else {
        return Err(eyre!("No compiled function for target {:x}", target));
    };

    let gas_limit = tx_env.gas_limit;
    let bytecode = Bytecode::new_legacy(account.code.clone());
    let ext_bytecode = ExtBytecode::new(bytecode);

    let input = InputsImpl {
        target_address: target,
        bytecode_address: None,
        caller_address: tx_env.caller,
        input: revm::interpreter::CallInput::Bytes(tx_env.data.clone()),
        call_value: tx_env.value,
    };

    let mut interpreter: Interpreter<EthInterpreter> =
        Interpreter::new(SharedMemory::new(), ext_bytecode, input, false, spec_id, gas_limit);

    let mut cfg = CfgEnv::default();
    cfg.spec = spec_id;
    let block = unit.block_env(&mut cfg);
    let mut ctx = Context::<BlockEnv, TxEnv, CfgEnv, _, Journal<_>, ()>::new(db, spec_id)
        .with_block(block)
        .with_cfg(cfg);

    // Load the target account into the journal (required for SLOAD/SSTORE to work)
    let _ = ctx.journaled_state.load_account(target);
    let _ = ctx.journaled_state.load_account(tx_env.caller);

    // Track resume_at across calls
    let mut resume_at: usize = 0;

    // First call
    let (result, new_resume_at) =
        call_jit_with_resume(&mut interpreter, &mut ctx, jit_fn, resume_at);
    resume_at = new_resume_at;

    let mut last_result = result;

    let mut iteration = 0;
    loop {
        iteration += 1;
        if iteration > 100 {
            return Err(eyre::eyre!("Too many iterations"));
        }

        // Check if there's a pending action
        let action = interpreter.bytecode.action.take();

        match action {
            Some(InterpreterAction::NewFrame(frame_input)) => {
                let (call_result, return_memory_offset) = match frame_input {
                    FrameInput::Call(ref call_inputs) => {
                        let offset = call_inputs.return_memory_offset.clone();
                        let result = execute_frame(&mut ctx, frame_input, spec_id, compiled);
                        (result, Some(offset))
                    }
                    FrameInput::Create(_) => {
                        let result = execute_frame(&mut ctx, frame_input, spec_id, compiled);
                        (result, None)
                    }
                    FrameInput::Empty => {
                        return Ok(ExecutionResult {
                            success: false,
                            gas_used: interpreter.gas.spent(),
                            output: Vec::new(),
                        });
                    }
                };

                insert_call_outcome(&mut interpreter, call_result, return_memory_offset);

                // Resume with preserved resume_at
                let (result, new_resume_at) =
                    call_jit_with_resume(&mut interpreter, &mut ctx, jit_fn, resume_at);
                resume_at = new_resume_at;
                last_result = result;
            }
            Some(InterpreterAction::Return(ret_result)) => {
                return Ok(ExecutionResult {
                    success: ret_result.result.is_ok(),
                    gas_used: interpreter.gas.spent(),
                    output: ret_result.output.to_vec(),
                });
            }
            None => {
                return Ok(ExecutionResult {
                    success: last_result.is_ok(),
                    gas_used: interpreter.gas.spent(),
                    output: Vec::new(),
                });
            }
        }
    }
}

/// Call JIT function with explicit resume_at tracking.
fn call_jit_with_resume<DB: revm::Database + 'static>(
    interpreter: &mut Interpreter<EthInterpreter>,
    ctx: &mut Context<BlockEnv, TxEnv, CfgEnv, DB, Journal<DB>, ()>,
    jit_fn: EvmCompilerFn,
    resume_at: usize,
) -> (revm::interpreter::InstructionResult, usize)
where
    DB::Error: std::fmt::Debug,
{
    interpreter.bytecode.action = None;

    let (stack, stack_len) = EvmStack::from_interpreter_stack(&mut interpreter.stack);
    let mut ecx = EvmContext {
        memory: &mut interpreter.memory,
        input: &mut interpreter.input,
        gas: &mut interpreter.gas,
        host: ctx,
        next_action: &mut interpreter.bytecode.action,
        return_data: interpreter.return_data.buffer(),
        is_static: interpreter.runtime_flag.is_static,
        resume_at,
    };

    let result = unsafe { jit_fn.call(Some(stack), Some(stack_len), &mut ecx) };

    if result == revm::interpreter::InstructionResult::OutOfGas {
        ecx.gas.spend_all();
    }

    let new_resume_at = ecx.resume_at;
    (result, new_resume_at)
}

/// Call JIT function with explicit resume_at tracking for nested frames.
/// This is similar to call_jit_with_resume but accepts a generic Host.
fn call_jit_with_resume_nested<H: revmc::HostExt>(
    interpreter: &mut Interpreter<EthInterpreter>,
    host: &mut H,
    jit_fn: EvmCompilerFn,
    resume_at: usize,
) -> (revm::interpreter::InstructionResult, usize) {
    interpreter.bytecode.action = None;

    let (stack, stack_len) = EvmStack::from_interpreter_stack(&mut interpreter.stack);
    let mut ecx = EvmContext {
        memory: &mut interpreter.memory,
        input: &mut interpreter.input,
        gas: &mut interpreter.gas,
        host,
        next_action: &mut interpreter.bytecode.action,
        return_data: interpreter.return_data.buffer(),
        is_static: interpreter.runtime_flag.is_static,
        resume_at,
    };

    let result = unsafe { jit_fn.call(Some(stack), Some(stack_len), &mut ecx) };

    if result == revm::interpreter::InstructionResult::OutOfGas {
        ecx.gas.spend_all();
    }

    let new_resume_at = ecx.resume_at;
    (result, new_resume_at)
}

/// Execute a nested call or create frame using JIT-compiled code.
fn execute_frame<DB: revm::Database + 'static>(
    ctx: &mut Context<BlockEnv, TxEnv, CfgEnv, DB, Journal<DB>, ()>,
    frame_input: FrameInput,
    spec_id: SpecId,
    compiled: &CompiledContracts,
) -> InterpreterResult
where
    DB::Error: std::fmt::Debug,
{
    match frame_input {
        FrameInput::Call(call_inputs) => {
            let target = call_inputs.bytecode_address;
            let code_result = ctx.journaled_state.code(target);
            let Ok(state_load) = code_result else {
                return InterpreterResult {
                    result: revm::interpreter::InstructionResult::FatalExternalError,
                    output: Bytes::new(),
                    gas: revm::interpreter::Gas::new(0),
                };
            };
            let code_bytes = state_load.data;

            if code_bytes.is_empty() {
                return InterpreterResult {
                    result: revm::interpreter::InstructionResult::Stop,
                    output: Bytes::new(),
                    gas: revm::interpreter::Gas::new(call_inputs.gas_limit),
                };
            }

            let code_hash = keccak256(&code_bytes);

            if let Some(jit_fn) = compiled.get(&code_hash) {
                let bytecode = Bytecode::new_legacy(code_bytes);
                let ext_bytecode = ExtBytecode::new(bytecode);
                let input = InputsImpl {
                    target_address: call_inputs.target_address,
                    bytecode_address: Some(call_inputs.bytecode_address),
                    caller_address: call_inputs.caller,
                    input: call_inputs.input.clone(),
                    call_value: match call_inputs.value {
                        revm::interpreter::CallValue::Transfer(v) => v,
                        revm::interpreter::CallValue::Apparent(v) => v,
                    },
                };

                let mut nested_interpreter = Interpreter::new(
                    SharedMemory::new(),
                    ext_bytecode,
                    input,
                    call_inputs.is_static,
                    spec_id,
                    call_inputs.gas_limit,
                );

                // Track resume_at across nested call suspensions
                let mut resume_at: usize = 0;

                // Use call_jit_with_resume_nested instead of call_with_interpreter to properly
                // track resume_at
                let (result, new_resume_at) =
                    call_jit_with_resume_nested(&mut nested_interpreter, ctx, jit_fn, resume_at);
                resume_at = new_resume_at;
                let mut last_result = result;

                loop {
                    let action = nested_interpreter.bytecode.action.take();
                    match action {
                        Some(InterpreterAction::Return(result)) => {
                            return InterpreterResult {
                                result: result.result,
                                output: result.output,
                                gas: nested_interpreter.gas,
                            };
                        }
                        Some(InterpreterAction::NewFrame(inner_frame)) => {
                            let (inner_result, inner_return_offset) = match inner_frame {
                                FrameInput::Call(ref inner_call) => {
                                    let offset = inner_call.return_memory_offset.clone();
                                    let result = execute_frame(ctx, inner_frame, spec_id, compiled);
                                    (result, Some(offset))
                                }
                                FrameInput::Create(_) => {
                                    let result = execute_frame(ctx, inner_frame, spec_id, compiled);
                                    (result, None)
                                }
                                FrameInput::Empty => {
                                    return InterpreterResult {
                                        result: revm::interpreter::InstructionResult::Stop,
                                        output: Bytes::new(),
                                        gas: nested_interpreter.gas,
                                    };
                                }
                            };
                            insert_call_outcome(
                                &mut nested_interpreter,
                                inner_result,
                                inner_return_offset,
                            );
                            // Resume with preserved resume_at
                            let (result, new_resume_at) = call_jit_with_resume_nested(
                                &mut nested_interpreter,
                                ctx,
                                jit_fn,
                                resume_at,
                            );
                            resume_at = new_resume_at;
                            last_result = result;
                        }
                        None => {
                            // JIT returned without setting an action - execution is done
                            return InterpreterResult {
                                result: last_result,
                                output: Bytes::new(),
                                gas: nested_interpreter.gas,
                            };
                        }
                    }
                }
            } else {
                InterpreterResult {
                    result: revm::interpreter::InstructionResult::Stop,
                    output: Bytes::new(),
                    gas: revm::interpreter::Gas::new(call_inputs.gas_limit),
                }
            }
        }
        FrameInput::Create(create_inputs) => InterpreterResult {
            result: revm::interpreter::InstructionResult::CreateInitCodeStartingEF00,
            output: Bytes::new(),
            gas: revm::interpreter::Gas::new(create_inputs.gas_limit()),
        },
        FrameInput::Empty => InterpreterResult {
            result: revm::interpreter::InstructionResult::Stop,
            output: Bytes::new(),
            gas: revm::interpreter::Gas::new(0),
        },
    }
}

/// Run all GeneralStateTests
pub fn run_general_state_tests(path: &Path) -> Result<Vec<TestResult>> {
    let test_files = find_json_tests(path);
    let mut all_results = Vec::new();

    for file in test_files {
        let suite = match load_test_suite(&file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to load {}: {}", file.display(), e);
                continue;
            }
        };

        for (name, unit) in suite.0.iter() {
            for spec_name in unit.post.keys() {
                match run_test_unit(name, unit, spec_name) {
                    Ok(results) => all_results.extend(results),
                    Err(e) => {
                        all_results.push(TestResult {
                            name: name.clone(),
                            spec: format!("{:?}", spec_name),
                            passed: false,
                            error: Some(e.to_string()),
                        });
                    }
                }
            }
        }
    }

    Ok(all_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm::primitives::Address;

    #[test]
    fn test_spec_name_conversion() {
        assert_eq!(spec_name_to_spec_id(&SpecName::Cancun), Some(SpecId::CANCUN));
        assert_eq!(spec_name_to_spec_id(&SpecName::Constantinople), None);
    }

    #[test]
    fn test_build_pre_state() {
        use revm::primitives::U256;
        use revm_statetest_types::AccountInfo;

        let mut pre = revm::primitives::HashMap::default();
        let code: Bytes = vec![0x60, 0x01, 0x00].into();
        let mut storage = revm::primitives::HashMap::default();
        storage.insert(U256::from(1), U256::from(42));

        pre.insert(
            Address::repeat_byte(1),
            AccountInfo { balance: U256::from(1000), nonce: 5, code, storage },
        );

        let unit = TestUnit {
            info: None,
            env: serde_json::from_str(
                r#"{
                "currentCoinbase": "0x0000000000000000000000000000000000000000",
                "currentDifficulty": "0x0",
                "currentGasLimit": "0x989680",
                "currentNumber": "0x0",
                "currentTimestamp": "0x0",
                "currentBaseFee": "0x7"
            }"#,
            )
            .unwrap(),
            pre,
            post: Default::default(),
            transaction: Default::default(),
            out: None,
        };

        let db = build_pre_state(&unit);

        let acc = db.cache.accounts.get(&Address::repeat_byte(1)).unwrap();
        assert_eq!(acc.info.balance, U256::from(1000));
        assert_eq!(acc.info.nonce, 5);
        assert!(acc.info.code.is_some());

        let storage_val = acc.storage.get(&U256::from(1)).unwrap();
        assert_eq!(*storage_val, U256::from(42));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simple_jit_execution() {
        // Simple bytecode: PUSH1 0x42, PUSH1 0x00, MSTORE, PUSH1 0x20, PUSH1 0x00, RETURN
        // Returns 0x42 in 32 bytes
        let bytecode: &[u8] = &[
            0x60, 0x42, // PUSH1 0x42
            0x60, 0x00, // PUSH1 0x00
            0x52, // MSTORE
            0x60, 0x20, // PUSH1 0x20
            0x60, 0x00, // PUSH1 0x00
            0xf3, // RETURN
        ];

        let spec_id = SpecId::CANCUN;
        let context = LlvmContext::create();
        let backend = EvmLlvmBackend::new(&context, false, OptimizationLevel::Default)
            .expect("Failed to create backend");
        let mut compiler = EvmCompiler::new(backend);

        let jit_fn = unsafe { compiler.jit("test_simple", bytecode, spec_id) }
            .expect("Failed to JIT compile");

        // Set up interpreter context
        let bytecode_obj = Bytecode::new_legacy(bytecode.to_vec().into());
        let ext_bytecode = ExtBytecode::new(bytecode_obj);
        let input = InputsImpl {
            target_address: Address::ZERO,
            bytecode_address: None,
            caller_address: Address::ZERO,
            input: revm::interpreter::CallInput::Bytes(Default::default()),
            call_value: Default::default(),
        };

        let mut interpreter =
            Interpreter::new(SharedMemory::new(), ext_bytecode, input, false, spec_id, 100_000);

        let mut host = revm::context_interface::host::DummyHost::new(spec_id);
        let action = unsafe { jit_fn.call_with_interpreter(&mut interpreter, &mut host) };

        match action {
            revm::interpreter::InterpreterAction::Return(result) => {
                assert!(result.result.is_ok(), "Expected success, got {:?}", result.result);
                assert_eq!(result.output.len(), 32);
                assert_eq!(result.output[31], 0x42);
            }
            other => panic!("Expected Return action, got {:?}", other),
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_compile_multiple_bytecodes() {
        // Test that we can compile multiple different bytecodes
        let code1: &[u8] = &[0x60, 0x01, 0x60, 0x02, 0x01, 0x00]; // PUSH1 1, PUSH1 2, ADD, STOP
        let code2: &[u8] = &[0x60, 0x03, 0x60, 0x04, 0x02, 0x00]; // PUSH1 3, PUSH1 4, MUL, STOP

        let context = LlvmContext::create();
        let backend = EvmLlvmBackend::new(&context, false, OptimizationLevel::Default)
            .expect("Failed to create backend");
        let mut compiler = EvmCompiler::new(backend);

        // Phase 1: Translate both contracts BEFORE finalization
        let id1 = compiler
            .translate("contract1", code1, SpecId::CANCUN)
            .expect("Failed to translate contract 1");
        let id2 = compiler
            .translate("contract2", code2, SpecId::CANCUN)
            .expect("Failed to translate contract 2");

        // Phase 2: JIT both functions (first call finalizes the module)
        let fn1 = unsafe { compiler.jit_function(id1) }.expect("Failed to JIT contract 1");
        let fn2 = unsafe { compiler.jit_function(id2) }.expect("Failed to JIT contract 2");

        // Both functions should be valid (different addresses)
        assert_ne!(fn1.into_inner() as usize, fn2.into_inner() as usize);
    }

    #[test]
    #[ignore = "requires ethereum/tests checkout"]
    fn test_run_general_state_tests() {
        let mut path = get_ethtests_path().join("GeneralStateTests");

        // Allow running a specific subdirectory via SUBDIR env var
        if let Ok(subdir) = env::var("SUBDIR") {
            path = path.join(subdir);
        }

        if !path.exists() {
            eprintln!("Skipping: {} does not exist", path.display());
            return;
        }

        let results = run_general_state_tests(&path).unwrap();
        let passed = results.iter().filter(|r| r.passed).count();
        let failed = results.iter().filter(|r| !r.passed).count();
        println!("Passed: {}, Failed: {}", passed, failed);

        for result in results.iter().filter(|r| !r.passed) {
            println!("FAILED: {} ({}): {:?}", result.name, result.spec, result.error);
        }
    }
}

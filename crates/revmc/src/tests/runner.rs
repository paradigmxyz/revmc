use super::*;
use interpreter::LoadAccountResult;
use revm_interpreter::{opcode as op, Contract, DummyHost, Host};
use revm_primitives::{
    spec_to_generic, BlobExcessGasAndPrice, BlockEnv, CfgEnv, Env, HashMap, TxEnv, EOF_MAGIC_BYTES,
};
use similar_asserts::assert_eq;
use std::{fmt, path::Path, sync::OnceLock};

pub struct TestCase<'a> {
    pub bytecode: &'a [u8],
    pub spec_id: SpecId,

    pub modify_ecx: Option<fn(&mut EvmContext<'_>)>,

    pub expected_return: InstructionResult,
    pub expected_stack: &'a [U256],
    pub expected_memory: &'a [u8],
    pub expected_gas: u64,
    pub expected_next_action: InterpreterAction,
    pub assert_host: Option<fn(&TestHost)>,
    pub assert_ecx: Option<fn(&EvmContext<'_>)>,
}

#[cfg(feature = "__fuzzing")]
impl<'a> arbitrary::Arbitrary<'a> for TestCase<'a> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let is_eof_enabled = u.arbitrary::<bool>()?;
        let spec_id_range = if is_eof_enabled {
            SpecId::PRAGUE_EOF as u8..=SpecId::PRAGUE_EOF as u8
        } else {
            0..=(SpecId::PRAGUE_EOF as u8 - 1)
        };
        let spec_id = SpecId::try_from_u8(u.int_in_range(spec_id_range)?).unwrap_or(DEF_SPEC);

        let mut bytecode: &'a [u8] = u.arbitrary()?;
        if is_eof_enabled && !bytecode.starts_with(&EOF_MAGIC_BYTES) && u.arbitrary()? {
            let code = eof_sections_unchecked(&[bytecode]);
            if revm_interpreter::analysis::validate_eof(&code).is_ok() {
                static mut STORAGE: Bytes = Bytes::new();
                unsafe {
                    STORAGE = code.raw;
                    bytecode = &STORAGE[..];
                }
            }
        }

        Ok(Self::what_interpreter_says(bytecode, spec_id))
    }
}

impl Default for TestCase<'_> {
    fn default() -> Self {
        Self {
            bytecode: &[],
            spec_id: DEF_SPEC,
            modify_ecx: None,
            expected_return: InstructionResult::Stop,
            expected_stack: &[],
            expected_memory: &[],
            expected_gas: 0,
            expected_next_action: InterpreterAction::None,
            assert_host: None,
            assert_ecx: None,
        }
    }
}

impl fmt::Debug for TestCase<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TestCase")
            .field("bytecode", &format_bytecode(self.bytecode, self.spec_id))
            .field("spec_id", &self.spec_id)
            .field("modify_ecx", &self.modify_ecx.is_some())
            .field("expected_return", &self.expected_return)
            .field("expected_stack", &self.expected_stack)
            .field("expected_memory", &MemDisplay(self.expected_memory))
            .field("expected_gas", &self.expected_gas)
            .field("expected_next_action", &self.expected_next_action)
            .field("assert_host", &self.assert_host.is_some())
            .field("assert_ecx", &self.assert_ecx.is_some())
            .finish()
    }
}

impl<'a> TestCase<'a> {
    pub fn what_interpreter_says(bytecode: &'a [u8], spec_id: SpecId) -> Self {
        Self {
            bytecode,
            spec_id,
            modify_ecx: None,
            expected_return: RETURN_WHAT_INTERPRETER_SAYS,
            expected_stack: STACK_WHAT_INTERPRETER_SAYS,
            expected_memory: MEMORY_WHAT_INTERPRETER_SAYS,
            expected_gas: GAS_WHAT_INTERPRETER_SAYS,
            expected_next_action: ACTION_WHAT_INTERPRETER_SAYS,
            assert_host: None,
            assert_ecx: None,
        }
    }
}

// Default values.
pub const DEF_SPEC: SpecId = SpecId::CANCUN;
pub const DEF_OPINFOS: &[OpcodeInfo; 256] = op_info_map(DEF_SPEC);

pub const DEF_GAS_LIMIT: u64 = 100_000;
pub const DEF_GAS_LIMIT_U256: U256 = U256::from_le_slice(&DEF_GAS_LIMIT.to_le_bytes());

/// Default code address.
pub const DEF_ADDR: Address = Address::repeat_byte(0xba);
pub const DEF_CALLER: Address = Address::repeat_byte(0xca);
pub static DEF_CD: &[u8] = &[0xaa; 64];
pub static DEF_RD: &[u8] = &[0xbb; 64];
pub static DEF_DATA: &[u8] = &[0xcc; 64];
pub const DEF_VALUE: U256 = uint!(123_456_789_U256);
pub static DEF_ENV: OnceLock<Env> = OnceLock::new();
pub static DEF_STORAGE: OnceLock<HashMap<U256, U256>> = OnceLock::new();
pub static DEF_CODEMAP: OnceLock<HashMap<Address, primitives::Bytecode>> = OnceLock::new();
pub const OTHER_ADDR: Address = Address::repeat_byte(0x69);
pub const DEF_BN: U256 = uint!(500_U256);

pub const RETURN_WHAT_INTERPRETER_SAYS: InstructionResult = InstructionResult::PrecompileError;
pub const STACK_WHAT_INTERPRETER_SAYS: &[U256] =
    &[U256::from_be_slice(&GAS_WHAT_INTERPRETER_SAYS.to_be_bytes())];
pub const MEMORY_WHAT_INTERPRETER_SAYS: &[u8] = &GAS_WHAT_INTERPRETER_SAYS.to_be_bytes();
pub const GAS_WHAT_INTERPRETER_SAYS: u64 = 0x4682e332d6612de1;
pub const ACTION_WHAT_INTERPRETER_SAYS: InterpreterAction = InterpreterAction::Return {
    result: InterpreterResult {
        gas: Gas::new(GAS_WHAT_INTERPRETER_SAYS),
        output: Bytes::from_static(MEMORY_WHAT_INTERPRETER_SAYS),
        result: RETURN_WHAT_INTERPRETER_SAYS,
    },
};

pub fn def_env() -> &'static Env {
    DEF_ENV.get_or_init(|| Env {
        cfg: {
            let mut cfg = CfgEnv::default();
            cfg.chain_id = 69;
            cfg
        },
        block: BlockEnv {
            number: DEF_BN,
            coinbase: Address::repeat_byte(0xcb),
            timestamp: U256::from(0x1234),
            gas_limit: U256::from(0x5678),
            basefee: U256::from(0x1231),
            difficulty: U256::from(0xcdef),
            prevrandao: Some(U256::from(0x0123).into()),
            blob_excess_gas_and_price: Some(BlobExcessGasAndPrice::new(50)),
        },
        tx: TxEnv {
            caller: Address::repeat_byte(0xcc),
            gas_limit: DEF_GAS_LIMIT,
            gas_price: U256::from(0x4567),
            transact_to: primitives::TransactTo::Call(DEF_ADDR),
            value: DEF_VALUE,
            data: DEF_CD.into(),
            nonce: None,
            chain_id: Some(420), // Different from `cfg.chain_id`.
            access_list: vec![],
            gas_priority_fee: Some(U256::from(0x69)),
            blob_hashes: vec![B256::repeat_byte(0xb7), B256::repeat_byte(0xb8)],
            max_fee_per_blob_gas: None,
            authorization_list: None,
            #[cfg(feature = "optimism")]
            optimism: Default::default(),
        },
    })
}

pub fn def_storage() -> &'static HashMap<U256, U256> {
    DEF_STORAGE.get_or_init(|| {
        HashMap::from([
            (U256::from(0), U256::from(1)),
            (U256::from(1), U256::from(2)),
            (U256::from(69), U256::from(42)),
        ])
    })
}

pub fn def_codemap() -> &'static HashMap<Address, primitives::Bytecode> {
    DEF_CODEMAP.get_or_init(|| {
        HashMap::from([
            //
            (
                OTHER_ADDR,
                primitives::Bytecode::new_raw(Bytes::from_static(&[
                    op::PUSH1,
                    0x69,
                    op::PUSH1,
                    0x42,
                    op::ADD,
                    op::STOP,
                ])),
            ),
        ])
    })
}

/// Wrapper around `DummyHost` that provides a stable environment and storage for testing.
pub struct TestHost {
    pub host: DummyHost,
    pub code_map: &'static HashMap<Address, primitives::Bytecode>,
    pub selfdestructs: Vec<(Address, Address)>,
}

impl Default for TestHost {
    fn default() -> Self {
        Self::new()
    }
}

impl TestHost {
    pub fn new() -> Self {
        Self {
            host: DummyHost {
                env: def_env().clone(),
                storage: def_storage().clone(),
                transient_storage: HashMap::new(),
                log: Vec::new(),
            },
            code_map: def_codemap(),
            selfdestructs: Vec::new(),
        }
    }
}

impl std::ops::Deref for TestHost {
    type Target = DummyHost;

    fn deref(&self) -> &Self::Target {
        &self.host
    }
}

impl std::ops::DerefMut for TestHost {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.host
    }
}

impl Host for TestHost {
    fn env(&self) -> &Env {
        self.host.env()
    }

    fn env_mut(&mut self) -> &mut Env {
        self.host.env_mut()
    }

    fn load_account(&mut self, address: Address) -> Option<LoadAccountResult> {
        self.host.load_account(address)
    }

    fn block_hash(&mut self, number: u64) -> Option<B256> {
        Some(U256::from(number).into())
    }

    fn balance(&mut self, address: Address) -> Option<(U256, bool)> {
        Some((U256::from(*address.last().unwrap()), false))
    }

    fn code(&mut self, address: Address) -> Option<(Bytes, bool)> {
        self.code_map
            .get(&address)
            .map(|b| (b.original_bytes(), false))
            .or_else(|| Some((Bytes::new(), false)))
    }

    fn code_hash(&mut self, address: Address) -> Option<(B256, bool)> {
        self.code_map.get(&address).map(|b| (b.hash_slow(), false)).or(Some((KECCAK_EMPTY, false)))
    }

    fn sload(&mut self, address: Address, index: U256) -> Option<(U256, bool)> {
        self.host.sload(address, index)
    }

    fn sstore(
        &mut self,
        address: Address,
        index: U256,
        value: U256,
    ) -> Option<interpreter::SStoreResult> {
        self.host.sstore(address, index, value)
    }

    fn tload(&mut self, address: Address, index: U256) -> U256 {
        self.host.tload(address, index)
    }

    fn tstore(&mut self, address: Address, index: U256, value: U256) {
        self.host.tstore(address, index, value)
    }

    fn log(&mut self, log: primitives::Log) {
        self.host.log(log)
    }

    fn selfdestruct(
        &mut self,
        address: Address,
        target: Address,
    ) -> Option<interpreter::SelfDestructResult> {
        self.selfdestructs.push((address, target));
        Some(interpreter::SelfDestructResult {
            had_value: false,
            target_exists: true,
            is_cold: false,
            previously_destroyed: false,
        })
    }
}

pub fn with_evm_context<F: FnOnce(&mut EvmContext<'_>, &mut EvmStack, &mut usize) -> R, R>(
    bytecode: &[u8],
    f: F,
) -> R {
    let contract = Contract {
        input: Bytes::from_static(DEF_CD),
        bytecode: revm_interpreter::analysis::to_analysed(revm_primitives::Bytecode::new_raw(
            Bytes::copy_from_slice(bytecode),
        )),
        hash: None,
        bytecode_address: None,
        target_address: DEF_ADDR,
        caller: DEF_CALLER,
        call_value: DEF_VALUE,
    };

    let mut interpreter = revm_interpreter::Interpreter::new(contract, DEF_GAS_LIMIT, false);
    interpreter.return_data_buffer = Bytes::from_static(DEF_RD);

    let mut host = TestHost::new();

    let (mut ecx, stack, stack_len) =
        EvmContext::from_interpreter_with_stack(&mut interpreter, &mut host);
    f(&mut ecx, stack, stack_len)
}

#[cfg(feature = "llvm")]
fn with_llvm_backend(opt_level: OptimizationLevel, f: impl FnOnce(EvmLlvmBackend<'_>)) {
    llvm::with_llvm_context(|cx| f(EvmLlvmBackend::new(cx, false, opt_level).unwrap()))
}

#[cfg(feature = "llvm")]
pub fn with_llvm_backend_jit(
    opt_level: OptimizationLevel,
    f: fn(&mut EvmCompiler<EvmLlvmBackend<'_>>),
) {
    with_llvm_backend(opt_level, |backend| f(&mut EvmCompiler::new(backend)));
}

pub fn set_test_dump<B: Backend>(compiler: &mut EvmCompiler<B>, module_path: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let mut dump_path = root.to_path_buf();
    dump_path.push("target");
    dump_path.push("tests_dump");
    // Skip `revmc::tests`.
    dump_path.extend(module_path.split("::").skip(2));
    dump_path.push(format!("{:?}", compiler.opt_level()));
    compiler.set_dump_to(Some(dump_path));
}

pub fn run_test_case<B: Backend>(test_case: &TestCase<'_>, compiler: &mut EvmCompiler<B>) {
    let TestCase { bytecode, spec_id, .. } = *test_case;
    compiler.inspect_stack_length(true);
    // Done manually in `fn eof` and friends.
    compiler.validate_eof(false);
    // compiler.debug_assertions(false);
    let f = unsafe { compiler.jit("test", bytecode, spec_id) }.unwrap();
    run_compiled_test_case(test_case, f);
}

fn run_compiled_test_case(test_case: &TestCase<'_>, f: EvmCompilerFn) {
    let TestCase {
        bytecode,
        spec_id,
        modify_ecx,
        expected_return,
        expected_stack,
        expected_memory,
        expected_gas,
        ref expected_next_action,
        assert_host,
        assert_ecx,
    } = *test_case;

    let is_eof_enabled = spec_id.is_enabled_in(SpecId::PRAGUE_EOF);

    if !is_eof_enabled && bytecode.starts_with(&primitives::EOF_MAGIC_BYTES) {
        panic!("EOF is not enabled in the current spec, forgot to set `spec_id`?");
    }

    with_evm_context(bytecode, |ecx, stack, stack_len| {
        if let Some(modify_ecx) = modify_ecx {
            modify_ecx(ecx);
        }

        if !cfg!(feature = "__fuzzing") && is_eof_enabled && !ecx.contract.bytecode.is_eof() {
            eprintln!("!!! WARNING: running legacy code under EOF !!!");
        }

        // Interpreter.
        let table = spec_to_generic!(test_case.spec_id, op::make_instruction_table::<_, SPEC>());
        let mut interpreter = ecx.to_interpreter(Default::default());
        let memory = interpreter.take_memory();
        let mut int_host = TestHost::new();

        let interpreter_action = interpreter.run(memory, &table, &mut int_host);

        let mut expected_return = expected_return;
        if expected_return == RETURN_WHAT_INTERPRETER_SAYS {
            expected_return = interpreter.instruction_result;
        } else {
            assert_eq!(
                interpreter.instruction_result, expected_return,
                "interpreter return value mismatch"
            );
        }

        let mut expected_stack = expected_stack;
        if expected_stack == STACK_WHAT_INTERPRETER_SAYS {
            expected_stack = interpreter.stack.data();
        } else {
            assert_eq!(interpreter.stack.data(), expected_stack, "interpreter stack mismatch");
        }

        let mut expected_memory = expected_memory;
        if expected_memory == MEMORY_WHAT_INTERPRETER_SAYS {
            expected_memory = interpreter.shared_memory.context_memory();
        } else {
            assert_eq!(
                MemDisplay(interpreter.shared_memory.context_memory()),
                MemDisplay(expected_memory),
                "interpreter memory mismatch"
            );
        }

        let mut expected_gas = expected_gas;
        if expected_gas == GAS_WHAT_INTERPRETER_SAYS {
            expected_gas = interpreter.gas.spent();
        } else {
            assert_eq!(interpreter.gas.spent(), expected_gas, "interpreter gas mismatch");
        }

        // This is what the interpreter returns when the internal action is None in `run`.
        let default_action = InterpreterAction::Return {
            result: InterpreterResult {
                result: interpreter.instruction_result,
                output: Bytes::new(),
                gas: interpreter.gas,
            },
        };
        let mut expected_next_action = expected_next_action;
        if *expected_next_action == ACTION_WHAT_INTERPRETER_SAYS {
            expected_next_action = &interpreter_action;
        } else {
            if expected_next_action.is_none() {
                expected_next_action = &default_action;
            }
            assert_actions(&interpreter_action, expected_next_action);
        }

        if let Some(assert_host) = assert_host {
            assert_host(&int_host);
        }

        let actual_return = unsafe { f.call(Some(stack), Some(stack_len), ecx) };

        if matches!(
            actual_return,
            // We can have a stack overflow/underflow before other error codes due to sections.
            |InstructionResult::StackOverflow| InstructionResult::StackUnderflow
            // Any OOG is equivalent. We skip `InvalidOperand` sometimes.
            | InstructionResult::OutOfGas | InstructionResult::MemoryOOG | InstructionResult::InvalidOperandOOG
        ) {
            assert_eq!(
                actual_return.is_error(),
                expected_return.is_error(),
                "return value mismatch: {actual_return:?} != {expected_return:?}"
            );
        } else {
            assert_eq!(actual_return, expected_return, "return value mismatch");
        }

        let actual_stack =
            stack.as_slice().iter().take(*stack_len).map(|x| x.to_u256()).collect::<Vec<_>>();

        // On EVM halt all available gas is consumed, so resulting stack, memory, and gas do not
        // matter. We do less work than the interpreter by bailing out earlier due to sections.
        if !actual_return.is_error() {
            assert_eq!(actual_stack, *expected_stack, "stack mismatch");

            assert_eq!(
                MemDisplay(ecx.memory.context_memory()),
                MemDisplay(expected_memory),
                "interpreter memory mismatch"
            );

            assert_eq!(ecx.gas.spent(), expected_gas, "gas mismatch");
        }

        let actual_next_action =
            if ecx.next_action.is_none() { &default_action } else { &*ecx.next_action };
        assert_actions(actual_next_action, expected_next_action);

        if let Some(_assert_host) = assert_host {
            #[cfg(not(feature = "__fuzzing"))]
            _assert_host(ecx.host.downcast_ref().unwrap());
        }

        if let Some(assert_ecx) = assert_ecx {
            assert_ecx(ecx);
        }
    });
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MemDisplay<'a>(&'a [u8]);
impl fmt::Debug for MemDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let chunks = self.0.chunks(32).map(revm_primitives::hex::encode_prefixed);
        f.debug_list().entries(chunks).finish()
    }
}

#[track_caller]
fn assert_actions(actual: &InterpreterAction, expected: &InterpreterAction) {
    match (actual, expected) {
        (
            InterpreterAction::Return { result },
            InterpreterAction::Return { result: expected_result },
        ) => {
            assert_eq!(result.result, expected_result.result, "result mismatch");
            assert_eq!(result.output, expected_result.output, "result output mismatch");
            if expected_result.gas.limit() != GAS_WHAT_INTERPRETER_SAYS {
                assert_eq!(result.gas.spent(), expected_result.gas.spent(), "result gas mismatch");
            }
        }
        (a, b) => assert_eq!(a, b, "next action mismatch"),
    }
}

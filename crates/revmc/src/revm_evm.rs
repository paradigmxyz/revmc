//! Generic [`revm`] JIT EVM and handler override.
//!
//! Provides [`JitEvm`] which wraps any mainnet-shaped [`EvmTr`]-based EVM and overrides
//! execution to look up compiled functions via [`JitBackend`] before falling
//! back to the interpreter.
//!
//! [`JitHandler`] is the underlying [`Handler`] implementation that can also be
//! used standalone.

use crate::runtime::{JitBackend, LookupDecision, LookupRequest};
use core::marker::PhantomData;
use revm_context_interface::{
    Cfg, ContextSetters, ContextTr, Database,
    journaled_state::JournalTr,
    local::LocalContextTr,
    result::{EVMError, HaltReason, InvalidTransaction},
};
use revm_handler::{
    EthFrame, EvmTr, EvmTrError, ExecuteEvm, FrameInitOrResult, FrameResult, Handler, ItemOrResult,
    PrecompileProvider, instructions::InstructionProvider,
};
use revm_interpreter::{
    InterpreterResult, interpreter::EthInterpreter, interpreter_action::FrameInit,
};
use revm_primitives::{B256Map, hardfork::SpecId};
use revm_state::EvmState;

/// Mainnet EVM with JIT-compiled function dispatch.
///
/// Wraps an inner [`EvmTr`] and uses [`JitBackend`] to look up compiled
/// functions before falling back to the interpreter.
///
/// Implements [`ExecuteEvm`] so it can be used as a drop-in replacement for
/// the standard mainnet EVM.
#[expect(missing_debug_implementations)]
pub struct JitEvm<EVM> {
    inner: EVM,
    backend: JitBackend,
    /// Cached lookup decisions keyed by `code_hash` alone.
    /// Invalidated when the `spec_id` changes.
    lookup_cache: B256Map<LookupDecision>,
    /// The `spec_id` the cache was built for; cleared on mismatch.
    lookup_cache_spec_id: SpecId,
}

impl<EVM: EvmTr> JitEvm<EVM>
where
    EVM::Context: ContextTr,
    <EVM::Context as ContextTr>::Cfg: Cfg<Spec: Into<SpecId>>,
{
    /// Creates a new JIT EVM from an inner EVM and backend.
    pub fn new(inner: EVM, backend: JitBackend) -> Self {
        let spec_id: SpecId = inner.ctx_ref().cfg().spec().into();
        Self {
            inner,
            backend,
            lookup_cache: B256Map::with_capacity_and_hasher(16, Default::default()),
            lookup_cache_spec_id: spec_id,
        }
    }

    /// Returns a reference to the inner EVM.
    pub const fn inner(&self) -> &EVM {
        &self.inner
    }

    /// Returns a mutable reference to the inner EVM.
    pub fn inner_mut(&mut self) -> &mut EVM {
        &mut self.inner
    }

    /// Returns a reference to the JIT backend.
    pub const fn backend(&self) -> &JitBackend {
        &self.backend
    }
}

impl<CTX, INSP, INST, PRECOMPILES> ExecuteEvm
    for JitEvm<revm_context::Evm<CTX, INSP, INST, PRECOMPILES, EthFrame>>
where
    CTX: ContextTr<Journal: JournalTr<State = EvmState>, Local: LocalContextTr> + ContextSetters,
    CTX::Cfg: Cfg<Spec: Into<SpecId>>,
    INST: InstructionProvider<Context = CTX, InterpreterTypes = EthInterpreter>,
    PRECOMPILES: PrecompileProvider<CTX, Output = InterpreterResult>,
{
    type ExecutionResult = revm_context_interface::result::ExecutionResult<HaltReason>;
    type State = EvmState;
    type Error = EVMError<<CTX::Db as Database>::Error, InvalidTransaction>;
    type Tx = <CTX as ContextTr>::Tx;
    type Block = <CTX as ContextTr>::Block;

    #[inline]
    fn transact_one(&mut self, tx: Self::Tx) -> Result<Self::ExecutionResult, Self::Error> {
        self.inner.ctx.set_tx(tx);
        let spec_id: SpecId = self.inner.ctx_ref().cfg().spec().into();
        if spec_id != self.lookup_cache_spec_id {
            self.lookup_cache.clear();
            self.lookup_cache_spec_id = spec_id;
        }
        JitHandler::new(&self.backend, &mut self.lookup_cache).run(&mut self.inner)
    }

    #[inline]
    fn finalize(&mut self) -> Self::State {
        self.inner.journal_mut().finalize()
    }

    #[inline]
    fn set_block(&mut self, block: Self::Block) {
        self.inner.ctx.set_block(block);
    }

    #[inline]
    fn replay(
        &mut self,
    ) -> Result<revm_context_interface::result::ResultAndState<HaltReason>, Self::Error> {
        JitHandler::new(&self.backend, &mut self.lookup_cache).run(&mut self.inner).map(|result| {
            let state = self.inner.journal_mut().finalize();
            revm_context_interface::result::ResultAndState::new(result, state)
        })
    }
}

// ---------------------------------------------------------------------------
// JitHandler
// ---------------------------------------------------------------------------

/// Custom handler that overrides `run_exec_loop` with JIT dispatch.
///
/// Wraps a generic [`EvmTr`] implementation and delegates all frame operations
/// to it, except `frame_run` which first checks the [`JitBackend`] for a
/// compiled function.
#[expect(missing_debug_implementations)]
pub struct JitHandler<'a, EVM, ERROR> {
    backend: &'a JitBackend,
    /// Lookup decision cache keyed by `code_hash` alone.
    /// The `spec_id` is constant within an execution; the caller invalidates on change.
    lookup_cache: &'a mut B256Map<LookupDecision>,
    _pd: PhantomData<fn() -> (EVM, ERROR)>,
}

impl<'a, EVM, ERROR> JitHandler<'a, EVM, ERROR> {
    /// Creates a new JIT handler from a backend and a mutable lookup cache.
    #[inline]
    pub fn new(backend: &'a JitBackend, lookup_cache: &'a mut B256Map<LookupDecision>) -> Self {
        Self { backend, lookup_cache, _pd: PhantomData }
    }
}

impl<EVM, ERROR> JitHandler<'_, EVM, ERROR>
where
    EVM: EvmTr<
            Frame = EthFrame,
            Context: ContextTr<Journal: JournalTr, Local: LocalContextTr>,
            Precompiles: PrecompileProvider<EVM::Context, Output = InterpreterResult>,
        >,
    <EVM::Context as ContextTr>::Cfg: Cfg<Spec: Into<SpecId>>,
    ERROR: EvmTrError<EVM>,
{
    #[inline]
    fn frame_run(&mut self, evm: &mut EVM) -> Result<FrameInitOrResult<EthFrame>, ERROR> {
        let spec_id: SpecId = evm.ctx_ref().cfg().spec().into();
        let frame = evm.frame_stack().get();
        let code_hash = frame.interpreter.bytecode.get_or_calculate_hash();

        let decision = self.lookup_cache.entry(code_hash).or_insert_with(|| {
            let code = frame.interpreter.bytecode.original_bytes();
            self.backend.lookup(LookupRequest { code_hash, code, spec_id })
        });

        Ok(match decision {
            LookupDecision::Compiled(program) => {
                let (ctx, _, _, frame_stack) = evm.all_mut();
                let frame = frame_stack.get();
                let action =
                    unsafe { program.func.call_with_interpreter(&mut frame.interpreter, ctx) };
                frame.process_next_action::<_, ERROR>(ctx, action).inspect(|i| {
                    if i.is_result() {
                        frame.set_finished(true);
                    }
                })?
            }
            LookupDecision::Interpret(_) => evm.frame_run()?,
        })
    }
}

impl<EVM, ERROR> Handler for JitHandler<'_, EVM, ERROR>
where
    EVM: EvmTr<
            Frame = EthFrame,
            Context: ContextTr<Journal: JournalTr, Local: LocalContextTr>,
            Precompiles: PrecompileProvider<EVM::Context, Output = InterpreterResult>,
        >,
    <EVM::Context as ContextTr>::Cfg: Cfg<Spec: Into<SpecId>>,
    ERROR: EvmTrError<EVM>,
{
    type Evm = EVM;
    type Error = ERROR;
    type HaltReason = HaltReason;

    #[inline]
    fn run_exec_loop(
        &mut self,
        evm: &mut Self::Evm,
        first_frame_input: FrameInit,
    ) -> Result<FrameResult, Self::Error> {
        let res = evm.frame_init(first_frame_input)?;

        if let ItemOrResult::Result(frame_result) = res {
            return Ok(frame_result);
        }

        loop {
            let call_or_result = self.frame_run(evm)?;

            let result = match call_or_result {
                ItemOrResult::Item(init) => match evm.frame_init(init)? {
                    ItemOrResult::Item(_) => {
                        continue;
                    }
                    ItemOrResult::Result(result) => result,
                },
                ItemOrResult::Result(result) => result,
            };

            if let Some(result) = evm.frame_return_result(result)? {
                return Ok(result);
            }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "llvm")]
mod tests {
    use super::*;
    use crate::runtime::{JitBackend, RuntimeConfig};
    use alloy_primitives::{Address, Bytes, TxKind, U256};
    use revm_bytecode::opcode as op;
    use revm_context::TxEnv;
    use revm_context_interface::result::{ExecutionResult, Output};
    use revm_database::{CacheDB, EmptyDB};
    use revm_database_interface::DatabaseCommit;
    use revm_handler::MainBuilder;

    fn blocking_backend() -> JitBackend {
        JitBackend::start(RuntimeConfig { blocking: true, ..Default::default() }).unwrap()
    }

    type TestInnerEvm = revm_handler::MainnetEvm<
        revm_context::Context<
            revm_context::BlockEnv,
            TxEnv,
            revm_context::CfgEnv,
            CacheDB<EmptyDB>,
        >,
    >;

    fn test_jit_evm(backend: JitBackend) -> JitEvm<TestInnerEvm> {
        let inner = revm_context::Context::new(CacheDB::default(), SpecId::CANCUN).build_mainnet();
        JitEvm::new(inner, backend)
    }

    fn deploy_contract(evm: &mut JitEvm<TestInnerEvm>, bytecode: &[u8]) -> Address {
        let len = bytecode.len();
        assert!(len <= 255);
        let offset = 10u8;
        let mut deploy_code = vec![
            op::PUSH1,
            len as u8,
            op::PUSH1,
            offset,
            op::PUSH0,
            op::CODECOPY,
            op::PUSH1,
            len as u8,
            op::PUSH0,
            op::RETURN,
        ];
        deploy_code.extend_from_slice(bytecode);

        let tx = TxEnv {
            kind: TxKind::Create,
            data: Bytes::copy_from_slice(&deploy_code),
            gas_limit: 1_000_000,
            ..Default::default()
        };

        let result = evm.transact_one(tx).unwrap();
        let state = evm.finalize();
        evm.inner_mut().db_mut().commit(state);
        match result {
            ExecutionResult::Success { output, .. } => match output {
                Output::Create(_, Some(addr)) => addr,
                other => panic!("expected Create output, got: {other:?}"),
            },
            other => panic!("expected Success, got: {other:?}"),
        }
    }

    #[test]
    fn jit_evm_simple_return() {
        let backend = blocking_backend();

        // PUSH1 0x42 PUSH0 MSTORE PUSH1 0x20 PUSH0 RETURN
        let runtime_code: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];

        let mut evm = test_jit_evm(backend.clone());
        let contract_addr = deploy_contract(&mut evm, runtime_code);

        let tx = TxEnv {
            kind: TxKind::Call(contract_addr),
            nonce: 1,
            gas_limit: 1_000_000,
            ..Default::default()
        };

        let result = evm.transact(tx).unwrap();
        match result.result {
            ExecutionResult::Success { output, .. } => {
                let data = match &output {
                    Output::Call(bytes) => bytes,
                    other => panic!("expected Call output, got: {other:?}"),
                };
                assert_eq!(U256::from_be_slice(data), U256::from(0x42));
            }
            other => panic!("expected Success, got: {other:?}"),
        }

        backend.shutdown().unwrap();
    }

    #[test]
    fn jit_evm_add() {
        let backend = blocking_backend();

        // PUSH1 1 PUSH1 2 ADD PUSH0 MSTORE PUSH1 0x20 PUSH0 RETURN
        let runtime_code: &[u8] =
            &[0x60, 0x01, 0x60, 0x02, 0x01, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];

        let mut evm = test_jit_evm(backend.clone());
        let contract_addr = deploy_contract(&mut evm, runtime_code);

        let tx = TxEnv {
            kind: TxKind::Call(contract_addr),
            nonce: 1,
            gas_limit: 1_000_000,
            ..Default::default()
        };

        let result = evm.transact(tx).unwrap();
        match result.result {
            ExecutionResult::Success { output, .. } => {
                let data = match &output {
                    Output::Call(bytes) => bytes,
                    other => panic!("expected Call output, got: {other:?}"),
                };
                assert_eq!(U256::from_be_slice(data), U256::from(3));
            }
            other => panic!("expected Success, got: {other:?}"),
        }

        backend.shutdown().unwrap();
    }

    #[test]
    fn jit_evm_fallback_empty_code() {
        let backend = blocking_backend();
        let mut evm = test_jit_evm(backend.clone());

        let tx = TxEnv {
            kind: TxKind::Call(Address::with_last_byte(0xEE)),
            gas_limit: 1_000_000,
            ..Default::default()
        };

        let result = evm.transact(tx).unwrap();
        assert!(
            matches!(result.result, ExecutionResult::Success { .. }),
            "expected Success for empty-code call, got: {:?}",
            result.result,
        );

        backend.shutdown().unwrap();
    }
}

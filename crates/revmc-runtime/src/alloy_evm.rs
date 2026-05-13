//! [`alloy_evm`] bindings for the revmc runtime.
//!
//! Provides [`JitEvm`] and [`JitEvmFactory`] which wrap the standard Ethereum EVM with
//! JIT-compiled function dispatch via [`JitBackend`].

use crate::{revm_evm, runtime::JitBackend};
use alloy_evm::{
    Database,
    env::EvmEnv,
    eth::{EthEvmContext, EthEvmFactory},
    evm::{Evm, EvmFactory},
    precompiles::PrecompilesMap,
};
use alloy_primitives::{Address, Bytes};
use core::error::Error;
use revm_context::{BlockEnv, CfgEnv, Evm as RevmEvm, TxEnv};
use revm_context_interface::{
    ContextTr,
    result::{EVMError, ExecutionResult, HaltReason, ResultAndState},
};
use revm_handler::{EthFrame, EvmTr, ExecuteEvm, SystemCallEvm, instructions::EthInstructions};
use revm_inspector::{InspectEvm, InspectSystemCallEvm, Inspector, InspectorEvmTr, NoOpInspector};
use revm_interpreter::interpreter::EthInterpreter;
use revm_primitives::hardfork::SpecId;
use revm_state::EvmState;

/// Inner EVM type that can be wrapped by [`JitEvm`].
pub trait JitEvmInner:
    EvmTr<
        Frame = EthFrame,
        Context: ContextTr<Db = Self::DB, Tx = TxEnv, Block = BlockEnv, Cfg = CfgEnv>,
    > + InspectorEvmTr
    + Sized
{
    /// Database type held by the inner EVM.
    type DB: Database;

    /// Consumes the EVM and returns its database and environment.
    fn finish(self) -> (Self::DB, EvmEnv);
}

impl<DB, I, INSTRUCTIONS, P, F> JitEvmInner for RevmEvm<EthEvmContext<DB>, I, INSTRUCTIONS, P, F>
where
    DB: Database,
    EthEvmContext<DB>: ContextTr<
            Db = DB,
            Tx = TxEnv,
            Block = BlockEnv,
            Cfg = CfgEnv,
            Journal: revm_inspector::JournalExt,
        >,
    Self: EvmTr<Frame = EthFrame, Context = EthEvmContext<DB>, Precompiles = P> + InspectorEvmTr,
{
    type DB = DB;

    fn finish(self) -> (DB, EvmEnv) {
        let revm_context::Context { block: block_env, cfg: cfg_env, journaled_state, .. } =
            self.ctx;
        (journaled_state.database, EvmEnv { block_env, cfg_env })
    }
}

type DbError<EVM> = <<EVM as JitEvmInner>::DB as revm_context_interface::Database>::Error;
type EthInnerEvm<DB, I> = RevmEvm<
    EthEvmContext<DB>,
    I,
    EthInstructions<EthInterpreter, EthEvmContext<DB>>,
    PrecompilesMap,
    EthFrame,
>;

/// EVM with JIT-compiled function dispatch.
///
/// Wraps a revm EVM inside a [`revm_evm::JitEvm`] which overrides `frame_run`
/// to look up compiled functions via [`JitBackend`] before falling back to the interpreter.
#[derive(derive_more::Debug)]
pub struct JitEvm<EVM> {
    #[debug(skip)]
    inner: revm_evm::JitEvm<EVM>,
    inspect: bool,
}

impl<EVM> JitEvm<EVM>
where
    EVM: EvmTr,
    EVM::Context: ContextTr,
{
    /// Creates a new JIT EVM from an inner revm EVM, inspect flag, and backend.
    pub fn new(inner: EVM, inspect: bool, backend: JitBackend) -> Self {
        Self { inner: revm_evm::JitEvm::new(inner, backend), inspect }
    }
}

impl<EVM> JitEvm<EVM> {
    /// Returns a reference to the inner EVM.
    pub const fn inner(&self) -> &EVM {
        self.inner.inner()
    }

    /// Returns a mutable reference to the inner EVM.
    pub fn inner_mut(&mut self) -> &mut EVM {
        self.inner.inner_mut()
    }

    /// Consumes the JIT EVM and returns the inner EVM.
    pub fn into_inner(self) -> EVM {
        self.inner.into_inner()
    }

    /// Returns a reference to the JIT backend.
    pub const fn backend(&self) -> &JitBackend {
        self.inner.backend()
    }
}

impl<EVM> core::ops::Deref for JitEvm<EVM> {
    type Target = EVM;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner.inner()
    }
}

impl<EVM> core::ops::DerefMut for JitEvm<EVM> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.inner_mut()
    }
}

impl<EVM> Evm for JitEvm<EVM>
where
    EVM: JitEvmInner,
    revm_evm::JitEvm<EVM>: ExecuteEvm<
            ExecutionResult = ExecutionResult<HaltReason>,
            State = EvmState,
            Error = EVMError<DbError<EVM>>,
            Tx = TxEnv,
            Block = BlockEnv,
        > + InspectEvm<Inspector = <EVM as InspectorEvmTr>::Inspector>
        + SystemCallEvm
        + InspectSystemCallEvm,
{
    type DB = EVM::DB;
    type Tx = TxEnv;
    type Error = EVMError<DbError<EVM>>;
    type HaltReason = HaltReason;
    type Spec = SpecId;
    type BlockEnv = BlockEnv;
    type Precompiles = <EVM as EvmTr>::Precompiles;
    type Inspector = <EVM as InspectorEvmTr>::Inspector;

    fn block(&self) -> &Self::BlockEnv {
        self.inner.ctx_ref().block()
    }

    fn cfg_env(&self) -> &revm_context::CfgEnv<Self::Spec> {
        self.inner.ctx_ref().cfg()
    }

    fn chain_id(&self) -> u64 {
        self.inner.ctx_ref().cfg().chain_id
    }

    fn transact_raw(
        &mut self,
        tx: Self::Tx,
    ) -> Result<ResultAndState<Self::HaltReason>, Self::Error> {
        if self.inspect { self.inner.inspect_tx(tx) } else { self.inner.transact(tx) }
    }

    fn transact_system_call(
        &mut self,
        caller: Address,
        contract: Address,
        data: Bytes,
    ) -> Result<ResultAndState<Self::HaltReason>, Self::Error> {
        if self.inspect {
            self.inner.inspect_system_call_with_caller(caller, contract, data)
        } else {
            self.inner.system_call_with_caller(caller, contract, data)
        }
    }

    fn finish(self) -> (Self::DB, EvmEnv<Self::Spec, Self::BlockEnv>) {
        self.into_inner().finish()
    }

    fn set_inspector_enabled(&mut self, enabled: bool) {
        self.inspect = enabled;
    }

    fn components(&self) -> (&Self::DB, &Self::Inspector, &Self::Precompiles) {
        let (ctx, _, precompiles, _, inspector) = self.inner.inner().all_inspector();
        (ctx.db(), inspector, precompiles)
    }

    fn components_mut(&mut self) -> (&mut Self::DB, &mut Self::Inspector, &mut Self::Precompiles) {
        let (ctx, _, precompiles, _, inspector) = self.inner.inner_mut().all_mut_inspector();
        (ctx.db_mut(), inspector, precompiles)
    }
}

/// Factory producing [`JitEvm`] instances.
#[derive(Clone, Debug)]
pub struct JitEvmFactory<F = EthEvmFactory> {
    inner: F,
    backend: JitBackend,
}

impl<F> JitEvmFactory<F> {
    /// Creates a new factory from an inner EVM factory and JIT backend.
    pub fn new(inner: F, backend: JitBackend) -> Self {
        Self { inner, backend }
    }

    /// Returns a reference to the inner EVM factory.
    pub const fn inner(&self) -> &F {
        &self.inner
    }

    /// Returns a reference to the JIT backend.
    pub const fn backend(&self) -> &JitBackend {
        &self.backend
    }
}

/// Factory whose produced EVM can be wrapped by [`JitEvm`].
pub trait JitEvmFactoryInner {
    /// The inner EVM type to wrap.
    type InnerEvm<DB: Database, I: Inspector<EthEvmContext<DB>>>: JitEvmInner<
            DB = DB,
            Context = EthEvmContext<DB>,
            Precompiles = PrecompilesMap,
            Inspector = I,
        >;

    /// Creates the inner EVM.
    fn create_inner_evm<DB: Database>(
        &self,
        db: DB,
        input: EvmEnv,
    ) -> Self::InnerEvm<DB, NoOpInspector>;

    /// Creates the inner EVM with an inspector.
    fn create_inner_evm_with_inspector<DB: Database, I: Inspector<EthEvmContext<DB>>>(
        &self,
        db: DB,
        input: EvmEnv,
        inspector: I,
    ) -> Self::InnerEvm<DB, I>;
}

impl JitEvmFactoryInner for EthEvmFactory {
    type InnerEvm<DB: Database, I: Inspector<EthEvmContext<DB>>> = EthInnerEvm<DB, I>;

    fn create_inner_evm<DB: Database>(
        &self,
        db: DB,
        input: EvmEnv,
    ) -> Self::InnerEvm<DB, NoOpInspector> {
        self.create_evm(db, input).into_inner()
    }

    fn create_inner_evm_with_inspector<DB: Database, I: Inspector<EthEvmContext<DB>>>(
        &self,
        db: DB,
        input: EvmEnv,
        inspector: I,
    ) -> Self::InnerEvm<DB, I> {
        self.create_evm_with_inspector(db, input, inspector).into_inner()
    }
}

impl<F> EvmFactory for JitEvmFactory<F>
where
    F: JitEvmFactoryInner,
{
    type Evm<DB: Database, I: Inspector<Self::Context<DB>>> = JitEvm<F::InnerEvm<DB, I>>;
    type Context<DB: Database> = EthEvmContext<DB>;
    type Tx = TxEnv;
    type Error<DBError: Error + Send + Sync + 'static> = EVMError<DBError>;
    type HaltReason = HaltReason;
    type Spec = SpecId;
    type BlockEnv = BlockEnv;
    type Precompiles = PrecompilesMap;

    fn create_evm<DB: Database>(&self, db: DB, input: EvmEnv) -> Self::Evm<DB, NoOpInspector> {
        let inner = self.inner.create_inner_evm(db, input);
        JitEvm::new(inner, false, self.backend.clone())
    }

    fn create_evm_with_inspector<DB: Database, I: Inspector<Self::Context<DB>>>(
        &self,
        db: DB,
        input: EvmEnv,
        inspector: I,
    ) -> Self::Evm<DB, I> {
        let inner = self.inner.create_inner_evm_with_inspector(db, input, inspector);
        JitEvm::new(inner, true, self.backend.clone())
    }
}

#[cfg(test)]
#[cfg(feature = "llvm")]
mod tests {
    use super::*;
    use crate::runtime::{JitBackend, RuntimeConfig};
    use alloy_evm::{env::EvmEnv, eth::EthEvmFactory};
    use alloy_primitives::{Address, Bytes, TxKind, U256};
    use revm_bytecode::opcode as op;
    use revm_context::TxEnv;
    use revm_context_interface::result::{ExecutionResult, Output};
    use revm_database::{CacheDB, EmptyDB};
    use revm_database_interface::DatabaseCommit;

    fn blocking_backend() -> JitBackend {
        JitBackend::new(RuntimeConfig { blocking: true, ..Default::default() }).unwrap()
    }

    fn deploy_contract<
        E: alloy_evm::Evm<DB = CacheDB<EmptyDB>, Tx = TxEnv, Error: std::fmt::Debug>,
    >(
        evm: &mut E,
        bytecode: &[u8],
    ) -> Address
    where
        E::HaltReason: std::fmt::Debug,
    {
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

        let result = evm.transact_raw(tx).unwrap();
        evm.db_mut().commit(result.state);
        match result.result {
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

        let factory = JitEvmFactory::new(EthEvmFactory::default(), backend.clone());
        let mut evm = factory.create_evm(CacheDB::<EmptyDB>::default(), EvmEnv::default());
        let contract_addr = deploy_contract(&mut evm, runtime_code);

        let tx = TxEnv {
            kind: TxKind::Call(contract_addr),
            nonce: 1,
            gas_limit: 1_000_000,
            ..Default::default()
        };

        let result = evm.transact_raw(tx).unwrap();
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

        let stats = backend.stats();
        assert!(stats.compilations_succeeded > 0, "expected compilations, got: {stats:?}");
        assert!(stats.resident_entries > 0, "expected resident entries, got: {stats:?}");
    }

    #[test]
    fn jit_evm_vs_eth_evm() {
        let backend = blocking_backend();

        // PUSH1 1 PUSH1 2 ADD PUSH0 MSTORE PUSH1 0x20 PUSH0 RETURN
        let runtime_code: &[u8] =
            &[0x60, 0x01, 0x60, 0x02, 0x01, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];

        // Run via JitEvmFactory.
        let jit_result = {
            let factory = JitEvmFactory::new(EthEvmFactory::default(), backend);
            let mut evm = factory.create_evm(CacheDB::<EmptyDB>::default(), EvmEnv::default());
            let addr = deploy_contract(&mut evm, runtime_code);
            evm.transact_raw(TxEnv {
                kind: TxKind::Call(addr),
                nonce: 1,
                gas_limit: 1_000_000,
                ..Default::default()
            })
            .unwrap()
        };

        // Run via standard EthEvmFactory.
        let eth_result = {
            let factory = EthEvmFactory::default();
            let mut evm = factory.create_evm(CacheDB::<EmptyDB>::default(), EvmEnv::default());
            let addr = deploy_contract(&mut evm, runtime_code);
            evm.transact_raw(TxEnv {
                kind: TxKind::Call(addr),
                nonce: 1,
                gas_limit: 1_000_000,
                ..Default::default()
            })
            .unwrap()
        };

        // Both should produce the same output.
        assert_eq!(format!("{:?}", jit_result.result), format!("{:?}", eth_result.result),);
    }

    #[test]
    fn jit_evm_factory_roundtrip() {
        let backend = blocking_backend();
        let factory = JitEvmFactory::new(EthEvmFactory::default(), backend);

        let mut evm = factory.create_evm(CacheDB::<EmptyDB>::default(), EvmEnv::default());
        assert_eq!(evm.chain_id(), 1);

        let tx = TxEnv {
            kind: TxKind::Call(Address::with_last_byte(0xDD)),
            gas_limit: 1_000_000,
            ..Default::default()
        };
        let result = evm.transact_raw(tx).unwrap();
        assert!(
            matches!(result.result, ExecutionResult::Success { .. }),
            "expected Success, got: {:?}",
            result.result,
        );
    }
}

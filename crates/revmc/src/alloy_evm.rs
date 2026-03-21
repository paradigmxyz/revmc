//! [`alloy_evm`] bindings for the revmc runtime.
//!
//! Provides [`JitEvm`] and [`JitEvmFactory`] which wrap the standard Ethereum EVM with
//! JIT-compiled function dispatch via [`JitCoordinatorHandle`].

use crate::runtime::{JitCoordinatorHandle, LookupDecision, LookupRequest};
use alloy_evm::{
    Database,
    env::EvmEnv,
    eth::{EthEvmBuilder, EthEvmContext},
    evm::{Evm, EvmFactory},
    precompiles::PrecompilesMap,
};
use alloy_primitives::{Address, Bytes};
use core::marker::PhantomData;
use revm::{
    ExecuteEvm, InspectEvm, Inspector, SystemCallEvm,
    context::{BlockEnv, ContextSetters, Evm as RevmEvm, TxEnv},
    context_interface::result::{EVMError, HaltReason, ResultAndState},
    handler::{
        EthFrame, EvmTr, FrameResult, Handler, ItemOrResult, PrecompileProvider,
        instructions::EthInstructions,
    },
    inspector::NoOpInspector,
    interpreter::{InterpreterResult, interpreter::EthInterpreter, interpreter_action::FrameInit},
    primitives::hardfork::SpecId,
};

type InnerEvm<DB, I, P> =
    RevmEvm<EthEvmContext<DB>, I, EthInstructions<EthInterpreter, EthEvmContext<DB>>, P, EthFrame>;

/// Ethereum EVM with JIT-compiled function dispatch.
///
/// Wraps the standard revm EVM and overrides execution to look up compiled functions via
/// [`JitCoordinatorHandle`] before falling back to the interpreter.
#[expect(missing_debug_implementations)]
pub struct JitEvm<DB: Database, I, PRECOMPILE = PrecompilesMap> {
    inner: InnerEvm<DB, I, PRECOMPILE>,
    inspect: bool,
    handle: JitCoordinatorHandle,
}

impl<DB: Database, I, P> JitEvm<DB, I, P> {
    /// Creates a new JIT EVM from an inner revm EVM and coordinator handle.
    pub const fn new(
        inner: InnerEvm<DB, I, P>,
        inspect: bool,
        handle: JitCoordinatorHandle,
    ) -> Self {
        Self { inner, inspect, handle }
    }

    /// Returns a reference to the coordinator handle.
    pub const fn handle(&self) -> &JitCoordinatorHandle {
        &self.handle
    }
}

impl<DB: Database, I, P> core::ops::Deref for JitEvm<DB, I, P> {
    type Target = EthEvmContext<DB>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner.ctx
    }
}

impl<DB: Database, I, P> core::ops::DerefMut for JitEvm<DB, I, P> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner.ctx
    }
}

impl<DB, I, P> Evm for JitEvm<DB, I, P>
where
    DB: Database,
    I: Inspector<EthEvmContext<DB>>,
    P: PrecompileProvider<EthEvmContext<DB>, Output = InterpreterResult>,
{
    type DB = DB;
    type Tx = TxEnv;
    type Error = EVMError<DB::Error>;
    type HaltReason = HaltReason;
    type Spec = SpecId;
    type BlockEnv = BlockEnv;
    type Precompiles = P;
    type Inspector = I;

    fn block(&self) -> &BlockEnv {
        &self.inner.ctx.block
    }

    fn chain_id(&self) -> u64 {
        self.inner.ctx.cfg.chain_id
    }

    fn transact_raw(
        &mut self,
        tx: Self::Tx,
    ) -> Result<ResultAndState<Self::HaltReason>, Self::Error> {
        if self.inspect {
            self.inner.inspect_tx(tx)
        } else {
            self.inner.ctx.set_tx(tx);
            let mut handler: JitHandler<'_, DB, I, P> =
                JitHandler { handle: &self.handle, _pd: PhantomData };
            handler.run(&mut self.inner).map(|result| {
                let state = self.inner.finalize();
                ResultAndState::new(result, state)
            })
        }
    }

    fn transact_system_call(
        &mut self,
        caller: Address,
        contract: Address,
        data: Bytes,
    ) -> Result<ResultAndState<Self::HaltReason>, Self::Error> {
        self.inner.system_call_with_caller(caller, contract, data)
    }

    fn finish(self) -> (Self::DB, EvmEnv<Self::Spec>) {
        let revm::Context { block: block_env, cfg: cfg_env, journaled_state, .. } = self.inner.ctx;
        (journaled_state.database, EvmEnv { block_env, cfg_env })
    }

    fn set_inspector_enabled(&mut self, enabled: bool) {
        self.inspect = enabled;
    }

    fn components(&self) -> (&Self::DB, &Self::Inspector, &Self::Precompiles) {
        (&self.inner.ctx.journaled_state.database, &self.inner.inspector, &self.inner.precompiles)
    }

    fn components_mut(&mut self) -> (&mut Self::DB, &mut Self::Inspector, &mut Self::Precompiles) {
        (
            &mut self.inner.ctx.journaled_state.database,
            &mut self.inner.inspector,
            &mut self.inner.precompiles,
        )
    }
}

/// Factory producing [`JitEvm`] instances.
#[derive(Clone)]
#[expect(missing_debug_implementations)]
pub struct JitEvmFactory {
    handle: JitCoordinatorHandle,
}

impl JitEvmFactory {
    /// Creates a new factory from a coordinator handle.
    pub const fn new(handle: JitCoordinatorHandle) -> Self {
        Self { handle }
    }
}

impl EvmFactory for JitEvmFactory {
    type Evm<DB: Database, I: Inspector<EthEvmContext<DB>>> = JitEvm<DB, I, PrecompilesMap>;
    type Context<DB: Database> = EthEvmContext<DB>;
    type Tx = TxEnv;
    type Error<DBError: core::error::Error + Send + Sync + 'static> = EVMError<DBError>;
    type HaltReason = HaltReason;
    type Spec = SpecId;
    type BlockEnv = BlockEnv;
    type Precompiles = PrecompilesMap;

    fn create_evm<DB: Database>(&self, db: DB, input: EvmEnv) -> Self::Evm<DB, NoOpInspector> {
        let inner = EthEvmBuilder::new(db, input).build().into_inner();
        JitEvm::new(inner, false, self.handle.clone())
    }

    fn create_evm_with_inspector<DB: Database, I: Inspector<Self::Context<DB>>>(
        &self,
        db: DB,
        input: EvmEnv,
        inspector: I,
    ) -> Self::Evm<DB, I> {
        let inner =
            EthEvmBuilder::new(db, input).activate_inspector(inspector).build().into_inner();
        JitEvm::new(inner, true, self.handle.clone())
    }
}

/// Custom handler that overrides only `run_exec_loop` with JIT dispatch.
struct JitHandler<'a, DB: Database, I, P> {
    handle: &'a JitCoordinatorHandle,
    _pd: PhantomData<(DB, I, P)>,
}

impl<DB, I, P> Handler for JitHandler<'_, DB, I, P>
where
    DB: Database,
    I: Inspector<EthEvmContext<DB>>,
    P: PrecompileProvider<EthEvmContext<DB>, Output = InterpreterResult>,
{
    type Evm = InnerEvm<DB, I, P>;
    type Error = EVMError<DB::Error>;
    type HaltReason = HaltReason;

    fn run_exec_loop(
        &mut self,
        evm: &mut Self::Evm,
        first_frame_input: FrameInit,
    ) -> Result<FrameResult, Self::Error> {
        let res = evm.frame_init(first_frame_input)?;
        if let ItemOrResult::Result(frame_result) = res {
            return Ok(frame_result);
        }

        let spec_id = evm.ctx.cfg.spec;

        loop {
            let call_or_result = {
                let frame = evm.frame_stack.get();
                let bytecode_hash = frame.interpreter.bytecode.get_or_calculate_hash();
                let code = frame.interpreter.bytecode.original_byte_slice();

                let req = LookupRequest { code_hash: bytecode_hash, code, spec_id };
                match self.handle.lookup(req) {
                    LookupDecision::Compiled(program) => {
                        let ctx = &mut evm.ctx;
                        let action = unsafe {
                            program.func.call_with_interpreter(&mut frame.interpreter, ctx)
                        };
                        frame.process_next_action::<_, Self::Error>(ctx, action).inspect(|i| {
                            if i.is_result() {
                                frame.set_finished(true);
                            }
                        })?
                    }
                    LookupDecision::Interpret(_) => evm.frame_run()?,
                }
            };

            let result = match call_or_result {
                ItemOrResult::Item(init) => match evm.frame_init(init)? {
                    ItemOrResult::Item(_) => continue,
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

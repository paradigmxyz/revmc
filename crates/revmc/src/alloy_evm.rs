//! [`alloy_evm`] bindings for the revmc runtime.
//!
//! Provides [`JitEvm`] and [`JitEvmFactory`] which wrap the standard Ethereum EVM with
//! JIT-compiled function dispatch via [`JitBackend`].

use crate::runtime::{JitBackend, LookupDecision, LookupRequest};
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
        EthFrame, EvmTr, FrameInitOrResult, FrameResult, Handler, ItemOrResult, PrecompileProvider,
        instructions::EthInstructions,
    },
    inspector::NoOpInspector,
    interpreter::{InterpreterResult, interpreter::EthInterpreter, interpreter_action::FrameInit},
    primitives::{B256Map, hardfork::SpecId},
};

type InnerEvm<DB, I, P> =
    RevmEvm<EthEvmContext<DB>, I, EthInstructions<EthInterpreter, EthEvmContext<DB>>, P, EthFrame>;

/// Ethereum EVM with JIT-compiled function dispatch.
///
/// Wraps the standard revm EVM and overrides execution to look up compiled functions via
/// [`JitBackend`] before falling back to the interpreter.
#[expect(missing_debug_implementations)]
pub struct JitEvm<DB: Database, I, PRECOMPILE = PrecompilesMap> {
    inner: InnerEvm<DB, I, PRECOMPILE>,
    inspect: bool,
    backend: JitBackend,

    /// Cached lookup decisions keyed by `code_hash` alone.
    /// Invalidated when the `spec_id` changes.
    lookup_cache: B256Map<LookupDecision>,
    /// The `spec_id` the cache was built for; cleared on mismatch.
    lookup_cache_spec_id: SpecId,
}

impl<DB: Database, I, P> JitEvm<DB, I, P> {
    /// Creates a new JIT EVM from an inner revm EVM and backend.
    pub fn new(inner: InnerEvm<DB, I, P>, inspect: bool, backend: JitBackend) -> Self {
        let spec_id = inner.cfg.spec;
        Self {
            inner,
            inspect,
            backend,
            lookup_cache: B256Map::with_capacity_and_hasher(16, Default::default()),
            lookup_cache_spec_id: spec_id,
        }
    }

    /// Returns a reference to the JIT backend.
    pub const fn backend(&self) -> &JitBackend {
        &self.backend
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
            let spec_id = self.inner.ctx.cfg.spec;
            if spec_id != self.lookup_cache_spec_id {
                self.lookup_cache.clear();
                self.lookup_cache_spec_id = spec_id;
            }
            JitHandler::new(&self.backend, &mut self.lookup_cache).run(&mut self.inner).map(|r| {
                let state = self.inner.finalize();
                ResultAndState::new(r, state)
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
    backend: JitBackend,
}

impl JitEvmFactory {
    /// Creates a new factory from a JIT backend.
    pub const fn new(backend: JitBackend) -> Self {
        Self { backend }
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
        JitEvm::new(inner, false, self.backend.clone())
    }

    fn create_evm_with_inspector<DB: Database, I: Inspector<Self::Context<DB>>>(
        &self,
        db: DB,
        input: EvmEnv,
        inspector: I,
    ) -> Self::Evm<DB, I> {
        let inner =
            EthEvmBuilder::new(db, input).activate_inspector(inspector).build().into_inner();
        JitEvm::new(inner, true, self.backend.clone())
    }
}

/// Custom handler that overrides only `run_exec_loop` with JIT dispatch.
struct JitHandler<'a, DB: Database, I, P> {
    backend: &'a JitBackend,
    /// Lookup decision cache borrowed from [`JitEvm`], keyed by `code_hash` alone.
    /// The `spec_id` is constant within an execution; the caller invalidates on change.
    lookup_cache: &'a mut B256Map<LookupDecision>,
    _pd: PhantomData<(DB, I, P)>,
}

impl<'a, DB, I, P> JitHandler<'a, DB, I, P>
where
    DB: Database,
    I: Inspector<EthEvmContext<DB>>,
    P: PrecompileProvider<EthEvmContext<DB>, Output = InterpreterResult>,
{
    #[inline]
    fn new(backend: &'a JitBackend, lookup_cache: &'a mut B256Map<LookupDecision>) -> Self {
        Self { backend, lookup_cache, _pd: PhantomData }
    }

    #[inline]
    fn frame_run(
        &mut self,
        evm: &mut <Self as Handler>::Evm,
    ) -> Result<FrameInitOrResult<<<Self as Handler>::Evm as EvmTr>::Frame>, <Self as Handler>::Error>
    {
        let spec_id = evm.cfg.spec;
        let frame = evm.frame_stack.get();
        let code_hash = frame.interpreter.bytecode.get_or_calculate_hash();

        // let mut cache_hit = true;
        let decision = self.lookup_cache.entry(code_hash).or_insert_with(|| {
            // cache_hit = false;
            let code = frame.interpreter.bytecode.original_bytes();
            self.backend.lookup(LookupRequest { code_hash, code, spec_id })
        });

        // if cache_hit && matches!(decision, LookupDecision::Interpret(InterpretReason::NotReady))
        // {     let code = frame.interpreter.bytecode.original_bytes();
        //     *decision = self.backend.lookup(LookupRequest { code_hash, code, spec_id });
        // }

        Ok(match decision {
            LookupDecision::Compiled(program) => {
                let ctx = &mut evm.ctx;
                let action =
                    unsafe { program.func.call_with_interpreter(&mut frame.interpreter, ctx) };
                frame.process_next_action::<_, <Self as Handler>::Error>(ctx, action).inspect(
                    |i| {
                        if i.is_result() {
                            frame.set_finished(true);
                        }
                    },
                )?
            }
            LookupDecision::Interpret(_) => evm.frame_run()?,
        })
    }
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
                ItemOrResult::Item(init) => {
                    match evm.frame_init(init)? {
                        ItemOrResult::Item(_) => {
                            continue;
                        }
                        // Do not pop the frame since no new frame was created
                        ItemOrResult::Result(result) => result,
                    }
                }
                ItemOrResult::Result(result) => result,
            };

            if let Some(result) = evm.frame_return_result(result)? {
                return Ok(result);
            }
        }
    }
}

//! Generic `revm` JIT EVM.
//!
//! Provides [`JitEvm`] which wraps any [`EvmTr`]-based EVM and overrides
//! `frame_run` to dispatch to JIT-compiled functions by code hash,
//! falling back to the interpreter for unknown contracts.

use crate::{EvmCompilerFn, RawEvmCompilerFn};
use revm_context_interface::{
    ContextSetters, ContextTr,
    journaled_state::JournalTr,
    result::{EVMError, HaltReason, InvalidTransaction, ResultAndState},
};
use revm_handler::{
    EthFrame, EvmTr, ExecuteEvm, FrameInitOrResult, FrameTr, Handler, ItemOrResult, MainnetHandler,
    PrecompileProvider, evm::ContextDbError,
};
use revm_interpreter::InterpreterResult;
use revm_primitives::{B256, map::B256Map};
use revm_state::EvmState;

/// Wrapper around any [`EvmTr`] that overrides [`EvmTr::frame_run`] to dispatch
/// to JIT-compiled functions by code hash, falling back to the interpreter.
///
/// The `F` parameter is a lookup function `(B256, &[u8]) -> Option<EvmCompilerFn>` called
/// when a code hash is not found in the precompiled map. Return `None` to fall
/// back to the interpreter; return `Some(f)` to execute `f` (e.g. compile on the fly).
#[allow(missing_debug_implementations)]
pub struct JitEvm<EVM, F = fn(B256, &[u8]) -> Option<EvmCompilerFn>> {
    inner: EVM,
    functions: B256Map<RawEvmCompilerFn>,
    on_miss: F,
}

fn no_miss(_: B256, _: &[u8]) -> Option<EvmCompilerFn> {
    None
}

impl<EVM> JitEvm<EVM> {
    /// Create a new JIT EVM wrapper that falls back to the interpreter on miss.
    pub fn new(inner: EVM, functions: B256Map<RawEvmCompilerFn>) -> Self {
        Self { inner, functions, on_miss: no_miss }
    }
}

impl<EVM, F> JitEvm<EVM, F> {
    /// Create a new JIT EVM wrapper with a custom miss handler.
    pub fn with_on_miss(inner: EVM, functions: B256Map<RawEvmCompilerFn>, on_miss: F) -> Self {
        Self { inner, functions, on_miss }
    }

    /// Consumes the wrapper and returns the inner EVM.
    pub fn into_inner(self) -> EVM {
        self.inner
    }
}

impl<EVM, F> core::ops::Deref for JitEvm<EVM, F> {
    type Target = EVM;
    fn deref(&self) -> &EVM {
        &self.inner
    }
}

impl<EVM, F> core::ops::DerefMut for JitEvm<EVM, F> {
    fn deref_mut(&mut self) -> &mut EVM {
        &mut self.inner
    }
}

impl<EVM, F> EvmTr for JitEvm<EVM, F>
where
    EVM: EvmTr<
            Frame = EthFrame,
            Precompiles: PrecompileProvider<EVM::Context, Output = InterpreterResult>,
        >,
    F: FnMut(B256, &[u8]) -> Option<EvmCompilerFn>,
{
    type Context = EVM::Context;
    type Instructions = EVM::Instructions;
    type Precompiles = EVM::Precompiles;
    type Frame = EVM::Frame;

    fn all(
        &self,
    ) -> (
        &Self::Context,
        &Self::Instructions,
        &Self::Precompiles,
        &revm_context_interface::FrameStack<Self::Frame>,
    ) {
        self.inner.all()
    }

    fn all_mut(
        &mut self,
    ) -> (
        &mut Self::Context,
        &mut Self::Instructions,
        &mut Self::Precompiles,
        &mut revm_context_interface::FrameStack<Self::Frame>,
    ) {
        self.inner.all_mut()
    }

    fn frame_init(
        &mut self,
        frame_input: <Self::Frame as FrameTr>::FrameInit,
    ) -> Result<
        ItemOrResult<&mut Self::Frame, <Self::Frame as FrameTr>::FrameResult>,
        ContextDbError<Self::Context>,
    > {
        self.inner.frame_init(frame_input)
    }

    fn frame_run(
        &mut self,
    ) -> Result<FrameInitOrResult<Self::Frame>, ContextDbError<Self::Context>> {
        let frame = self.inner.frame_stack().get();
        let code_hash = frame.interpreter.bytecode.get_or_calculate_hash();

        let f = if let Some(&raw_fn) = self.functions.get(&code_hash) {
            Some(EvmCompilerFn::new(raw_fn))
        } else {
            let code = frame.interpreter.bytecode.original_bytes();
            (self.on_miss)(code_hash, &code)
        };

        if let Some(f) = f {
            let (ctx, _, _, frame_stack) = self.inner.all_mut();
            let frame = frame_stack.get();
            let action = unsafe { f.call_with_interpreter(&mut frame.interpreter, ctx) };
            Ok(frame.process_next_action::<_, ContextDbError<Self::Context>>(ctx, action).inspect(
                |i| {
                    if i.is_result() {
                        frame.set_finished(true);
                    }
                },
            )?)
        } else {
            self.inner.frame_run()
        }
    }

    fn frame_return_result(
        &mut self,
        result: <Self::Frame as FrameTr>::FrameResult,
    ) -> Result<Option<<Self::Frame as FrameTr>::FrameResult>, ContextDbError<Self::Context>> {
        self.inner.frame_return_result(result)
    }
}

impl<EVM, F> ExecuteEvm for JitEvm<EVM, F>
where
    EVM: EvmTr<
            Frame = EthFrame,
            Context: ContextTr<Journal: JournalTr<State = EvmState>> + ContextSetters,
            Precompiles: PrecompileProvider<EVM::Context, Output = InterpreterResult>,
        >,
    F: FnMut(B256, &[u8]) -> Option<EvmCompilerFn>,
{
    type ExecutionResult = revm_context_interface::result::ExecutionResult<HaltReason>;
    type State = EvmState;
    type Error = EVMError<
        <<EVM::Context as ContextTr>::Db as revm_context_interface::Database>::Error,
        InvalidTransaction,
    >;
    type Tx = <EVM::Context as ContextTr>::Tx;
    type Block = <EVM::Context as ContextTr>::Block;

    fn transact_one(&mut self, tx: Self::Tx) -> Result<Self::ExecutionResult, Self::Error> {
        self.ctx_mut().set_tx(tx);
        MainnetHandler::default().run(self)
    }

    fn finalize(&mut self) -> Self::State {
        self.ctx_mut().journal_mut().finalize()
    }

    fn set_block(&mut self, block: Self::Block) {
        self.ctx_mut().set_block(block);
    }

    fn replay(&mut self) -> Result<ResultAndState<HaltReason>, Self::Error> {
        MainnetHandler::default().run(self).map(|result| {
            let state = self.ctx_mut().journal_mut().finalize();
            ResultAndState::new(result, state)
        })
    }
}

//! Generic [`revm`] handler override with JIT dispatch.
//!
//! Provides [`JitHandler`] which wraps any [`EvmTr`]-based EVM and overrides
//! `frame_run` to look up compiled functions via [`JitBackend`] before falling
//! back to the interpreter.

use crate::runtime::{JitBackend, LookupDecision, LookupRequest};
use core::marker::PhantomData;
use revm::{
    context_interface::{
        Cfg, ContextTr, journaled_state::JournalTr, local::LocalContextTr, result::HaltReason,
    },
    handler::{
        EthFrame, EvmTr, EvmTrError, FrameInitOrResult, FrameResult, Handler, ItemOrResult,
        PrecompileProvider,
    },
    interpreter::{InterpreterResult, interpreter_action::FrameInit},
    primitives::{B256Map, hardfork::SpecId},
};

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

        // Peek at the frame to populate the lookup cache without holding a long borrow.
        let code_hash = evm.frame_stack().get().interpreter.bytecode.get_or_calculate_hash();
        let is_compiled = self.lookup_cache.contains_key(&code_hash)
            && matches!(self.lookup_cache.get(&code_hash), Some(LookupDecision::Compiled(_)));

        if !is_compiled {
            // Populate cache if missing.
            if !self.lookup_cache.contains_key(&code_hash) {
                let code = evm.frame_stack().get().interpreter.bytecode.original_bytes();
                let decision = self.backend.lookup(LookupRequest { code_hash, code, spec_id });
                self.lookup_cache.insert(code_hash, decision);
            }

            if !matches!(self.lookup_cache.get(&code_hash), Some(LookupDecision::Compiled(_))) {
                return Ok(evm.frame_run()?);
            }
        }

        // We know there's a compiled program — get simultaneous ctx + frame access.
        let program = match self.lookup_cache.get(&code_hash) {
            Some(LookupDecision::Compiled(p)) => p.clone(),
            _ => unreachable!(),
        };
        let (ctx, _, _, frame_stack) = evm.all_mut();
        let frame = frame_stack.get();
        let action = unsafe { program.func.call_with_interpreter(&mut frame.interpreter, ctx) };
        frame.process_next_action::<_, ERROR>(ctx, action).inspect(|i| {
            if i.is_result() {
                frame.set_finished(true);
            }
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

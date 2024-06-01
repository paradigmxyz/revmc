//! Minimal, `no_std` runner.

#![no_std]

extern crate alloc;

// This dependency is needed to export the necessary symbols used by the compiled bytecodes,
// but we don't use it directly, so silence the unused crate depedency warning.
use revm_jit_builtins as _;

use alloc::sync::Arc;
use revm::{handler::register::EvmHandler, primitives::B256, Database};
use revm_jit_context::EvmCompilerFn;

pub fn get_evm<'a, DB: Database + 'static>(db: DB) -> revm::Evm<'a, ExternalContext, DB> {
    revm::Evm::builder()
        .with_db(db)
        .with_external_context(ExternalContext::new())
        .append_handler_register(register_handler)
        .build()
}

pub struct ExternalContext;

impl ExternalContext {
    fn new() -> Self {
        Self
    }

    fn get_function(&self, bytecode_hash: B256) -> Option<EvmCompilerFn> {
        // Some way to get the function, either linked statically or dynamically.
        let _ = bytecode_hash;
        None
    }
}

// This `+ 'static` bound is only necessary here because of an internal cfg feature.
fn register_handler<DB: Database + 'static>(handler: &mut EvmHandler<'_, ExternalContext, DB>) {
    let prev = handler.execution.execute_frame.clone();
    handler.execution.execute_frame = Arc::new(move |frame, memory, tables, context| {
        let interpreter = frame.interpreter_mut();
        let bytecode_hash = interpreter.contract.hash.unwrap_or_default();
        if let Some(f) = context.external.get_function(bytecode_hash) {
            Ok(unsafe { f.call_with_interpreter_and_memory(interpreter, memory, context) })
        } else {
            prev(frame, memory, tables, context)
        }
    });
}

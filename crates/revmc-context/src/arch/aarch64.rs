use super::{EXIT_RESULT_OFFSET, EXIT_SP_OFFSET};
use crate::{EvmContext, EvmStack, RawEvmCompilerFn};
use revm_interpreter::InstructionResult;

/// Entry trampoline: saves callee-saved registers, stores SP into
/// `ecx.exit_sp`, and calls the JIT function.
///
/// On normal return the JIT function's `InstructionResult` is forwarded.
/// On abnormal exit (builtin error), [`revmc_exit`] restores SP and
/// returns the `exit_result` stored in `EvmContext`.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn revmc_entry(
    ecx: *mut EvmContext<'_>,
    stack: *mut EvmStack,
    stack_len: *mut usize,
    f: RawEvmCompilerFn,
) -> InstructionResult {
    // AAPCS64: x0=ecx, x1=stack, x2=stack_len, x3=f
    // The JIT function takes (ecx, stack, stack_len) — already in x0, x1, x2.
    core::arch::naked_asm!(
        // Save FP/LR and callee-saved registers (5 pairs = 80 bytes).
        "stp x29, x30, [sp, #-80]!",
        "stp x19, x20, [sp, #16]",
        "stp x21, x22, [sp, #32]",
        "stp x23, x24, [sp, #48]",
        "stp x25, x26, [sp, #64]",
        // Save SP into ecx->exit_sp.
        "mov x9, sp",
        "str x9, [x0, {exit_sp}]",
        "blr x3",
        // Normal return — w0 = InstructionResult.
        "ldp x25, x26, [sp, #64]",
        "ldp x23, x24, [sp, #48]",
        "ldp x21, x22, [sp, #32]",
        "ldp x19, x20, [sp, #16]",
        "ldp x29, x30, [sp], #80",
        "ret",
        exit_sp = const EXIT_SP_OFFSET,
    )
}

/// Exit trampoline: loads `ecx.exit_result`, restores the saved SP,
/// pops callee-saved registers and returns to the caller of `revmc_entry`.
///
/// # Safety
///
/// Must only be called from a builtin that was invoked through `revmc_entry`.
#[unsafe(naked)]
pub unsafe extern "C" fn revmc_exit(ecx: *const EvmContext<'_>) -> ! {
    core::arch::naked_asm!(
        // Restore the saved SP (must read before clobbering x0).
        "ldr x9, [x0, {exit_sp}]",
        // Load exit_result into the return register.
        "ldrb w0, [x0, {exit_result}]",
        "mov sp, x9",
        // Restore callee-saved registers and return.
        "ldp x25, x26, [sp, #64]",
        "ldp x23, x24, [sp, #48]",
        "ldp x21, x22, [sp, #32]",
        "ldp x19, x20, [sp, #16]",
        "ldp x29, x30, [sp], #80",
        "ret",
        exit_result = const EXIT_RESULT_OFFSET,
        exit_sp = const EXIT_SP_OFFSET,
    )
}

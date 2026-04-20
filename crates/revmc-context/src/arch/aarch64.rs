use super::{EXIT_RESULT_OFFSET, EXIT_SP_OFFSET};
use crate::{EvmContext, EvmStack, RawEvmCompilerFn};
use revm_interpreter::InstructionResult;

/// Entry trampoline: saves callee-saved registers, stores SP into
/// `ecx.exit_sp`, shuffles args, and calls the JIT function.
///
/// On normal return the JIT function's `InstructionResult` is forwarded.
/// On abnormal exit (builtin error), [`revmc_exit`] restores SP and
/// returns the `exit_result` stored in `EvmContext`.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn revmc_entry(
    f: RawEvmCompilerFn,
    ecx: *mut EvmContext<'_>,
    stack: *mut EvmStack,
    stack_len: *mut usize,
) -> InstructionResult {
    // AArch64 AAPCS64 calling convention:
    //   x0 = f, x1 = ecx, x2 = stack, x3 = stack_len
    //   Callee-saved: x19-x28, x29 (FP), x30 (LR)
    //   SP must be 16-byte aligned at all times.
    core::arch::naked_asm!(
        // Save FP/LR and callee-saved registers (5 pairs = 80 bytes).
        "stp x29, x30, [sp, #-80]!",
        "stp x19, x20, [sp, #16]",
        "stp x21, x22, [sp, #32]",
        "stp x23, x24, [sp, #48]",
        "stp x25, x26, [sp, #64]",
        // Save SP into ecx->exit_sp.
        "mov x9, sp",
        "str x9, [x1, {exit_sp}]",
        // Shuffle args: f(ecx, stack, stack_len).
        "mov x9, x0",  // save function pointer
        "mov x0, x1",  // arg0 = ecx
        "mov x1, x2",  // arg1 = stack
        "mov x2, x3",  // arg2 = stack_len
        "blr x9",
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
/// pops callee-saved registers and returns to the caller of [`revmc_entry`].
///
/// # Safety
///
/// Must only be called from a builtin that was invoked through [`revmc_entry`].
#[unsafe(naked)]
pub unsafe extern "C" fn revmc_exit(ecx: *const EvmContext<'_>) -> ! {
    core::arch::naked_asm!(
        // Load exit_result into the return register.
        "ldrb w0, [x0, {exit_result}]",
        // Restore the saved SP.
        "ldr x9, [x0, {exit_sp}]",
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

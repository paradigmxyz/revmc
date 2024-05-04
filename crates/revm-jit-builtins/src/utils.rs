use crate::gas;
use revm_interpreter::{as_usize_saturated, InstructionResult};
use revm_jit_context::{EvmContext, EvmWord};

/// Splits the stack pointer into `N` elements by casting it to an array.
///
/// NOTE: this returns the arguments in **reverse order**. Use [`read_words!`] to get them in order.
///
/// The returned lifetime is valid for the entire duration of the builtin.
///
/// # Safety
///
/// Caller must ensure that `N` matches the number of elements popped in JIT code.
#[inline(always)]
pub(crate) unsafe fn read_words_rev<'a, const N: usize>(sp: *mut EvmWord) -> &'a mut [EvmWord; N] {
    &mut *sp.cast::<[EvmWord; N]>()
}

#[inline]
pub(crate) fn resize_memory(
    ecx: &mut EvmContext<'_>,
    offset: usize,
    len: usize,
) -> InstructionResult {
    let size = offset.saturating_add(len);
    if size > ecx.memory.len() {
        // TODO: Memory limit
        if !revm_interpreter::interpreter::resize_memory(ecx.memory, ecx.gas, size) {
            return InstructionResult::MemoryOOG;
        }
    }
    InstructionResult::Continue
}

#[inline]
pub(crate) unsafe fn copy_operation(
    ecx: &mut EvmContext<'_>,
    rev![memory_offset, data_offset, len]: &mut [EvmWord; 3],
    data: &[u8],
) -> InstructionResult {
    let len = try_into_usize!(len);
    gas_opt!(ecx, gas::dyn_verylowcopy_cost(len as u64));
    if len == 0 {
        return InstructionResult::Continue;
    }
    let memory_offset = try_into_usize!(memory_offset);
    resize_memory!(ecx, memory_offset, len);
    let data_offset = data_offset.to_u256();
    let data_offset = as_usize_saturated!(data_offset);
    ecx.memory.set_data(memory_offset, data_offset, len, data);
    InstructionResult::Continue
}

#[inline(always)]
pub(crate) const unsafe fn decouple_lt<'b, T: ?Sized>(x: &T) -> &'b T {
    core::mem::transmute(x)
}

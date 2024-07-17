use crate::gas;
use revm_interpreter::{as_usize_saturated, Gas, InstructionResult, SharedMemory};
use revmc_context::{EvmContext, EvmWord};

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
pub(crate) fn ensure_memory(
    ecx: &mut EvmContext<'_>,
    offset: usize,
    len: usize,
) -> InstructionResult {
    ensure_memory_inner(ecx.memory, ecx.gas, offset, len)
}

#[inline]
pub(crate) fn ensure_memory_inner(
    memory: &mut SharedMemory,
    gas: &mut Gas,
    offset: usize,
    len: usize,
) -> InstructionResult {
    let new_size = offset.saturating_add(len);
    if new_size > memory.len() {
        return resize_memory_inner(memory, gas, new_size);
    }
    InstructionResult::Continue
}

#[inline]
pub(crate) fn resize_memory(ecx: &mut EvmContext<'_>, new_size: usize) -> InstructionResult {
    resize_memory_inner(ecx.memory, ecx.gas, new_size)
}

fn resize_memory_inner(
    memory: &mut SharedMemory,
    gas: &mut Gas,
    new_size: usize,
) -> InstructionResult {
    // TODO: Memory limit
    if !revm_interpreter::interpreter::resize_memory(memory, gas, new_size) {
        return InstructionResult::MemoryOOG;
    }
    InstructionResult::Continue
}

pub(crate) unsafe fn copy_operation(
    ecx: &mut EvmContext<'_>,
    rev![memory_offset, data_offset, len]: &mut [EvmWord; 3],
    data: &[u8],
) -> InstructionResult {
    let len = try_into_usize!(len);
    if len != 0 {
        gas_opt!(ecx, gas::dyn_verylowcopy_cost(len as u64));
        let memory_offset = try_into_usize!(memory_offset);
        ensure_memory!(ecx, memory_offset, len);
        let data_offset = data_offset.to_u256();
        let data_offset = as_usize_saturated!(data_offset);
        ecx.memory.set_data(memory_offset, data_offset, len, data);
    }
    InstructionResult::Continue
}

#[inline(always)]
pub(crate) const unsafe fn decouple_lt<'b, T: ?Sized>(x: &T) -> &'b T {
    core::mem::transmute(x)
}

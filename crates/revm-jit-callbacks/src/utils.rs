use revm_interpreter::{as_usize_saturated, gas as rgas, InstructionResult};
use revm_jit_context::{EvmContext, EvmWord};

/// Splits the stack pointer into `N` elements by casting it to an array.
///
/// NOTE: this returns the arguments in **reverse order**. Use [`read_words!`] to get them in order.
///
/// The returned lifetime is valid for the entire duration of the callback.
///
/// # Safety
///
/// Caller must ensure that `N` matches the number of elements popped in JIT code.
#[inline(always)]
pub unsafe fn read_words_rev<'a, const N: usize>(sp: *mut EvmWord) -> &'a mut [EvmWord; N] {
    &mut *sp.cast::<[EvmWord; N]>()
}

#[inline]
pub fn resize_memory(ecx: &mut EvmContext<'_>, offset: usize, len: usize) -> InstructionResult {
    let size = offset.saturating_add(len);
    if size > ecx.memory.len() {
        let rounded_size = revm_interpreter::interpreter::next_multiple_of_32(size);

        // TODO: Memory limit

        // TODO: try_resize
        if !ecx.gas.record_memory(rgas::memory_gas(rounded_size / 32)) {
            return InstructionResult::MemoryLimitOOG;
        }
        ecx.memory.resize(rounded_size);
    }
    InstructionResult::Continue
}

#[inline]
pub unsafe fn copy_operation(
    ecx: &mut EvmContext<'_>,
    rev![memory_offset, data_offset, len]: &mut [EvmWord; 3],
    data: &[u8],
) -> InstructionResult {
    let len = tri!(usize::try_from(len));
    gas_opt!(ecx, rgas::verylowcopy_cost(len as u64));
    if len == 0 {
        return InstructionResult::Continue;
    }
    let memory_offset = try_into_usize!(memory_offset.as_u256());
    resize_memory!(ecx, memory_offset, len);
    let data_offset = data_offset.to_u256();
    let data_offset = as_usize_saturated!(data_offset);
    ecx.memory.set_data(memory_offset, data_offset, len, data);
    InstructionResult::Continue
}

pub unsafe fn decouple_lt<'b, T: ?Sized>(x: &T) -> &'b T {
    core::mem::transmute(x)
}

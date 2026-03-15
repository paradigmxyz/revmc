use revm_interpreter::{Gas, InstructionResult, SharedMemory, as_usize_saturated};
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
    InstructionResult::Stop
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
    // Calculate words needed (memory is always word-aligned)
    let new_num_words = revm_interpreter::interpreter::num_words(new_size);
    let current_words = gas.memory().words_num;

    if new_num_words > current_words {
        // Calculate gas cost for memory expansion
        // memory_gas(num_words, linear_cost, quadratic_cost)
        // MEMORY = 3 (linear cost per word), MEMORY_QUAD_COEFFICIENT = 512 (1/512 for quadratic)
        let new_cost = crate::gas::memory_gas(new_num_words, 3, 512);
        let old_cost = crate::gas::memory_gas(current_words, 3, 512);
        let cost = new_cost.saturating_sub(old_cost);

        if !gas.record_cost(cost) {
            return InstructionResult::MemoryOOG;
        }

        // Update memory words tracking
        gas.memory_mut().words_num = new_num_words;

        // Resize the actual memory (must be word-aligned, as per EVM spec)
        memory.resize(new_num_words * 32);
    }
    InstructionResult::Stop
}

pub(crate) unsafe fn copy_operation(
    ecx: &mut EvmContext<'_>,
    memory_offset: &mut EvmWord,
    data_offset: &mut EvmWord,
    len: &mut EvmWord,
    data: &[u8],
) -> InstructionResult {
    let len = try_into_usize!(len);
    if len != 0 {
        gas!(ecx, ecx.host.gas_params().copy_cost(len));
        let memory_offset = try_into_usize!(memory_offset);
        ensure_memory!(ecx, memory_offset, len);
        let data_offset = data_offset.to_u256();
        let data_offset = as_usize_saturated!(data_offset);
        ecx.memory.set_data(memory_offset, data_offset, len, data);
    }
    InstructionResult::Stop
}

use revm_interpreter::InstructionResult;
use revm_primitives::U256;
use std::mem::MaybeUninit;

/// The signature of a JIT'd EVM bytecode.
pub type JitEvmFn = unsafe extern "C" fn(*mut ContextStack) -> InstructionResult;

/// JIT EVM context stack.
#[repr(C, align(32))]
#[allow(missing_debug_implementations)]
pub struct ContextStack([MaybeUninit<u8>; 32 * 1024]);

#[allow(clippy::new_without_default)]
impl ContextStack {
    /// The size of the stack.
    pub const SIZE: usize = 32 * 1024;

    /// Creates a new stack.
    #[inline]
    pub fn new() -> Self {
        Self(unsafe { MaybeUninit::uninit().assume_init() })
    }

    /// Returns the stack as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8; ContextStack::SIZE] {
        unsafe { std::mem::transmute(&self.0) }
    }

    /// Returns the stack as a slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8; ContextStack::SIZE] {
        unsafe { std::mem::transmute(&mut self.0) }
    }

    /// Returns the word at the given index.
    pub fn word(&self, index: usize) -> U256 {
        let offset = index * 32;
        let bytes = &self.as_slice()[offset..offset + 32];
        if cfg!(target_endian = "big") {
            U256::from_be_slice(bytes)
        } else {
            U256::from_le_slice(bytes)
        }
    }
}

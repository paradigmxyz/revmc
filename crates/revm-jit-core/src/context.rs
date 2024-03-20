use std::mem::MaybeUninit;

/// The signature of a JIT'd EVM bytecode.
pub type JitEvmFn = unsafe extern "C" fn(*mut ContextStack) -> Ret;

/// The return value of a JIT'd EVM bytecode function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum Ret {
    /// `STOP` instruction.
    Stop,
    /// Stack underflow.
    StackUnderflow,
    /// Stack overflow.
    StackOverflow,
}

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
}

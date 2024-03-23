use revm_interpreter::{Gas, InstructionResult};
use revm_primitives::U256;
use std::{fmt, mem::MaybeUninit, ptr};

/// The raw function signature of a JIT'd EVM bytecode.
///
/// Prefer using [`JitEvmFn`] instead of this type. See [`JitEvmFn::call`] for more information.
pub type RawJitEvmFn = unsafe extern "C" fn(
    gas: *mut Gas,
    stack: *mut EvmStack,
    stack_len: *mut usize,
) -> InstructionResult;

/// A JIT'd EVM bytecode function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitEvmFn(RawJitEvmFn);

impl JitEvmFn {
    /// Wraps the function.
    #[inline]
    pub const fn new(f: RawJitEvmFn) -> Self {
        Self(f)
    }

    /// Unwraps the function.
    #[inline]
    pub const fn into_inner(self) -> RawJitEvmFn {
        self.0
    }

    /// Calls the function.
    ///
    /// Arguments:
    /// - `gas`: Pointer to the gas object. Must be `Some` if `disable_gas` is set to `false` (the
    ///   default).
    /// - `stack`: Pointer to the stack. Must be `Some` if `pass_stack_through_args` is set to
    ///   `true`.
    /// - `stack_len`: Pointer to the stack length. Must be `Some` if `pass_stack_len_through_args`
    ///   is set to `true`.
    ///
    /// These conditions are enforced at runtime if `debug_assertions` is set to `true`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the arguments are valid.
    #[inline(always)]
    pub unsafe fn call(
        self,
        gas: Option<&mut Gas>,
        stack: Option<&mut EvmStack>,
        stack_len: Option<&mut usize>,
    ) -> InstructionResult {
        (self.0)(option_as_mut_ptr(gas), option_as_mut_ptr(stack), option_as_mut_ptr(stack_len))
    }
}

/// JIT EVM context stack.
#[repr(C)]
#[allow(missing_debug_implementations)]
pub struct EvmStack([MaybeUninit<EvmWord>; 1024]);

#[allow(clippy::new_without_default)]
impl EvmStack {
    /// The size of the stack in bytes.
    pub const SIZE: usize = 32 * Self::CAPACITY;

    /// The size of the stack in U256 elements.
    pub const CAPACITY: usize = 1024;

    /// Creates a new EVM stack, allocated on the stack.
    ///
    /// Use [`EvmStack::new_heap`] to create a stack on the heap.
    #[inline]
    pub fn new() -> Self {
        Self(unsafe { MaybeUninit::uninit().assume_init() })
    }

    /// Creates a vector that can be used as a stack.
    #[inline]
    pub fn new_heap() -> Vec<EvmWord> {
        Vec::with_capacity(1024)
    }

    /// Creates a stack from a mutable vector.
    ///
    /// # Panics
    ///
    /// Panics if the vector's capacity is less than the stack size.
    #[inline]
    pub fn from_vec(vec: &Vec<EvmWord>) -> &Self {
        assert!(vec.capacity() >= Self::CAPACITY);
        unsafe { &*vec.as_ptr().cast() }
    }

    /// Creates a stack from a mutable vector.
    ///
    /// The JIT'd function will overwrite the internal contents of the vector, and will not
    /// set the length. This is simply to have the stack allocated on the heap.
    ///
    /// # Panics
    ///
    /// Panics if the vector's capacity is less than the stack size.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use revm_jit_core::EvmStack;
    /// let mut stack_buf = EvmStack::new_heap();
    /// let stack = EvmStack::from_mut_vec(&mut stack_buf);
    /// assert_eq!(stack.as_slice().len(), EvmStack::CAPACITY);
    /// ```
    #[inline]
    pub fn from_mut_vec(vec: &mut Vec<EvmWord>) -> &mut Self {
        assert!(vec.capacity() >= Self::CAPACITY);
        unsafe { &mut *vec.as_mut_ptr().cast() }
    }

    /// Returns the stack as a byte array.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; EvmStack::SIZE] {
        unsafe { &*self.0.as_ptr().cast() }
    }

    /// Returns the stack as a byte array.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; EvmStack::SIZE] {
        unsafe { &mut *self.0.as_mut_ptr().cast() }
    }

    /// Returns the stack as a slice.
    #[inline]
    pub const fn as_slice(&self) -> &[EvmWord; EvmStack::CAPACITY] {
        unsafe { &*self.0.as_ptr().cast() }
    }

    /// Returns the stack as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [EvmWord; EvmStack::CAPACITY] {
        unsafe { &mut *self.0.as_mut_ptr().cast() }
    }
}

/// A native-endian 256-bit unsigned integer, aligned to 32 bytes.
///
/// This ends up being a simple no-op wrapper around `U256` on little-endian targets, modulo the
/// stricter alignment requirement, thanks to the `U256` representation being identical.
#[repr(C, align(32))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct EvmWord([u8; 32]);

macro_rules! fmt_impl {
    ($trait:ident) => {
        impl fmt::$trait for EvmWord {
            #[inline]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.to_u256().fmt(f)
            }
        }
    };
}

fmt_impl!(Debug);
fmt_impl!(Display);
fmt_impl!(LowerHex);
fmt_impl!(UpperHex);
fmt_impl!(Binary);

impl From<U256> for EvmWord {
    #[inline]
    fn from(value: U256) -> Self {
        #[cfg(target_endian = "little")]
        return unsafe { std::mem::transmute(value) };
        #[cfg(target_endian = "big")]
        return Self(value.to_be_bytes());
    }
}

impl From<&U256> for EvmWord {
    #[inline]
    fn from(value: &U256) -> Self {
        Self::from(*value)
    }
}

impl From<&mut U256> for EvmWord {
    #[inline]
    fn from(value: &mut U256) -> Self {
        Self::from(*value)
    }
}

impl EvmWord {
    /// The zero value.
    pub const ZERO: Self = Self([0; 32]);

    /// Creates a new value from native-endian bytes.
    #[inline]
    pub const fn from_ne_bytes(x: [u8; 32]) -> Self {
        Self(x)
    }

    /// Converts a [`U256`].
    #[inline]
    pub const fn from_u256(value: U256) -> Self {
        #[cfg(target_endian = "little")]
        return unsafe { std::mem::transmute(value) };
        #[cfg(target_endian = "big")]
        return Self(value.to_be_bytes());
    }

    /// Converts a [`U256`] reference to a [`U256`].
    #[inline]
    #[cfg(target_endian = "little")]
    pub const fn from_u256_ref(value: &U256) -> &Self {
        unsafe { &*(value as *const U256 as *const Self) }
    }

    /// Converts a [`U256`] mutable reference to a [`U256`].
    #[inline]
    #[cfg(target_endian = "little")]
    pub fn from_u256_mut(value: &mut U256) -> &mut Self {
        unsafe { &mut *(value as *mut U256 as *mut Self) }
    }

    /// Return the memory representation of this integer as a byte array in big-endian (network)
    /// byte order.
    #[inline]
    pub fn to_be_bytes(self) -> [u8; 32] {
        self.to_be().to_ne_bytes()
    }

    /// Return the memory representation of this integer as a byte array in little-endian byte
    /// order.
    #[inline]
    pub fn to_le_bytes(self) -> [u8; 32] {
        self.to_le().to_ne_bytes()
    }

    /// Return the memory representation of this integer as a byte array in native byte order.
    #[inline]
    pub const fn to_ne_bytes(self) -> [u8; 32] {
        self.0
    }

    /// Converts `self` to big endian from the target's endianness.
    #[inline]
    pub fn to_be(self) -> Self {
        #[cfg(target_endian = "little")]
        return self.swap_bytes();
        #[cfg(target_endian = "big")]
        return self;
    }

    /// Converts `self` to little endian from the target's endianness.
    #[inline]
    pub fn to_le(self) -> Self {
        #[cfg(target_endian = "little")]
        return self;
        #[cfg(target_endian = "big")]
        return self.swap_bytes();
    }

    /// Reverses the byte order of the integer.
    #[inline]
    pub fn swap_bytes(mut self) -> Self {
        self.0.reverse();
        self
    }

    /// Casts this value to a [`U256`]. This is a no-op on little-endian systems.
    #[cfg(target_endian = "little")]
    #[inline]
    pub const fn as_u256(&self) -> &U256 {
        unsafe { &*(self as *const Self as *const U256) }
    }

    /// Casts this value to a [`U256`]. This is a no-op on little-endian systems.
    #[cfg(target_endian = "little")]
    #[inline]
    pub fn as_u256_mut(&mut self) -> &mut U256 {
        unsafe { &mut *(self as *mut Self as *mut U256) }
    }

    /// Converts this value to a [`U256`]. This is a simple copy on little-endian systems.
    #[inline]
    pub const fn to_u256(&self) -> U256 {
        #[cfg(target_endian = "little")]
        return *self.as_u256();
        #[cfg(target_endian = "big")]
        return U256::from_be_bytes(self.0);
    }

    /// Converts this value to a [`U256`]. This is a no-op on little-endian systems.
    #[inline]
    pub const fn into_u256(self) -> U256 {
        #[cfg(target_endian = "little")]
        return unsafe { std::mem::transmute(self) };
        #[cfg(target_endian = "big")]
        return U256::from_be_bytes(self.0);
    }
}

#[inline(always)]
fn option_as_mut_ptr<T>(opt: Option<&mut T>) -> *mut T {
    match opt {
        Some(ref_) => ref_,
        None => ptr::null_mut(),
    }
}

#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::{fmt, mem::MaybeUninit, ptr};
use revm_interpreter::{
    Contract, FunctionStack, Gas, Host, InstructionResult, Interpreter, InterpreterAction,
    InterpreterResult, SharedMemory, EMPTY_SHARED_MEMORY,
};
use revm_primitives::{Address, Bytes, Env, U256};

#[cfg(feature = "host-ext-any")]
use core::any::Any;

/// The EVM bytecode compiler runtime context.
///
/// This is a simple wrapper around the interpreter's resources, allowing the compiled function to
/// access the memory, contract, gas, host, and other resources.
pub struct EvmContext<'a> {
    /// The memory.
    pub memory: &'a mut SharedMemory,
    /// Contract information and call data.
    pub contract: &'a mut Contract,
    /// The gas.
    pub gas: &'a mut Gas,
    /// The host.
    pub host: &'a mut dyn HostExt,
    /// The return action.
    pub next_action: &'a mut InterpreterAction,
    /// The return data.
    pub return_data: &'a [u8],
    /// The function stack.
    pub func_stack: &'a mut FunctionStack,
    /// Whether the context is static.
    pub is_static: bool,
    /// Whether the context is EOF init.
    pub is_eof_init: bool,
    /// An index that is used internally to keep track of where execution should resume.
    /// `0` is the initial state.
    #[doc(hidden)]
    pub resume_at: usize,
}

impl fmt::Debug for EvmContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvmContext").field("memory", &self.memory).finish_non_exhaustive()
    }
}

impl<'a> EvmContext<'a> {
    /// Creates a new context from an interpreter.
    #[inline]
    pub fn from_interpreter(interpreter: &'a mut Interpreter, host: &'a mut dyn HostExt) -> Self {
        Self::from_interpreter_with_stack(interpreter, host).0
    }

    /// Creates a new context from an interpreter.
    #[inline]
    pub fn from_interpreter_with_stack<'b: 'a>(
        interpreter: &'a mut Interpreter,
        host: &'b mut dyn HostExt,
    ) -> (Self, &'a mut EvmStack, &'a mut usize) {
        let (stack, stack_len) = EvmStack::from_interpreter_stack(&mut interpreter.stack);
        let resume_at = ResumeAt::load(
            interpreter.instruction_pointer,
            interpreter.contract.bytecode.original_byte_slice(),
        );
        let this = Self {
            memory: &mut interpreter.shared_memory,
            contract: &mut interpreter.contract,
            gas: &mut interpreter.gas,
            host,
            next_action: &mut interpreter.next_action,
            return_data: &interpreter.return_data_buffer,
            func_stack: &mut interpreter.function_stack,
            is_static: interpreter.is_static,
            is_eof_init: interpreter.is_eof_init,
            resume_at,
        };
        (this, stack, stack_len)
    }

    /// Creates a new interpreter by cloning the context.
    pub fn to_interpreter(&self, stack: revm_interpreter::Stack) -> Interpreter {
        let bytecode = self.contract.bytecode.bytecode().clone();
        Interpreter {
            is_eof: self.contract.bytecode.is_eof(),
            instruction_pointer: bytecode.as_ptr(),
            bytecode,
            function_stack: FunctionStack {
                return_stack: self.func_stack.return_stack.clone(),
                current_code_idx: self.func_stack.current_code_idx,
            },
            is_eof_init: self.is_eof_init,
            contract: self.contract.clone(),
            instruction_result: InstructionResult::Continue,
            gas: *self.gas,
            shared_memory: self.memory.clone(),
            stack,
            return_data_buffer: self.return_data.to_vec().into(),
            is_static: self.is_static,
            next_action: self.next_action.clone(),
        }
    }
}

/// Extension trait for [`Host`].
#[cfg(not(feature = "host-ext-any"))]
pub trait HostExt: Host {}

#[cfg(not(feature = "host-ext-any"))]
impl<T: Host> HostExt for T {}

/// Extension trait for [`Host`].
#[cfg(feature = "host-ext-any")]
pub trait HostExt: Host + Any {
    #[doc(hidden)]
    fn as_any(&self) -> &dyn Any;
    #[doc(hidden)]
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

#[cfg(feature = "host-ext-any")]
impl<T: Host + Any> HostExt for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(feature = "host-ext-any")]
#[doc(hidden)]
impl dyn HostExt {
    /// Attempts to downcast the host to a concrete type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.as_any().downcast_ref()
    }

    /// Attempts to downcast the host to a concrete type.
    pub fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.as_any_mut().downcast_mut()
    }
}

/// Declare [`RawEvmCompilerFn`] functions in an `extern "C"` block.
///
/// # Examples
///
/// ```no_run
/// use revmc_context::{extern_revmc, EvmCompilerFn};
///
/// extern_revmc! {
///    /// A simple function that returns `Continue`.
///    pub fn test_fn;
/// }
///
/// let test_fn = EvmCompilerFn::new(test_fn);
/// ```
#[macro_export]
macro_rules! extern_revmc {
    ($( $(#[$attr:meta])* $vis:vis fn $name:ident; )+) => {
        #[allow(improper_ctypes)]
        extern "C" {
            $(
                $(#[$attr])*
                $vis fn $name(
                    gas: *mut $crate::private::revm_interpreter::Gas,
                    stack: *mut $crate::EvmStack,
                    stack_len: *mut usize,
                    env: *const $crate::private::revm_primitives::Env,
                    contract: *const $crate::private::revm_interpreter::Contract,
                    ecx: *mut $crate::EvmContext<'_>,
                ) -> $crate::private::revm_interpreter::InstructionResult;
            )+
        }
    };
}

/// The raw function signature of a bytecode function.
///
/// Prefer using [`EvmCompilerFn`] instead of this type. See [`EvmCompilerFn::call`] for more
/// information.
// When changing the signature, also update the corresponding declarations in `fn translate`.
pub type RawEvmCompilerFn = unsafe extern "C" fn(
    gas: *mut Gas,
    stack: *mut EvmStack,
    stack_len: *mut usize,
    env: *const Env,
    contract: *const Contract,
    ecx: *mut EvmContext<'_>,
) -> InstructionResult;

/// An EVM bytecode function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvmCompilerFn(RawEvmCompilerFn);

impl From<RawEvmCompilerFn> for EvmCompilerFn {
    #[inline]
    fn from(f: RawEvmCompilerFn) -> Self {
        Self::new(f)
    }
}

impl From<EvmCompilerFn> for RawEvmCompilerFn {
    #[inline]
    fn from(f: EvmCompilerFn) -> Self {
        f.into_inner()
    }
}

impl EvmCompilerFn {
    /// Wraps the function.
    #[inline]
    pub const fn new(f: RawEvmCompilerFn) -> Self {
        Self(f)
    }

    /// Unwraps the function.
    #[inline]
    pub const fn into_inner(self) -> RawEvmCompilerFn {
        self.0
    }

    /// Calls the function by re-using the interpreter's resources and memory.
    ///
    /// See [`call_with_interpreter_and_memory`](Self::call_with_interpreter_and_memory) for more
    /// information.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the function is safe to call.
    #[inline]
    pub unsafe fn call_with_interpreter_and_memory(
        self,
        interpreter: &mut Interpreter,
        memory: &mut SharedMemory,
        host: &mut dyn HostExt,
    ) -> InterpreterAction {
        interpreter.shared_memory = core::mem::replace(memory, EMPTY_SHARED_MEMORY);
        let result = self.call_with_interpreter(interpreter, host);
        *memory = interpreter.take_memory();
        result
    }

    /// Calls the function by re-using the interpreter's resources.
    ///
    /// This behaves the same as [`Interpreter::run`], returning an [`InstructionResult`] in the
    /// interpreter's [`instruction_result`](Interpreter::instruction_result) field and the next
    /// action in the [`next_action`](Interpreter::next_action) field.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the function is safe to call.
    #[inline]
    pub unsafe fn call_with_interpreter(
        self,
        interpreter: &mut Interpreter,
        host: &mut dyn HostExt,
    ) -> InterpreterAction {
        interpreter.next_action = InterpreterAction::None;

        let (mut ecx, stack, stack_len) =
            EvmContext::from_interpreter_with_stack(interpreter, host);
        let result = self.call(Some(stack), Some(stack_len), &mut ecx);

        // Set the remaining gas to 0 if the result is `OutOfGas`,
        // as it might have overflown inside of the function.
        if result == InstructionResult::OutOfGas {
            ecx.gas.spend_all();
        }

        let resume_at = ecx.resume_at;
        // Set in EXTCALL soft failure.
        let return_data_is_empty = ecx.return_data.is_empty();

        ResumeAt::store(&mut interpreter.instruction_pointer, resume_at);
        if return_data_is_empty {
            interpreter.return_data_buffer.clear();
        }

        interpreter.instruction_result = result;
        if interpreter.next_action.is_some() {
            core::mem::take(&mut interpreter.next_action)
        } else {
            InterpreterAction::Return {
                result: InterpreterResult { result, output: Bytes::new(), gas: interpreter.gas },
            }
        }
    }

    /// Calls the function.
    ///
    /// Arguments:
    /// - `stack`: Pointer to the stack. Must be `Some` if `local_stack` is set to `false`.
    /// - `stack_len`: Pointer to the stack length. Must be `Some` if `inspect_stack_length` is set
    ///   to `true`.
    /// - `ecx`: The context object.
    ///
    /// These conditions are enforced at runtime if `debug_assertions` is set to `true`.
    ///
    /// Use of this method is discouraged, as setup and cleanup need to be done manually.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the arguments are valid and that the function is safe to call.
    #[inline]
    pub unsafe fn call(
        self,
        stack: Option<&mut EvmStack>,
        stack_len: Option<&mut usize>,
        ecx: &mut EvmContext<'_>,
    ) -> InstructionResult {
        (self.0)(
            ecx.gas,
            option_as_mut_ptr(stack),
            option_as_mut_ptr(stack_len),
            ecx.host.env(),
            ecx.contract,
            ecx,
        )
    }

    /// Same as [`call`](Self::call) but with `#[inline(never)]`.
    ///
    /// Use of this method is discouraged, as setup and cleanup need to be done manually.
    ///
    /// # Safety
    ///
    /// See [`call`](Self::call).
    #[inline(never)]
    pub unsafe fn call_noinline(
        self,
        stack: Option<&mut EvmStack>,
        stack_len: Option<&mut usize>,
        ecx: &mut EvmContext<'_>,
    ) -> InstructionResult {
        self.call(stack, stack_len, ecx)
    }
}

/// EVM context stack.
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

    /// Creates a stack from the interpreter's stack. Assumes that the stack is large enough.
    #[inline]
    pub fn from_interpreter_stack(stack: &mut revm_interpreter::Stack) -> (&mut Self, &mut usize) {
        debug_assert!(stack.data().capacity() >= Self::CAPACITY);
        unsafe {
            let data = Self::from_mut_ptr(stack.data_mut().as_mut_ptr().cast());
            // Vec { data: ptr, cap: usize, len: usize }
            let len = &mut *(stack.data_mut() as *mut Vec<_>).cast::<usize>().add(2);
            debug_assert_eq!(stack.len(), *len);
            (data, len)
        }
    }

    /// Creates a stack from a vector's buffer.
    ///
    /// # Panics
    ///
    /// Panics if the vector's capacity is less than the required stack capacity.
    #[inline]
    pub fn from_vec(vec: &Vec<EvmWord>) -> &Self {
        assert!(vec.capacity() >= Self::CAPACITY);
        unsafe { Self::from_ptr(vec.as_ptr()) }
    }

    /// Creates a stack from a mutable vector's buffer.
    ///
    /// The bytecode function will overwrite the internal contents of the vector, and will not
    /// set the length. This is simply to have the stack allocated on the heap.
    ///
    /// # Panics
    ///
    /// Panics if the vector's capacity is less than the required stack capacity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use revmc_context::EvmStack;
    /// let mut stack_buf = EvmStack::new_heap();
    /// let stack = EvmStack::from_mut_vec(&mut stack_buf);
    /// assert_eq!(stack.as_slice().len(), EvmStack::CAPACITY);
    /// ```
    #[inline]
    pub fn from_mut_vec(vec: &mut Vec<EvmWord>) -> &mut Self {
        assert!(vec.capacity() >= Self::CAPACITY);
        unsafe { Self::from_mut_ptr(vec.as_mut_ptr()) }
    }

    /// Creates a stack from a slice.
    ///
    /// # Panics
    ///
    /// Panics if the slice's length is less than the required stack capacity.
    #[inline]
    pub const fn from_slice(slice: &[EvmWord]) -> &Self {
        assert!(slice.len() >= Self::CAPACITY);
        unsafe { Self::from_ptr(slice.as_ptr()) }
    }

    /// Creates a stack from a mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if the slice's length is less than the required stack capacity.
    #[inline]
    pub fn from_mut_slice(slice: &mut [EvmWord]) -> &mut Self {
        assert!(slice.len() >= Self::CAPACITY);
        unsafe { Self::from_mut_ptr(slice.as_mut_ptr()) }
    }

    /// Creates a stack from a pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pointer is valid and points to at least [`EvmStack::SIZE`]
    /// bytes.
    #[inline]
    pub const unsafe fn from_ptr<'a>(ptr: *const EvmWord) -> &'a Self {
        &*ptr.cast()
    }

    /// Creates a stack from a mutable pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pointer is valid and points to at least [`EvmStack::SIZE`]
    /// bytes.
    #[inline]
    pub unsafe fn from_mut_ptr<'a>(ptr: *mut EvmWord) -> &'a mut Self {
        &mut *ptr.cast()
    }

    /// Returns the stack as a byte array.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; Self::SIZE] {
        unsafe { &*self.0.as_ptr().cast() }
    }

    /// Returns the stack as a byte array.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; Self::SIZE] {
        unsafe { &mut *self.0.as_mut_ptr().cast() }
    }

    /// Returns the stack as a slice.
    #[inline]
    pub const fn as_slice(&self) -> &[EvmWord; Self::CAPACITY] {
        unsafe { &*self.0.as_ptr().cast() }
    }

    /// Returns the stack as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [EvmWord; Self::CAPACITY] {
        unsafe { &mut *self.0.as_mut_ptr().cast() }
    }
}

/// A native-endian 256-bit unsigned integer, aligned to 8 bytes.
///
/// This is a transparent wrapper around [`U256`] on little-endian targets.
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct EvmWord([u8; 32]);

macro_rules! impl_fmt {
    ($($trait:ident),* $(,)?) => {
        $(
            impl fmt::$trait for EvmWord {
                #[inline]
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    self.to_u256().fmt(f)
                }
            }
        )*
    };
}

impl_fmt!(Debug, Display, Binary, Octal, LowerHex, UpperHex);

macro_rules! impl_conversions_through_u256 {
    ($($ty:ty),*) => {
        $(
            impl From<$ty> for EvmWord {
                #[inline]
                fn from(value: $ty) -> Self {
                    Self::from_u256(U256::from(value))
                }
            }

            impl From<&$ty> for EvmWord {
                #[inline]
                fn from(value: &$ty) -> Self {
                    Self::from(*value)
                }
            }

            impl From<&mut $ty> for EvmWord {
                #[inline]
                fn from(value: &mut $ty) -> Self {
                    Self::from(*value)
                }
            }

            impl TryFrom<EvmWord> for $ty {
                type Error = ();

                #[inline]
                fn try_from(value: EvmWord) -> Result<Self, Self::Error> {
                    value.to_u256().try_into().map_err(drop)
                }
            }

            impl TryFrom<&EvmWord> for $ty {
                type Error = ();

                #[inline]
                fn try_from(value: &EvmWord) -> Result<Self, Self::Error> {
                    (*value).try_into()
                }
            }

            impl TryFrom<&mut EvmWord> for $ty {
                type Error = ();

                #[inline]
                fn try_from(value: &mut EvmWord) -> Result<Self, Self::Error> {
                    (*value).try_into()
                }
            }
        )*
    };
}

impl_conversions_through_u256!(bool, u8, u16, u32, u64, usize, u128);

impl From<U256> for EvmWord {
    #[inline]
    fn from(value: U256) -> Self {
        Self::from_u256(value)
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

    /// Creates a new value from big-endian bytes.
    #[inline]
    pub fn from_be_bytes(x: [u8; 32]) -> Self {
        Self::from_be(Self(x))
    }

    /// Creates a new value from little-endian bytes.
    #[inline]
    pub fn from_le_bytes(x: [u8; 32]) -> Self {
        Self::from_le(Self(x))
    }

    /// Converts an integer from big endian to the target's endianness.
    #[inline]
    pub fn from_be(x: Self) -> Self {
        #[cfg(target_endian = "little")]
        return x.swap_bytes();
        #[cfg(target_endian = "big")]
        return x;
    }

    /// Converts an integer from little endian to the target's endianness.
    #[inline]
    pub fn from_le(x: Self) -> Self {
        #[cfg(target_endian = "little")]
        return x;
        #[cfg(target_endian = "big")]
        return x.swap_bytes();
    }

    /// Converts a [`U256`].
    #[inline]
    pub const fn from_u256(value: U256) -> Self {
        #[cfg(target_endian = "little")]
        return unsafe { core::mem::transmute::<U256, Self>(value) };
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
        return unsafe { core::mem::transmute::<Self, U256>(self) };
        #[cfg(target_endian = "big")]
        return U256::from_be_bytes(self.0);
    }

    /// Converts this value to an [`Address`].
    #[inline]
    pub fn to_address(self) -> Address {
        Address::from_word(self.to_be_bytes().into())
    }
}

/// Logic for handling the `resume_at` field.
///
/// This is stored in the [`Interpreter::instruction_pointer`] field.
struct ResumeAt;

impl ResumeAt {
    fn load(ip: *const u8, code: &[u8]) -> usize {
        if code.as_ptr_range().contains(&ip) {
            0
        } else {
            ip as usize
        }
    }

    fn store(ip: &mut *const u8, value: usize) {
        *ip = value as *const u8;
    }
}

#[inline(always)]
fn option_as_mut_ptr<T>(opt: Option<&mut T>) -> *mut T {
    match opt {
        Some(ref_) => ref_,
        None => ptr::null_mut(),
    }
}

// Macro re-exports.
// Not public API.
#[doc(hidden)]
pub mod private {
    pub use revm_interpreter;
    pub use revm_primitives;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions() {
        let mut word = EvmWord::ZERO;
        assert_eq!(usize::try_from(word), Ok(0));
        assert_eq!(usize::try_from(&word), Ok(0));
        assert_eq!(usize::try_from(&mut word), Ok(0));
    }

    extern_revmc! {
        #[link_name = "__test_fn"]
        fn test_fn;
    }

    #[no_mangle]
    extern "C" fn __test_fn(
        _gas: *mut Gas,
        _stack: *mut EvmStack,
        _stack_len: *mut usize,
        _env: *const Env,
        _contract: *const Contract,
        _ecx: *mut EvmContext<'_>,
    ) -> InstructionResult {
        InstructionResult::Continue
    }

    #[test]
    fn extern_macro() {
        let _f1 = EvmCompilerFn::new(test_fn);
        let _f2 = EvmCompilerFn::new(__test_fn);
        assert_eq!(test_fn as usize, __test_fn as usize);
    }

    #[test]
    fn borrowing_host() {
        #[allow(unused)]
        struct BHost<'a>(&'a mut Env);
        #[allow(unused)]
        impl Host for BHost<'_> {
            fn env(&self) -> &Env {
                self.0
            }
            fn env_mut(&mut self) -> &mut Env {
                self.0
            }
            fn load_account(
                &mut self,
                address: Address,
            ) -> Option<revm_interpreter::LoadAccountResult> {
                unimplemented!()
            }
            fn block_hash(&mut self, number: u64) -> Option<revm_primitives::B256> {
                unimplemented!()
            }
            fn balance(&mut self, address: Address) -> Option<(U256, bool)> {
                unimplemented!()
            }
            fn code(&mut self, address: Address) -> Option<(revm_primitives::Bytes, bool)> {
                unimplemented!()
            }
            fn code_hash(&mut self, address: Address) -> Option<(revm_primitives::B256, bool)> {
                unimplemented!()
            }
            fn sload(&mut self, address: Address, index: U256) -> Option<(U256, bool)> {
                unimplemented!()
            }
            fn sstore(
                &mut self,
                address: Address,
                index: U256,
                value: U256,
            ) -> Option<revm_interpreter::SStoreResult> {
                unimplemented!()
            }
            fn tload(&mut self, address: Address, index: U256) -> U256 {
                unimplemented!()
            }
            fn tstore(&mut self, address: Address, index: U256, value: U256) {
                unimplemented!()
            }
            fn log(&mut self, log: revm_primitives::Log) {
                unimplemented!()
            }
            fn selfdestruct(
                &mut self,
                address: Address,
                target: Address,
            ) -> Option<revm_interpreter::SelfDestructResult> {
                unimplemented!()
            }
        }

        #[allow(unused_mut)]
        let mut env = Env::default();
        #[cfg(not(feature = "host-ext-any"))]
        let env = &mut env;
        #[cfg(feature = "host-ext-any")]
        let env = Box::leak(Box::new(env));

        let mut host = BHost(env);
        let f = EvmCompilerFn::new(test_fn);
        let mut interpreter = Interpreter::new(Contract::default(), u64::MAX, false);

        let (mut ecx, stack, stack_len) =
            EvmContext::from_interpreter_with_stack(&mut interpreter, &mut host);
        let r = unsafe { f.call(Some(stack), Some(stack_len), &mut ecx) };
        assert_eq!(r, InstructionResult::Continue);

        let r = unsafe { f.call_with_interpreter(&mut interpreter, &mut host) };
        assert_eq!(
            r,
            InterpreterAction::Return {
                result: InterpreterResult {
                    result: InstructionResult::Continue,
                    output: Bytes::new(),
                    gas: Gas::new(u64::MAX),
                }
            }
        );
    }
}

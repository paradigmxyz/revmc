#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::{fmt, mem::MaybeUninit, ptr};
use revm_interpreter::{
    interpreter_types::{Jumps, ReturnData},
    Gas, Host, InputsImpl, InstructionResult, Interpreter, InterpreterAction, InterpreterResult,
    SharedMemory,
};
use revm_primitives::{ruint, Address, Bytes, U256};

#[cfg(feature = "host-ext-any")]
use core::any::Any;

/// The EVM bytecode compiler runtime context.
///
/// This is a simple wrapper around the interpreter's resources, allowing the compiled function to
/// access the memory, input, gas, host, and other resources.
///
/// # Safety
/// This struct uses `#[repr(C)]` to ensure a stable field layout since the JIT compiler
/// generates code that accesses fields by offset using `offset_of!`.
#[repr(C)]
pub struct EvmContext<'a> {
    /// The memory.
    pub memory: &'a mut SharedMemory,
    /// Input information (target address, caller, input data, call value).
    pub input: &'a mut InputsImpl,
    /// The gas.
    pub gas: &'a mut Gas,
    /// The host.
    pub host: &'a mut dyn HostExt,
    /// The return action.
    pub next_action: &'a mut Option<InterpreterAction>,
    /// The return data.
    pub return_data: &'a [u8],
    /// Whether the context is static.
    pub is_static: bool,
    /// An index that is used internally to keep track of where execution should resume.
    /// `0` is the initial state.
    #[doc(hidden)]
    pub resume_at: usize,
    /// Pointer to the contract bytecode (for CODECOPY in AOT-cached code).
    pub bytecode_ptr: *const u8,
    /// Length of the contract bytecode.
    pub bytecode_len: usize,
}

// Static assertions to ensure the struct layout matches expectations.
// These offsets are used by the JIT compiler to access fields.
const _: () = {
    use core::mem::offset_of;
    assert!(core::mem::size_of::<EvmContext<'_>>() == 96);
    // Key fields accessed by JIT code
    assert!(offset_of!(EvmContext<'_>, memory) == 0);
    assert!(offset_of!(EvmContext<'_>, resume_at) == 72);
    assert!(offset_of!(EvmContext<'_>, bytecode_ptr) == 80);
    assert!(offset_of!(EvmContext<'_>, bytecode_len) == 88);
};

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
        use revm_interpreter::interpreter_types::LegacyBytecode;

        let (stack, stack_len) = EvmStack::from_interpreter_stack(&mut interpreter.stack);
        let bytecode_slice = interpreter.bytecode.bytecode_slice();
        let resume_at = ResumeAt::load(interpreter.bytecode.pc(), bytecode_slice);
        let bytecode_ptr = bytecode_slice.as_ptr();
        let bytecode_len = bytecode_slice.len();
        let this = Self {
            memory: &mut interpreter.memory,
            input: &mut interpreter.input,
            gas: &mut interpreter.gas,
            host,
            next_action: &mut interpreter.bytecode.action,
            return_data: interpreter.return_data.buffer(),
            is_static: interpreter.runtime_flag.is_static,
            resume_at,
            bytecode_ptr,
            bytecode_len,
        };
        (this, stack, stack_len)
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
                    input: *const $crate::private::revm_interpreter::InputsImpl,
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
    input: *const InputsImpl,
    ecx: *mut EvmContext<'_>,
) -> InstructionResult;

/// An EVM bytecode function.
#[derive(Clone, Copy, Debug, Hash)]
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
        interpreter.memory = core::mem::replace(memory, SharedMemory::invalid());
        let result = self.call_with_interpreter(interpreter, host);
        *memory = core::mem::replace(&mut interpreter.memory, SharedMemory::invalid());
        result
    }

    /// Calls the function by re-using the interpreter's resources.
    ///
    /// This behaves similarly to `Interpreter::run_plain`, returning an [`InstructionResult`]
    /// and the next action in an [`InterpreterAction`].
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
        interpreter.bytecode.action = None;

        let (mut ecx, stack, stack_len) =
            EvmContext::from_interpreter_with_stack(interpreter, host);
        let result = self.call(Some(stack), Some(stack_len), &mut ecx);

        // Set the remaining gas to 0 if the result is `OutOfGas`,
        // as it might have overflown inside of the function.
        if result == InstructionResult::OutOfGas {
            ecx.gas.spend_all();
        }

        let return_data_is_empty = ecx.return_data.is_empty();

        if return_data_is_empty {
            interpreter.return_data.0.clear();
        }

        if let Some(action) = interpreter.bytecode.action.take() {
            action
        } else {
            InterpreterAction::Return(InterpreterResult {
                result,
                output: Bytes::new(),
                gas: interpreter.gas,
            })
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
        (self.0)(ecx.gas, option_as_mut_ptr(stack), option_as_mut_ptr(stack_len), ecx.input, ecx)
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
    /// # Panics
    ///
    /// Panics if the vector's capacity is less than the required stack capacity.
    #[inline]
    pub fn from_mut_vec(vec: &mut Vec<EvmWord>) -> &mut Self {
        assert!(vec.capacity() >= Self::CAPACITY);
        unsafe { Self::from_mut_ptr(vec.as_mut_ptr()) }
    }

    /// Creates a stack from a pointer to a buffer.
    ///
    /// # Safety
    ///
    /// See [`from_vec`](Self::from_vec).
    #[inline]
    pub const unsafe fn from_ptr(ptr: *const EvmWord) -> &'static Self {
        &*ptr.cast::<Self>()
    }

    /// Creates a stack from a mutable pointer to a buffer.
    ///
    /// # Safety
    ///
    /// See [`from_mut_vec`](Self::from_mut_vec).
    #[inline]
    pub unsafe fn from_mut_ptr(ptr: *mut EvmWord) -> &'static mut Self {
        &mut *ptr.cast::<Self>()
    }

    /// Returns a pointer to the stack.
    #[inline]
    pub const fn as_ptr(&self) -> *const EvmWord {
        self.0.as_ptr().cast()
    }

    /// Returns a mutable pointer to the stack.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut EvmWord {
        self.0.as_mut_ptr().cast()
    }

    /// Returns a slice of the stack.
    #[inline]
    pub fn as_slice(&self) -> &[EvmWord] {
        // SAFETY: EvmWord is repr(C) and same layout as [u8; 32]
        unsafe { core::slice::from_raw_parts(self.as_ptr(), Self::CAPACITY) }
    }

    /// Returns a mutable slice of the stack.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [EvmWord] {
        // SAFETY: EvmWord is repr(C) and same layout as [u8; 32]
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), Self::CAPACITY) }
    }

    /// Returns the word at the given index as a reference.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&EvmWord> {
        self.0.get(index).map(|slot| unsafe { slot.assume_init_ref() })
    }

    /// Returns the word at the given index as a mutable reference.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut EvmWord> {
        self.0.get_mut(index).map(|slot| unsafe { slot.assume_init_mut() })
    }

    /// Returns the word at the given index as a reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &EvmWord {
        self.0.get_unchecked(index).assume_init_ref()
    }

    /// Returns the word at the given index as a mutable reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut EvmWord {
        self.0.get_unchecked_mut(index).assume_init_mut()
    }

    /// Sets the value at the top of the stack to `value`, and grows the stack by 1.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the stack is not full.
    #[inline]
    pub unsafe fn push(&mut self, value: EvmWord, len: &mut usize) {
        self.set_unchecked(*len, value);
        *len += 1;
    }

    /// Returns the value at the top of the stack.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the stack is not empty.
    #[inline]
    pub unsafe fn top_unchecked(&self, len: usize) -> &EvmWord {
        self.get_unchecked(len - 1)
    }

    /// Returns the value at the top of the stack as a mutable reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the stack is not empty.
    #[inline]
    pub unsafe fn top_unchecked_mut(&mut self, len: usize) -> &mut EvmWord {
        self.get_unchecked_mut(len - 1)
    }

    /// Returns the value at the given index from the top of the stack.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `len >= n + 1`.
    #[inline]
    pub unsafe fn from_top_unchecked(&self, len: usize, n: usize) -> &EvmWord {
        self.get_unchecked(len - n - 1)
    }

    /// Returns the value at the given index from the top of the stack as a mutable reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `len >= n + 1`.
    #[inline]
    pub unsafe fn from_top_unchecked_mut(&mut self, len: usize, n: usize) -> &mut EvmWord {
        self.get_unchecked_mut(len - n - 1)
    }

    /// Sets the value at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: EvmWord) {
        *self.0.get_unchecked_mut(index) = MaybeUninit::new(value);
    }
}

/// An EVM stack word, which is stored in native-endian order.
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(missing_debug_implementations)]
pub struct EvmWord([u8; 32]);

impl Default for EvmWord {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Debug for EvmWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x")?;
        for byte in &self.to_be_bytes() {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl TryFrom<EvmWord> for usize {
    type Error = ruint::FromUintError<Self>;

    #[inline]
    fn try_from(w: EvmWord) -> Result<Self, Self::Error> {
        Self::try_from(&w)
    }
}

impl TryFrom<&EvmWord> for usize {
    type Error = ruint::FromUintError<Self>;

    #[inline]
    fn try_from(w: &EvmWord) -> Result<Self, Self::Error> {
        w.to_u256().try_into()
    }
}

impl TryFrom<&mut EvmWord> for usize {
    type Error = ruint::FromUintError<Self>;

    #[inline]
    fn try_from(w: &mut EvmWord) -> Result<Self, Self::Error> {
        Self::try_from(&*w)
    }
}

impl From<U256> for EvmWord {
    #[inline]
    fn from(u: U256) -> Self {
        Self::from_u256(u)
    }
}

impl EvmWord {
    /// Zero.
    pub const ZERO: Self = Self([0; 32]);

    /// Create a new word from big-endian bytes.
    #[inline]
    pub fn from_be_bytes(bytes: [u8; 32]) -> Self {
        Self::from_be(Self(bytes))
    }

    /// Create a new word from little-endian bytes.
    #[inline]
    pub fn from_le_bytes(bytes: [u8; 32]) -> Self {
        Self::from_le(Self(bytes))
    }

    /// Create a new word from native-endian bytes.
    #[inline]
    pub const fn from_ne_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Create a new word from a [`U256`]. This is a no-op on little-endian systems.
    #[inline]
    pub const fn from_u256(u: U256) -> Self {
        #[cfg(target_endian = "little")]
        return unsafe { core::mem::transmute::<U256, Self>(u) };
        #[cfg(target_endian = "big")]
        return Self(u.to_be_bytes());
    }

    /// Converts a big-endian representation into a native one.
    #[inline]
    pub fn from_be(x: Self) -> Self {
        #[cfg(target_endian = "little")]
        return x.swap_bytes();
        #[cfg(target_endian = "big")]
        return x;
    }

    /// Converts a little-endian representation into a native one.
    #[inline]
    pub fn from_le(x: Self) -> Self {
        #[cfg(target_endian = "little")]
        return x;
        #[cfg(target_endian = "big")]
        return x.swap_bytes();
    }

    /// Return the memory representation of this integer as a byte array in big-endian byte order.
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
/// This is stored in the bytecode's PC.
struct ResumeAt;

impl ResumeAt {
    fn load(pc: usize, code: &[u8]) -> usize {
        if pc < code.len() {
            0
        } else {
            pc
        }
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
        _input: *const InputsImpl,
        _ecx: *mut EvmContext<'_>,
    ) -> InstructionResult {
        InstructionResult::Stop
    }

    #[test]
    fn extern_macro() {
        let _f1 = EvmCompilerFn::new(test_fn);
        let _f2 = EvmCompilerFn::new(__test_fn);
        assert_eq!(test_fn as *const () as usize, __test_fn as *const () as usize);
    }
}

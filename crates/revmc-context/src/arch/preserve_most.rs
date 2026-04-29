/// Defines a function whose generated-code callers use LLVM's `preserve_mostcc`.
///
/// The wrapper preserves the extra caller-saved registers that `preserve_mostcc`
/// promises to keep live across the call, then calls the provided inner Rust
/// function body using the platform C ABI.
#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! preserve_most {
    (
        $(#[$attr:meta])*
        $vis:vis unsafe extern "C" fn $name:ident as $inner:ident(
            $($arg:ident : $ty:ty),* $(,)?
        ) $body:block
    ) => {
        #[inline(never)]
        unsafe extern "C" fn $inner($($arg: $ty),*) $body

        $(#[$attr])*
        #[unsafe(naked)]
        $vis unsafe extern "C" fn $name($($arg: $ty),*) {
            core::arch::naked_asm!(
                "push rax",
                "push rcx",
                "push rdx",
                "push rsi",
                "push rdi",
                "push r8",
                "push r9",
                "push r10",
                "sub rsp, 8",
                "call {inner}",
                "add rsp, 8",
                "pop r10",
                "pop r9",
                "pop r8",
                "pop rdi",
                "pop rsi",
                "pop rdx",
                "pop rcx",
                "pop rax",
                "ret",
                inner = sym $inner,
            );
        }
    };
}

/// Defines a function whose generated-code callers use LLVM's `preserve_mostcc`.
///
/// The wrapper preserves the extra caller-saved registers that `preserve_mostcc`
/// promises to keep live across the call, then calls the provided inner Rust
/// function body using the platform C ABI.
#[cfg(target_arch = "aarch64")]
#[macro_export]
macro_rules! preserve_most {
    (
        $(#[$attr:meta])*
        $vis:vis unsafe extern "C" fn $name:ident as $inner:ident(
            $($arg:ident : $ty:ty),* $(,)?
        ) $body:block
    ) => {
        #[inline(never)]
        unsafe extern "C" fn $inner($($arg: $ty),*) $body

        $(#[$attr])*
        #[unsafe(naked)]
        $vis unsafe extern "C" fn $name($($arg: $ty),*) {
            core::arch::naked_asm!(
                "stp x9,  x10, [sp, #-64]!",
                "stp x11, x12, [sp, #16]",
                "stp x13, x14, [sp, #32]",
                "stp x15, x30, [sp, #48]",
                "bl {inner}",
                "ldp x15, x30, [sp, #48]",
                "ldp x13, x14, [sp, #32]",
                "ldp x11, x12, [sp, #16]",
                "ldp x9,  x10, [sp], #64",
                "ret",
                inner = sym $inner,
            );
        }
    };
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[macro_export]
macro_rules! preserve_most {
    (
        $(#[$attr:meta])*
        $vis:vis unsafe extern "C" fn $name:ident as $inner:ident(
            $($arg:ident : $ty:ty),* $(,)?
        ) $body:block
    ) => {
        $(#[$attr])*
        $vis unsafe extern "C" fn $name($($arg: $ty),*) $body
    };
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicUsize, Ordering};

    static VALUE: AtomicUsize = AtomicUsize::new(0);

    crate::preserve_most! {
        unsafe extern "C" fn preserve_most_test_wrapper as preserve_most_test_inner(
            lhs: usize,
            rhs: usize,
        ) {
            VALUE.store(lhs + rhs, Ordering::Relaxed);
        }
    }

    #[test]
    fn wrapper_calls_inner() {
        VALUE.store(0, Ordering::Relaxed);
        unsafe { preserve_most_test_wrapper(20, 22) };
        assert_eq!(VALUE.load(Ordering::Relaxed), 42);
    }
}

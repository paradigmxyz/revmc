use crate::{EvmContext, EvmWord};
use revm_interpreter::gas;
use revm_jit_backend::{Attribute, TypeMethods};
use revm_primitives::{FrontierSpec, SpecId, SpuriousDragonSpec, KECCAK_EMPTY, U256};

macro_rules! callbacks {
    (@count) => { 0 };
    (@count $first:tt $(, $rest:tt)*) => { 1 + callbacks!(@count $($rest),*) };

    ($bcx:ident; $ptr:ident; $usize:ident;
     $($ident:ident = $(#[$attr:expr])* $name:ident($($params:expr),* $(,)?) $ret:expr),* $(,)?
    ) => {
        /// Callbacks that can be called by the JIT-compiled functions.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub(crate) enum Callback {
            $($ident,)*
        }

        #[allow(unused_variables)]
        impl Callback {
            pub(crate) const COUNT: usize = callbacks!(@count $($ident),*);

            pub(crate) const fn name(self) -> &'static str {
                match self {
                    $(Self::$ident => stringify!($name),)*
                }
            }

            pub(crate) fn addr(self) -> usize {
                match self {
                    $(Self::$ident => self::$name as usize,)*
                }
            }

            pub(crate) fn ret<B: TypeMethods>(self, $bcx: &mut B) -> Option<B::Type> {
                let $ptr = $bcx.type_ptr();
                let $usize = $bcx.type_ptr_sized_int();
                match self {
                    $(Self::$ident => $ret,)*
                }
            }

            pub(crate) fn params<B: TypeMethods>(self, $bcx: &mut B) -> Vec<B::Type> {
                let $ptr = $bcx.type_ptr();
                let $usize = $bcx.type_ptr_sized_int();
                match self {
                    $(Self::$ident => vec![$($params),*],)*
                }
            }

            pub(crate) fn attrs(self) -> &'static [Attribute] {
                #[allow(unused_imports)]
                use Attribute::*;
                match self {
                    $(Self::$ident => &[$($attr)*]),*
                }
            }
        }
    };
}

callbacks! { bcx; ptr; usize;
    Panic     = panic(ptr, usize) None,
    AddMod    = addmod(ptr) None,
    MulMod    = mulmod(ptr) None,
    Exp       = exp(ptr, bcx.type_int(8)) Some(bcx.type_int(64)),
    Keccak256 = keccak256(ptr, ptr) Some(bcx.type_int(64)),
}

/* ------------------------------------- Callback Functions ------------------------------------- */
// NOTE: All functions MUST be `extern "C"` and their parameters must match the ones declared above.
//
// The `sp` parameter points to the top of the stack.
// `sp` functions are called with the length of the stack already checked and substracted.
// All they have to do is read from `sp` and write the result to the **last** returned pointer.
// This represents "pushing" the result onto the stack.

pub(crate) unsafe extern "C" fn panic(ptr: *const u8, len: usize) -> ! {
    let msg = std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len));
    panic!("{msg}");
}

pub(crate) unsafe extern "C" fn addmod(sp: *mut EvmWord) {
    read_words!(sp, a, b, c);
    *c = a.to_u256().add_mod(b.to_u256(), c.to_u256()).into();
}

pub(crate) unsafe extern "C" fn mulmod(sp: *mut EvmWord) {
    read_words!(sp, a, b, c);
    *c = a.to_u256().mul_mod(b.to_u256(), c.to_u256()).into();
}

pub(crate) unsafe extern "C" fn exp(sp: *mut EvmWord, spec_id: SpecId) -> u64 {
    read_words!(sp, base, exponent_ptr);
    let exponent = exponent_ptr.to_u256();
    let gas = if SpecId::enabled(spec_id, SpecId::SPURIOUS_DRAGON) {
        gas::exp_cost::<SpuriousDragonSpec>(exponent)
    } else {
        gas::exp_cost::<FrontierSpec>(exponent)
    };
    if let Some(gas) = gas {
        *exponent_ptr = base.to_u256().pow(exponent).into();
        gas
    } else {
        u64::MAX
    }
}

pub(crate) unsafe extern "C" fn keccak256(ecx: &mut EvmContext<'_>, sp: *mut EvmWord) -> u64 {
    read_words!(sp, offset, len_ptr);
    let len = len_ptr.to_u256();
    if len == U256::ZERO {
        *len_ptr = EvmWord::from_be_bytes(KECCAK_EMPTY.0);
        return gas::KECCAK256;
    }
    let Ok(len) = usize::try_from(len) else { return u64::MAX };
    let Some(mut gas) = gas::keccak256_cost(len as u64) else { return u64::MAX };
    let Ok(offset) = offset.try_into() else { return u64::MAX };

    // Copied from `resize_memory!`.
    let size = len.saturating_add(offset);
    if size > ecx.memory.len() {
        let rounded_size = revm_interpreter::interpreter::next_multiple_of_32(len);

        // TODO: Memory limit

        let words_num = rounded_size / 32;
        let Some(gas2) = gas.checked_add(gas::memory_gas(words_num)) else { return u64::MAX };
        gas = gas2;
        ecx.memory.resize(rounded_size);
    }

    let data = ecx.memory.slice(offset, len);
    *len_ptr = EvmWord::from_be_bytes(revm_primitives::keccak256(data).0);

    gas
}

// --- utils ---

/// Same as `read_words_rev`, but returns the arguments in the order they were passed.
macro_rules! read_words {
    ($sp:expr, $($words:ident),+ $(,)?) => {
        let reverse_tokens!($($words),+) = unsafe { read_words_rev($sp) };
    };
}
use read_words;

// Credits: <https://github.com/AuroransSolis/rustconf-2023/blob/665a645d751dfe0e483261e3abca25ab4bb9e13a/reverse-tokens/src/main.rs>
macro_rules! reverse_tokens {
	(@rev [$first:tt$(, $rest:tt)*] [$($rev:tt),*]) => {
		reverse_tokens! {
			@rev [$($rest),*][$first $(, $rev)*]
		}
	};
	(@rev [] [$($rev:tt),*]) => {
		[$($rev)*]
	};
	($($tt:tt)+) => {
		reverse_tokens! {
			@rev [$($tt),+] []
		}
	};
}
use reverse_tokens;

/// Splits the stack pointer into `N` elements by casting it to an array.
/// This has the same effect as popping `N` elements from the stack since the JIT function
/// has already modified the length.
///
/// NOTE: this returns the arguments in **reverse order**. Use [`read_words!`] to get them in order.
///
/// The returned lifetime is valid for the entire duration of the callback.
///
/// # Safety
///
/// Caller must ensure that `N` matches the number of elements popped in JIT code.
#[inline(always)]
unsafe fn read_words_rev<'a, const N: usize>(sp: *mut EvmWord) -> &'a mut [EvmWord; N] {
    &mut *sp.sub(N).cast::<[EvmWord; N]>()
}

use alloc::vec::Vec;
use revm_jit_backend::{Attribute, Backend, Builder, FunctionAttributeLocation, TypeMethods};

/// Callback cache.
#[derive(Debug)]
pub struct Callbacks<B: Backend>([Option<B::Function>; Callback::COUNT]);

impl<B: Backend> Default for Callbacks<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Callbacks<B> {
    /// Create a new cache.
    pub fn new() -> Self {
        Self([None; Callback::COUNT])
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Get the function for the given callback.
    pub fn get(&mut self, cb: Callback, bcx: &mut B::Builder<'_>) -> B::Function {
        *self.0[cb as usize].get_or_insert_with(|| Self::init(cb, bcx))
    }

    #[cold]
    fn init(cb: Callback, bcx: &mut B::Builder<'_>) -> B::Function {
        let mut name = cb.name();
        let mangle_prefix = "__revm_jit_callback_";
        let storage;
        if !name.starts_with(mangle_prefix) {
            storage = [mangle_prefix, name].concat();
            name = storage.as_str();
        }
        bcx.get_function(name).inspect(|r| trace!(name, ?r, "pre-existing")).unwrap_or_else(|| {
            let r = Self::build(name, cb, bcx);
            trace!(name, ?r, "built");
            r
        })
    }

    fn build(name: &str, cb: Callback, bcx: &mut B::Builder<'_>) -> B::Function {
        let ret = cb.ret(bcx);
        let params = cb.params(bcx);
        let address = cb.addr();
        let linkage = revm_jit_backend::Linkage::Import;
        let f = bcx.add_callback_function(name, ret, &params, address, linkage);
        let default_attrs: &[Attribute] = if cb == Callback::Panic {
            &[
                Attribute::Cold,
                Attribute::NoReturn,
                Attribute::NoFree,
                Attribute::NoRecurse,
                Attribute::NoSync,
            ]
        } else {
            &[
                Attribute::WillReturn,
                Attribute::NoFree,
                Attribute::NoRecurse,
                Attribute::NoSync,
                Attribute::NoUnwind,
                Attribute::Speculatable,
            ]
        };
        for attr in default_attrs.iter().chain(cb.attrs()).copied() {
            bcx.add_function_attribute(Some(f), attr, FunctionAttributeLocation::Function);
        }
        f
    }
}

macro_rules! callbacks {
    (@count) => { 0 };
    (@count $first:tt $(, $rest:tt)*) => { 1 + callbacks!(@count $($rest),*) };

    (|$bcx:ident| { $($init:tt)* }
     $($ident:ident = $(#[$attr:expr])* $name:ident($($params:expr),* $(,)?) $ret:expr),* $(,)?
    ) => {
        /// Callbacks that can be called by the JIT-compiled functions.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Callback {
            $($ident,)*
        }

        #[allow(unused_variables)]
        impl Callback {
            pub const COUNT: usize = callbacks!(@count $($ident),*);

            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$ident => stringify!($name),)*
                }
            }

            pub fn addr(self) -> usize {
                match self {
                    $(Self::$ident => crate::$name as usize,)*
                }
            }

            pub fn ret<B: TypeMethods>(self, $bcx: &mut B) -> Option<B::Type> {
                $($init)*
                match self {
                    $(Self::$ident => $ret,)*
                }
            }

            pub fn params<B: TypeMethods>(self, $bcx: &mut B) -> Vec<B::Type> {
                $($init)*
                match self {
                    $(Self::$ident => vec![$($params),*],)*
                }
            }

            pub fn attrs(self) -> &'static [Attribute] {
                #[allow(unused_imports)]
                use Attribute::*;
                match self {
                    $(Self::$ident => &[$($attr)*]),*
                }
            }
        }
    };
}

// TODO: Parameter attributes, especially `dereferenceable(<size>)` and `sret(<ty>)`.
callbacks! {
    |bcx| {
        let ptr = bcx.type_ptr();
        let usize = bcx.type_ptr_sized_int();
        let bool = bcx.type_int(1);
        let u8 = bcx.type_int(8);
    }

    Panic          = __revm_jit_panic(ptr, usize) None,

    AddMod         = __revm_jit_addmod(ptr) None,
    MulMod         = __revm_jit_mulmod(ptr) None,
    Exp            = __revm_jit_exp(ptr, ptr, u8) Some(u8),
    Keccak256      = __revm_jit_keccak256(ptr, ptr) Some(u8),
    Balance        = __revm_jit_balance(ptr, ptr, u8) Some(u8),
    CallDataCopy   = __revm_jit_calldatacopy(ptr, ptr) Some(u8),
    CodeCopy       = __revm_jit_codecopy(ptr, ptr) Some(u8),
    ExtCodeSize    = __revm_jit_extcodesize(ptr, ptr, u8) Some(u8),
    ExtCodeCopy    = __revm_jit_extcodecopy(ptr, ptr, u8) Some(u8),
    ReturnDataCopy = __revm_jit_returndatacopy(ptr, ptr) Some(u8),
    ExtCodeHash    = __revm_jit_extcodehash(ptr, ptr, u8) Some(u8),
    BlockHash      = __revm_jit_blockhash(ptr, ptr) Some(u8),
    SelfBalance    = __revm_jit_self_balance(ptr, ptr) Some(u8),
    BlobHash       = __revm_jit_blob_hash(ptr, ptr) None,
    BlobBaseFee    = __revm_jit_blob_base_fee(ptr, ptr) None,
    Mload          = __revm_jit_mload(ptr, ptr) Some(u8),
    Mstore         = __revm_jit_mstore(ptr, ptr) Some(u8),
    Mstore8        = __revm_jit_mstore8(ptr, ptr) Some(u8),
    Sload          = __revm_jit_sload(ptr, ptr, u8) Some(u8),
    Sstore         = __revm_jit_sstore(ptr, ptr, u8) Some(u8),
    Msize          = __revm_jit_msize(ptr) Some(usize),
    Tstore         = __revm_jit_tstore(ptr, ptr) None,
    Tload          = __revm_jit_tload(ptr, ptr) None,
    Log            = __revm_jit_log(ptr, ptr, u8) Some(u8),

    Create         = __revm_jit_create(ptr, ptr, u8, bool) Some(u8),
    DoReturn       = __revm_jit_do_return(ptr, ptr, u8) Some(u8),
    SelfDestruct   = __revm_jit_selfdestruct(ptr, ptr, u8) Some(u8),
}

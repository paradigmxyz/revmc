use revmc_backend::{Attribute, Backend, Builder, FunctionAttributeLocation, TypeMethods};

// Must be kept in sync with `remvc-build`.
const MANGLE_PREFIX: &str = "__revmc_builtin_";

/// Builtin cache.
#[derive(Debug)]
pub struct Builtins<B: Backend>([Option<B::Function>; Builtin::COUNT]);

unsafe impl<B: Backend> Send for Builtins<B> {}

impl<B: Backend> Default for Builtins<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Builtins<B> {
    /// Create a new cache.
    pub fn new() -> Self {
        Self([None; Builtin::COUNT])
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Get the function for the given builtin.
    pub fn get(&mut self, builtin: Builtin, bcx: &mut B::Builder<'_>) -> B::Function {
        *self.0[builtin as usize].get_or_insert_with(|| Self::init(builtin, bcx))
    }

    #[cold]
    fn init(builtin: Builtin, bcx: &mut B::Builder<'_>) -> B::Function {
        let name = builtin.name();
        debug_assert!(name.starts_with(MANGLE_PREFIX), "{name:?}");
        bcx.get_function(name).inspect(|r| trace!(name, ?r, "pre-existing")).unwrap_or_else(|| {
            let r = Self::build(name, builtin, bcx);
            trace!(name, ?r, "built");
            r
        })
    }

    fn build(name: &str, builtin: Builtin, bcx: &mut B::Builder<'_>) -> B::Function {
        let ret = builtin.ret(bcx);
        let params = builtin.params(bcx);
        let address = builtin.addr();
        let linkage = revmc_backend::Linkage::Import;
        let f = bcx.add_function(name, &params, ret, Some(address), linkage);
        let default_attrs: &[Attribute] = if builtin == Builtin::Panic {
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
            ]
        };
        for attr in default_attrs.iter().chain(builtin.attrs()).copied() {
            bcx.add_function_attribute(Some(f), attr, FunctionAttributeLocation::Function);
        }
        let param_attrs = builtin.param_attrs();
        for (i, param_attrs) in param_attrs.iter().enumerate() {
            for attr in param_attrs {
                bcx.add_function_attribute(
                    Some(f),
                    *attr,
                    FunctionAttributeLocation::Param(i as u32),
                );
            }
        }
        f
    }
}

macro_rules! builtins {
    (@count) => { 0 };
    (@count $first:tt $(, $rest:tt)*) => { 1 + builtins!(@count $($rest),*) };

    (@param_attr $default:ident) => { $default() };
    (@param_attr $default:ident $name:expr) => { $name };

    (@types |$bcx:ident| { $($types_init:tt)* }
     @param_attrs |$op:ident| { $($attrs_init:tt)* }
     $($ident:ident = $(#[$attr:expr])* $name:ident($($(@[$param_attr:expr])? $params:expr),* $(,)?) $ret:expr),* $(,)?
    ) => { paste::paste! {
        /// Builtins that can be called by the compiled functions.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Builtin {
            $($ident,)*
        }

        #[allow(unused_variables)]
        impl Builtin {
            pub const COUNT: usize = builtins!(@count $($ident),*);

            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$ident => stringify!($name),)*
                }
            }

            pub fn addr(self) -> usize {
                match self {
                    $(Self::$ident => crate::$name as *const () as usize,)*
                }
            }

            pub fn ret<B: TypeMethods>(self, $bcx: &mut B) -> Option<B::Type> {
                $($types_init)*
                match self {
                    $(Self::$ident => $ret,)*
                }
            }

            pub fn params<B: TypeMethods>(self, $bcx: &mut B) -> Vec<B::Type> {
                $($types_init)*
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

            #[allow(non_upper_case_globals)]
            pub fn param_attrs(self) -> Vec<Vec<Attribute>> {
                #[allow(unused_imports)]
                use Attribute::*;
                let $op = self;
                let default = || vec![Attribute::NoUndef];
                $($attrs_init)*
                match self {
                    $(Self::$ident => vec![$(builtins!(@param_attr default $($param_attr)?)),*]),*
                }
            }
        }
    }};
}

builtins! {
    @types |bcx| {
        let ptr = bcx.type_ptr();
        let usize = bcx.type_ptr_sized_int();
        let bool = bcx.type_int(1);
        let u8 = bcx.type_int(8);
    }

    @param_attrs |op| {
        fn size_and_align<T>() -> Vec<Attribute> {
            size_and_align2(Some(core::mem::size_of::<T>()), core::mem::align_of::<T>())
        }

        fn size_and_align2(size: Option<usize>, align: usize) -> Vec<Attribute> {
            let mut vec = Vec::with_capacity(5);
            vec.push(Attribute::NoAlias);
            vec.push(Attribute::NoCapture);
            vec.push(Attribute::NoUndef);
            vec.push(Attribute::Align(align as u64));
            if let Some(size) = size {
                vec.push(Attribute::Dereferenceable(size as u64));
            }
            vec
        }

        let ecx = || size_and_align::<revmc_context::EvmContext<'static>>();

        let word = || size_and_align::<revmc_context::EvmWord>();

        let word_readonly = || {
            let mut v = size_and_align::<revmc_context::EvmWord>();
            v.push(Attribute::ReadOnly);
            v
        };

        let sp_dyn = || size_and_align2(None, core::mem::align_of::<revmc_context::EvmWord>());
    }

    Panic          = __revmc_builtin_panic(ptr, usize) None,

    UDiv           = __revmc_builtin_udiv(@[word()] ptr, @[word()] ptr) None,
    URem           = __revmc_builtin_urem(@[word()] ptr, @[word()] ptr) None,
    SDiv           = __revmc_builtin_sdiv(@[word()] ptr, @[word()] ptr) None,
    SRem           = __revmc_builtin_srem(@[word()] ptr, @[word()] ptr) None,
    AddMod         = __revmc_builtin_addmod(@[word()] ptr, @[word()] ptr, @[word()] ptr) None,
    MulMod         = __revmc_builtin_mulmod(@[word()] ptr, @[word()] ptr, @[word()] ptr) None,
    Exp            = __revmc_builtin_exp(@[ecx()] ptr, @[word()] ptr, @[word()] ptr) Some(u8),
    Keccak256      = __revmc_builtin_keccak256(@[ecx()] ptr, @[word()] ptr, @[word()] ptr) Some(u8),
    Balance        = __revmc_builtin_balance(@[ecx()] ptr, @[word()] ptr, u8) Some(u8),
    Origin         = __revmc_builtin_origin(@[ecx()] ptr, @[word()] ptr) None,
    CallDataLoad   = __revmc_builtin_calldataload(@[ecx()] ptr, @[word()] ptr) None,
    CallDataSize   = __revmc_builtin_calldatasize(@[ecx()] ptr) Some(usize),
    CallDataCopy   = __revmc_builtin_calldatacopy(@[ecx()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr) Some(u8),
    CodeCopy       = __revmc_builtin_codecopy(@[ecx()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr) Some(u8),
    GasPrice       = __revmc_builtin_gas_price(@[ecx()] ptr, @[word()] ptr) None,
    ExtCodeSize    = __revmc_builtin_extcodesize(@[ecx()] ptr, @[word()] ptr, u8) Some(u8),
    ExtCodeCopy    = __revmc_builtin_extcodecopy(@[ecx()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr, u8) Some(u8),
    ReturnDataCopy = __revmc_builtin_returndatacopy(@[ecx()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr) Some(u8),
    ExtCodeHash    = __revmc_builtin_extcodehash(@[ecx()] ptr, @[word()] ptr, u8) Some(u8),
    BlockHash      = __revmc_builtin_blockhash(@[ecx()] ptr, @[word()] ptr) Some(u8),
    Coinbase       = __revmc_builtin_coinbase(@[ecx()] ptr, @[word()] ptr) None,
    Timestamp      = __revmc_builtin_timestamp(@[ecx()] ptr, @[word()] ptr) None,
    Number         = __revmc_builtin_number(@[ecx()] ptr, @[word()] ptr) None,
    Difficulty     = __revmc_builtin_difficulty(@[ecx()] ptr, @[word()] ptr, u8) None,
    GasLimit       = __revmc_builtin_gaslimit(@[ecx()] ptr, @[word()] ptr) None,
    ChainId        = __revmc_builtin_chainid(@[ecx()] ptr, @[word()] ptr) None,
    SelfBalance    = __revmc_builtin_self_balance(@[ecx()] ptr, @[word()] ptr) Some(u8),
    Basefee        = __revmc_builtin_basefee(@[ecx()] ptr, @[word()] ptr) None,
    BlobHash       = __revmc_builtin_blob_hash(@[ecx()] ptr, @[word()] ptr) None,
    BlobBaseFee    = __revmc_builtin_blob_base_fee(@[ecx()] ptr, @[word()] ptr) None,
    Sload          = __revmc_builtin_sload(@[ecx()] ptr, @[word()] ptr, u8) Some(u8),
    Sstore         = __revmc_builtin_sstore(@[ecx()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr, u8) Some(u8),
    Msize          = __revmc_builtin_msize(@[ecx()] ptr) Some(usize),
    Tstore         = __revmc_builtin_tstore(@[ecx()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr) Some(u8),
    Tload          = __revmc_builtin_tload(@[ecx()] ptr, @[word()] ptr) None,
    Mcopy          = __revmc_builtin_mcopy(@[ecx()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr) Some(u8),
    Log            = __revmc_builtin_log(@[ecx()] ptr, @[sp_dyn()] ptr, u8) Some(u8),

    Create         = __revmc_builtin_create(@[ecx()] ptr, @[sp_dyn()] ptr, u8, u8) Some(u8),
    Call           = __revmc_builtin_call(@[ecx()] ptr, @[sp_dyn()] ptr, u8, u8) Some(u8),
    DoReturn       = __revmc_builtin_do_return(@[ecx()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr, u8) Some(u8),
    SelfDestruct   = __revmc_builtin_selfdestruct(@[ecx()] ptr, @[word_readonly()] ptr, u8) Some(u8),

    ResizeMemory   = __revmc_builtin_resize_memory(@[ecx()] ptr, usize) Some(u8),
    Mload          = __revmc_builtin_mload(@[ecx()] ptr, @[word()] ptr) Some(u8),
    Mstore         = __revmc_builtin_mstore(@[ecx()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr) Some(u8),
    Mstore8        = __revmc_builtin_mstore8(@[ecx()] ptr, @[word_readonly()] ptr, @[word_readonly()] ptr) Some(u8),
}

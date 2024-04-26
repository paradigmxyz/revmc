use alloc::vec::Vec;
use revm_jit_backend::{Attribute, Backend, Builder, FunctionAttributeLocation, TypeMethods};

/// Builtin cache.
#[derive(Debug)]
pub struct Builtins<B: Backend>([Option<B::Function>; Builtin::COUNT]);

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
        let mut name = builtin.name();
        let mangle_prefix = "__revm_jit_builtin_";
        let storage;
        if !name.starts_with(mangle_prefix) {
            storage = [mangle_prefix, name].concat();
            name = storage.as_str();
        }
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
        let linkage = revm_jit_backend::Linkage::Import;
        let f = bcx.add_function(name, ret, &params, address, linkage);
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
                Attribute::Speculatable,
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
                    $(Self::$ident => crate::$name as usize,)*
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

            fn op(self) -> u8 {
                use revm_interpreter::opcode::*;
                const PANIC: u8 = 0;
                const LOG: u8 = LOG0;
                const DORETURN: u8 = RETURN;

                match self {
                    $(Self::$ident => [<$ident:upper>]),*
                }
            }
        }
    }};
}

// NOTE: If the format of this macro invocation is changed,
// the build script support crate must be updated as well.
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

        let op = op.op();
        let (inputs, outputs) = if let Some(info) = revm_interpreter::opcode::OPCODE_INFO_JUMPTABLE[op as usize] {
            (info.inputs(), info.outputs())
        } else {
            (0, 0)
        };

        let ecx = size_and_align::<revm_jit_context::EvmContext<'static>>();

        let sp_dyn = size_and_align2(None, core::mem::align_of::<revm_jit_context::EvmWord>());

        let mut sp = sp_dyn.clone();
        sp.push(Attribute::Dereferenceable((core::mem::size_of::<revm_jit_context::EvmWord>() * inputs as usize) as u64));
        match (inputs, outputs) {
            (0, 0) => sp.push(Attribute::ReadNone),
            (0, 1..) => sp.push(Attribute::WriteOnly),
            (1.., 0) => sp.push(Attribute::ReadOnly),
            (1.., 1..) => {}
        }
    }

    Panic          = __revm_jit_builtin_panic(ptr, usize) None,

    AddMod         = __revm_jit_builtin_addmod(@[sp] ptr) None,
    MulMod         = __revm_jit_builtin_mulmod(@[sp] ptr) None,
    Exp            = __revm_jit_builtin_exp(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    Keccak256      = __revm_jit_builtin_keccak256(@[ecx] ptr, @[sp] ptr) Some(u8),
    Balance        = __revm_jit_builtin_balance(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    CallDataCopy   = __revm_jit_builtin_calldatacopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    CodeSize       = __revm_jit_builtin_codesize(@[ecx] ptr) Some(usize),
    CodeCopy       = __revm_jit_builtin_codecopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    ExtCodeSize    = __revm_jit_builtin_extcodesize(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    ExtCodeCopy    = __revm_jit_builtin_extcodecopy(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    ReturnDataCopy = __revm_jit_builtin_returndatacopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    ExtCodeHash    = __revm_jit_builtin_extcodehash(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    BlockHash      = __revm_jit_builtin_blockhash(@[ecx] ptr, @[sp] ptr) Some(u8),
    SelfBalance    = __revm_jit_builtin_self_balance(@[ecx] ptr, @[sp] ptr) Some(u8),
    BlobHash       = __revm_jit_builtin_blob_hash(@[ecx] ptr, @[sp] ptr) None,
    BlobBaseFee    = __revm_jit_builtin_blob_base_fee(@[ecx] ptr, @[sp] ptr) None,
    Mload          = __revm_jit_builtin_mload(@[ecx] ptr, @[sp] ptr) Some(u8),
    Mstore         = __revm_jit_builtin_mstore(@[ecx] ptr, @[sp] ptr) Some(u8),
    Mstore8        = __revm_jit_builtin_mstore8(@[ecx] ptr, @[sp] ptr) Some(u8),
    Sload          = __revm_jit_builtin_sload(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    Sstore         = __revm_jit_builtin_sstore(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    Msize          = __revm_jit_builtin_msize(@[ecx] ptr) Some(usize),
    Tstore         = __revm_jit_builtin_tstore(@[ecx] ptr, @[sp] ptr) None,
    Tload          = __revm_jit_builtin_tload(@[ecx] ptr, @[sp] ptr) None,
    Mcopy          = __revm_jit_builtin_mcopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    Log            = __revm_jit_builtin_log(@[ecx] ptr, @[sp_dyn] ptr, u8) Some(u8),

    Create         = __revm_jit_builtin_create(@[ecx] ptr, @[sp_dyn] ptr, u8, u8) Some(u8),
    Call           = __revm_jit_builtin_call(@[ecx] ptr, @[sp_dyn] ptr, u8, u8) Some(u8),
    DoReturn       = __revm_jit_builtin_do_return(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    SelfDestruct   = __revm_jit_builtin_selfdestruct(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
}

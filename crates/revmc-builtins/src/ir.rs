use revmc_backend::{Attribute, Backend, Builder, FunctionAttributeLocation, TypeMethods};

// Must be kept in sync with `remvc-build`.
const MANGLE_PREFIX: &str = "__revmc_builtin_";

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
                const RESIZEMEMORY: u8 = 0;
                const FUNCSTACKPUSH: u8 = 0;
                const FUNCSTACKPOP: u8 = 0;
                const FUNCSTACKGROW: u8 = 0;

                match self {
                    $(Self::$ident => [<$ident:upper>]),*
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

        let op = op.op();
        let (inputs, outputs) = if let Some(info) = revm_interpreter::opcode::OPCODE_INFO_JUMPTABLE[op as usize] {
            (info.inputs(), info.outputs())
        } else {
            (0, 0)
        };

        let ecx = size_and_align::<revmc_context::EvmContext<'static>>();

        let sp_dyn = size_and_align2(None, core::mem::align_of::<revmc_context::EvmWord>());

        let mut sp = sp_dyn.clone();
        // `sp` is at `top - inputs`, we have access to `max(inputs, outputs)` words.
        let n_stack_words = inputs.max(outputs);
        let size_of_word = core::mem::size_of::<revmc_context::EvmWord>();
        sp.push(Attribute::Dereferenceable(size_of_word as u64 * n_stack_words as u64));
        match (inputs, outputs) {
            (0, 0) => sp.push(Attribute::ReadNone),
            (0, 1..) => sp.push(Attribute::WriteOnly),
            (1.., 0) => sp.push(Attribute::ReadOnly),
            (1.., 1..) => {}
        }
    }

    Panic          = __revmc_builtin_panic(ptr, usize) None,

    AddMod         = __revmc_builtin_addmod(@[sp] ptr) None,
    MulMod         = __revmc_builtin_mulmod(@[sp] ptr) None,
    Exp            = __revmc_builtin_exp(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    Keccak256      = __revmc_builtin_keccak256(@[ecx] ptr, @[sp] ptr) Some(u8),
    Balance        = __revmc_builtin_balance(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    CallDataCopy   = __revmc_builtin_calldatacopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    CodeSize       = __revmc_builtin_codesize(@[ecx] ptr) Some(usize),
    CodeCopy       = __revmc_builtin_codecopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    GasPrice       = __revmc_builtin_gas_price(@[ecx] ptr, @[sp] ptr) None,
    ExtCodeSize    = __revmc_builtin_extcodesize(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    ExtCodeCopy    = __revmc_builtin_extcodecopy(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    ReturnDataCopy = __revmc_builtin_returndatacopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    ExtCodeHash    = __revmc_builtin_extcodehash(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    BlockHash      = __revmc_builtin_blockhash(@[ecx] ptr, @[sp] ptr) Some(u8),
    Difficulty     = __revmc_builtin_difficulty(@[ecx] ptr, @[sp] ptr, u8) None,
    SelfBalance    = __revmc_builtin_self_balance(@[ecx] ptr, @[sp] ptr) Some(u8),
    BlobHash       = __revmc_builtin_blob_hash(@[ecx] ptr, @[sp] ptr) None,
    BlobBaseFee    = __revmc_builtin_blob_base_fee(@[ecx] ptr, @[sp] ptr) None,
    Sload          = __revmc_builtin_sload(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    Sstore         = __revmc_builtin_sstore(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    Msize          = __revmc_builtin_msize(@[ecx] ptr) Some(usize),
    Tstore         = __revmc_builtin_tstore(@[ecx] ptr, @[sp] ptr) Some(u8),
    Tload          = __revmc_builtin_tload(@[ecx] ptr, @[sp] ptr) None,
    Mcopy          = __revmc_builtin_mcopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    Log            = __revmc_builtin_log(@[ecx] ptr, @[sp_dyn] ptr, u8) Some(u8),
    DataLoad       = __revmc_builtin_data_load(@[ecx] ptr, @[sp] ptr) None,
    DataCopy       = __revmc_builtin_data_copy(@[ecx] ptr, @[sp] ptr) Some(u8),
    ReturnDataLoad = __revmc_builtin_returndataload(@[ecx] ptr, @[sp] ptr) None,

    EofCreate      = __revmc_builtin_eof_create(@[ecx] ptr, @[sp] ptr, usize) Some(u8),
    ReturnContract = __revmc_builtin_return_contract(@[ecx] ptr, @[sp] ptr, usize) Some(u8),
    Create         = __revmc_builtin_create(@[ecx] ptr, @[sp_dyn] ptr, u8, u8) Some(u8),
    Call           = __revmc_builtin_call(@[ecx] ptr, @[sp_dyn] ptr, u8, u8) Some(u8),
    ExtCall        = __revmc_builtin_ext_call(@[ecx] ptr, @[sp_dyn] ptr, u8, u8) Some(u8),
    DoReturn       = __revmc_builtin_do_return(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    SelfDestruct   = __revmc_builtin_selfdestruct(@[ecx] ptr, @[sp] ptr, u8) Some(u8),

    FuncStackPush  = __revmc_builtin_func_stack_push(@[ecx] ptr, ptr, usize) Some(u8),
    FuncStackPop   = __revmc_builtin_func_stack_pop(@[ecx] ptr) Some(ptr),
    FuncStackGrow  = __revmc_builtin_func_stack_grow(@[ecx] ptr) None,

    ResizeMemory   = __revmc_builtin_resize_memory(@[ecx] ptr, usize) Some(u8),
}

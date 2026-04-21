use revmc_backend::{Attribute, Backend, Builder, FunctionAttributeLocation, TypeMethods};

// Must be kept in sync with `remvc-build`.
const MANGLE_PREFIX: &str = "__revmc_builtin_";

/// Builtin function information.
#[derive(Debug)]
pub struct BuiltinInfo<B: Backend> {
    pub func: B::Function,
}

impl<B: Backend> Clone for BuiltinInfo<B> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Backend> Copy for BuiltinInfo<B> {}

/// Builtin cache.
#[derive(Debug)]
pub struct Builtins<B: Backend>([Option<BuiltinInfo<B>>; Builtin::COUNT]);

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
    pub fn get(&mut self, builtin: Builtin, bcx: &mut B::Builder<'_>) -> BuiltinInfo<B> {
        *self.0[builtin as usize].get_or_insert_with(|| Self::init(builtin, bcx))
    }

    #[cold]
    fn init(builtin: Builtin, bcx: &mut B::Builder<'_>) -> BuiltinInfo<B> {
        let name = builtin.name();
        debug_assert!(name.starts_with(MANGLE_PREFIX), "{name:?}");
        let func = bcx.get_function(name).unwrap_or_else(|| Self::build(name, builtin, bcx));
        BuiltinInfo { func }
    }

    fn build(name: &str, builtin: Builtin, bcx: &mut B::Builder<'_>) -> B::Function {
        let ret = builtin.ret(bcx);
        let params = builtin.params(bcx);
        let param_attrs = builtin.param_attrs();
        let address = builtin.addr();
        let linkage = revmc_backend::Linkage::Import;
        let f = bcx.add_function(name, &params, ret, Some(address), linkage);
        let mut attrs = Vec::with_capacity(16);
        attrs.extend(builtin.attrs());
        attrs.extend([
            Attribute::NoFree,
            Attribute::NoRecurse,
            Attribute::NoSync,
            Attribute::NoUnwind,
        ]);
        for attr in attrs {
            bcx.add_function_attribute(Some(f), attr, FunctionAttributeLocation::Function);
        }
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

            fn op(self) -> u8 {
                use revm_bytecode::opcode::*;

                // _in_out
                const _0_0: u8 = STOP;
                const _0_1: u8 = PUSH0;
                const _1_0: u8 = POP;

                const PANIC: u8 = _0_0;
                const ASSERTSPECID: u8 = _0_0;

                const KECCAK256CC: u8 = _0_1;

                const CALLDATALOADC: u8 = _0_1;
                const MLOADC: u8 = _0_1;
                const SLOADC: u8 = _0_1;

                const MSTORECD: u8 = _1_0;
                const MSTOREDC: u8 = _1_0;
                const MSTORECC: u8 = _0_0;

                const LOG: u8 = _0_0;
                const DORETURN: u8 = RETURN;
                const DORETURNCC: u8 = _0_0;

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
        let u8 = bcx.type_int(8);
    }

    @param_attrs |op| {
        fn size_and_align<T>() -> Vec<Attribute> {
            size_and_align_with(Some(core::mem::size_of::<T>()), core::mem::align_of::<T>())
        }

        fn size_and_align_with(size: Option<usize>, align: usize) -> Vec<Attribute> {
            let mut vec = Vec::with_capacity(8);
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
        let (inputs, outputs) = if let Some(info) = revm_bytecode::opcode::OPCODE_INFO[op as usize] {
            (info.inputs(), info.outputs())
        } else {
            (0, 0)
        };

        let ecx_base = size_and_align::<revmc_context::EvmContext<'static>>();
        let mut ecx = ecx_base.clone();
        ecx.push(Attribute::Writable);
        let mut ecx_ro = ecx_base;
        ecx_ro.push(Attribute::ReadOnly);

        let sp_base = size_and_align_with(None, core::mem::align_of::<revmc_context::EvmWord>());
        let mut sp_dyn = sp_base.clone();
        sp_dyn.push(Attribute::Writable);

        let mut sp = sp_base;
        // `sp` is at `top - inputs`, we have access to `max(inputs, outputs)` words.
        let n_stack_words = inputs.max(outputs);
        let size_of_word = core::mem::size_of::<revmc_context::EvmWord>();
        sp.push(Attribute::Dereferenceable(size_of_word as u64 * n_stack_words as u64));
        match (inputs, outputs) {
            (0, 0) => sp.push(Attribute::ReadNone),
            (0, 1..) => {
                sp.push(Attribute::Writable);
                sp.push(Attribute::WriteOnly);
                sp.push(Attribute::Initializes(size_of_word as u64 * outputs as u64));
            }
            (1.., 0) => sp.push(Attribute::ReadOnly),
            (1.., 1..) => sp.push(Attribute::Writable),
        }
    }

    Panic          = __revmc_builtin_panic(ptr, usize) None,
    AssertSpecId   = __revmc_builtin_assert_spec_id(@[ecx] ptr, u8) None,

    Div            = __revmc_builtin_div(@[sp] ptr) None,
    SDiv           = __revmc_builtin_sdiv(@[sp] ptr) None,
    Mod            = __revmc_builtin_mod(@[sp] ptr) None,
    SMod           = __revmc_builtin_smod(@[sp] ptr) None,
    AddMod         = __revmc_builtin_addmod(@[sp] ptr) None,
    MulMod         = __revmc_builtin_mulmod(@[sp] ptr) None,
    Exp            = __revmc_builtin_exp(@[ecx] ptr, @[sp] ptr) Some(u8),
    Keccak256      = __revmc_builtin_keccak256(@[ecx] ptr, @[sp] ptr) Some(u8),
    Keccak256CC    = __revmc_builtin_keccak256_cc(@[ecx] ptr, @[sp] ptr, usize, usize) Some(u8),
    Address        = __revmc_builtin_address(@[ecx_ro] ptr, @[sp] ptr) None,
    Balance        = __revmc_builtin_balance(@[ecx] ptr, @[sp] ptr) Some(u8),
    Origin         = __revmc_builtin_origin(@[ecx_ro] ptr, @[sp] ptr) None,
    Caller         = __revmc_builtin_caller(@[ecx_ro] ptr, @[sp] ptr) None,
    CallValue      = __revmc_builtin_call_value(@[ecx_ro] ptr, @[sp] ptr) None,
    CallDataLoad   = __revmc_builtin_calldataload(@[ecx_ro] ptr, @[sp] ptr) None,
    CallDataLoadC  = __revmc_builtin_calldataload_c(@[ecx_ro] ptr, @[sp] ptr, usize) None,
    CallDataCopy   = __revmc_builtin_calldatacopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    CodeCopy       = __revmc_builtin_codecopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    GasPrice       = __revmc_builtin_gas_price(@[ecx_ro] ptr, @[sp] ptr) None,
    ExtCodeSize    = __revmc_builtin_extcodesize(@[ecx] ptr, @[sp] ptr) Some(u8),
    ExtCodeCopy    = __revmc_builtin_extcodecopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    ReturnDataSize = __revmc_builtin_returndatasize(@[ecx_ro] ptr) Some(usize),
    ReturnDataCopy = __revmc_builtin_returndatacopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    ExtCodeHash    = __revmc_builtin_extcodehash(@[ecx] ptr, @[sp] ptr) Some(u8),
    BlockHash      = __revmc_builtin_blockhash(@[ecx] ptr, @[sp] ptr) Some(u8),
    Coinbase       = __revmc_builtin_coinbase(@[ecx_ro] ptr, @[sp] ptr) None,
    Timestamp      = __revmc_builtin_timestamp(@[ecx_ro] ptr, @[sp] ptr) None,
    Number         = __revmc_builtin_number(@[ecx_ro] ptr, @[sp] ptr) None,
    Difficulty     = __revmc_builtin_difficulty(@[ecx_ro] ptr, @[sp] ptr) None,
    GasLimit       = __revmc_builtin_gaslimit(@[ecx_ro] ptr, @[sp] ptr) None,
    ChainId        = __revmc_builtin_chainid(@[ecx_ro] ptr, @[sp] ptr) None,
    SelfBalance    = __revmc_builtin_self_balance(@[ecx] ptr, @[sp] ptr) Some(u8),
    Basefee        = __revmc_builtin_basefee(@[ecx_ro] ptr, @[sp] ptr) None,
    BlobHash       = __revmc_builtin_blob_hash(@[ecx_ro] ptr, @[sp] ptr) None,
    BlobBaseFee    = __revmc_builtin_blob_base_fee(@[ecx_ro] ptr, @[sp] ptr) None,
    SlotNum        = __revmc_builtin_slot_num(@[ecx_ro] ptr, @[sp] ptr) None,
    Mload          = __revmc_builtin_mload(@[ecx] ptr, @[sp] ptr) Some(u8),
    MloadC         = __revmc_builtin_mload_c(@[ecx] ptr, @[sp] ptr, usize) Some(u8),
    Mstore         = __revmc_builtin_mstore(@[ecx] ptr, @[sp] ptr) Some(u8),
    MstoreCD       = __revmc_builtin_mstore_cd(@[ecx] ptr, usize, @[sp] ptr) Some(u8),
    MstoreDC       = __revmc_builtin_mstore_dc(@[ecx] ptr, @[sp] ptr, usize) Some(u8),
    MstoreCC       = __revmc_builtin_mstore_cc(@[ecx] ptr, usize, usize) Some(u8),
    Mstore8        = __revmc_builtin_mstore8(@[ecx] ptr, @[sp] ptr) Some(u8),
    Sload          = __revmc_builtin_sload(@[ecx] ptr, @[sp] ptr) Some(u8),
    SloadC         = __revmc_builtin_sload_c(@[ecx] ptr, @[sp] ptr, usize) Some(u8),
    Sstore         = __revmc_builtin_sstore(@[ecx] ptr, @[sp] ptr) Some(u8),
    Msize          = __revmc_builtin_msize(@[ecx_ro] ptr) Some(usize),
    Tload          = __revmc_builtin_tload(@[ecx] ptr, @[sp] ptr) None,
    Tstore         = __revmc_builtin_tstore(@[ecx] ptr, @[sp] ptr) Some(u8),
    Mcopy          = __revmc_builtin_mcopy(@[ecx] ptr, @[sp] ptr) Some(u8),
    Log            = __revmc_builtin_log(@[ecx] ptr, @[sp_dyn] ptr, u8) Some(u8),

    Create         = __revmc_builtin_create(@[ecx] ptr, @[sp_dyn] ptr, u8) Some(u8),
    Call           = __revmc_builtin_call(@[ecx] ptr, @[sp_dyn] ptr, u8) Some(u8),
    DoReturn       = __revmc_builtin_do_return(@[ecx] ptr, @[sp] ptr, u8) Some(u8),
    DoReturnCC     = __revmc_builtin_do_return_cc(@[ecx] ptr, usize, usize, u8) Some(u8),
    SelfDestruct   = __revmc_builtin_selfdestruct(@[ecx] ptr, @[sp] ptr) Some(u8),
}

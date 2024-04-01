use super::Callback;
use revm_jit_backend::{Attribute, Backend, Builder, FunctionAttributeLocation};

/// Callback cache.
pub(crate) struct Callbacks<B: Backend>([Option<B::Function>; Callback::COUNT]);

impl<B: Backend> Callbacks<B> {
    pub(crate) fn new() -> Self {
        Self([None; Callback::COUNT])
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::new();
    }

    pub(crate) fn get(&mut self, cb: Callback, bcx: &mut B::Builder<'_>) -> B::Function {
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
        bcx.get_function(name).unwrap_or_else(|| Self::build(name, cb, bcx))
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

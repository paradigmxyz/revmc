//! Safe wrappers for LLVM ORC and LLJIT, which are not yet implemented in [`inkwell`].
//!
//! Header files implemented:
//! - `LLJIT.h`
//! - `LLJITUtils.h` (experimental, not available in `llvm-sys`)
//! - `Orc.h`
//! - `OrcEE.h` (experimental)
//!
//! Note that lifetimes are implicit here, I just didn't bother to add them.

#![allow(missing_debug_implementations)]
#![allow(clippy::new_without_default)]
#![allow(clippy::missing_safety_doc)]

use inkwell::{
    context::ContextRef,
    llvm_sys::{
        error::*,
        orc2::{lljit::*, *},
        prelude::*,
    },
    module::Module,
    support::LLVMString,
    targets::TargetMachine,
};
use std::{
    ffi::{c_char, c_void, CStr, CString},
    fmt,
    mem::{self, MaybeUninit},
    panic::AssertUnwindSafe,
    ptr,
};

/// A thread-safe LLVM context.
///
/// Must use a lock to access the context in multi-threaded scenarios.
///
/// See [the ORCv2 docs](https://releases.llvm.org/17.0.1/docs/ORCv2.html).
pub struct ThreadSafeContext {
    ctx: LLVMOrcThreadSafeContextRef,
}

impl ThreadSafeContext {
    /// Creates a new thread-safe context.
    pub fn new() -> Self {
        unsafe { Self::from_inner(LLVMOrcCreateNewThreadSafeContext()) }
    }

    /// Wraps a raw pointer.
    pub unsafe fn from_inner(ctx: LLVMOrcThreadSafeContextRef) -> Self {
        Self { ctx }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcThreadSafeContextRef {
        self.ctx
    }

    /// Get a reference to the wrapped LLVMContext.
    pub fn get_context(&self) -> ContextRef<'_> {
        let ptr = unsafe { LLVMOrcThreadSafeContextGetContext(self.as_inner()) };
        // `ContextRef::new` is private.
        unsafe { std::mem::transmute(ptr) }
    }

    /// Create a ThreadSafeModule wrapper around the given LLVM module.
    pub fn create_module<'ctx>(&'ctx self, module: Module<'ctx>) -> ThreadSafeModule {
        ThreadSafeModule::create_in_context(module, self)
    }
}

unsafe impl Send for ThreadSafeContext {}

impl Drop for ThreadSafeContext {
    fn drop(&mut self) {
        unsafe { LLVMOrcDisposeThreadSafeContext(self.ctx) };
    }
}

/// A thread-safe module.
///
/// See [the ORCv2 docs](https://releases.llvm.org/17.0.1/docs/ORCv2.html).
// NOTE: A lifetime is not neeeded according to `LLVMOrcCreateNewThreadSafeContext`
// > Ownership of the underlying ThreadSafeContext data is shared: Clients can and should dispose of
// > their ThreadSafeContext as soon as they no longer need to refer to it directly. Other
// > references (e.g. from ThreadSafeModules) will keep the data alive as long as it is needed.
pub struct ThreadSafeModule {
    ptr: LLVMOrcThreadSafeModuleRef,
}

impl ThreadSafeModule {
    /// Creates a new module with the given name in a new context.
    pub fn create(name: &str) -> (Self, ThreadSafeContext) {
        let cx = ThreadSafeContext::new();
        let m = cx.get_context().create_module(name);
        (Self::create_in_context(m, &cx), cx)
    }

    /// Create a ThreadSafeModule wrapper around the given LLVM module.
    pub fn create_in_context<'ctx>(module: Module<'ctx>, ctx: &'ctx ThreadSafeContext) -> Self {
        let ptr = unsafe { LLVMOrcCreateNewThreadSafeModule(module.as_mut_ptr(), ctx.as_inner()) };
        mem::forget(module);
        Self { ptr }
    }

    /// Wraps a raw pointer.
    pub unsafe fn from_inner(ptr: LLVMOrcThreadSafeModuleRef) -> Self {
        Self { ptr }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcThreadSafeModuleRef {
        self.ptr
    }

    /// Runs the given closure with the module.
    ///
    /// This implicitly locks the associated context.
    pub fn with_module<'a>(
        &'a self,
        mut f: &mut dyn FnMut(&Module<'a>) -> Result<(), String>,
    ) -> Result<(), LLVMString> {
        extern "C" fn shim(ctx: *mut c_void, m: LLVMModuleRef) -> LLVMErrorRef {
            let f = ctx.cast::<&mut dyn FnMut(&Module<'_>) -> Result<(), String>>();
            let m = unsafe { Module::new(m) };
            let res = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe { (*f)(&m) }));
            mem::forget(m);
            cvt_cb_res(res)
        }

        let ctx = &mut f as *mut _ as *mut c_void;
        cvt(unsafe { LLVMOrcThreadSafeModuleWithModuleDo(self.as_inner(), shim, ctx) })
    }
}

unsafe impl Send for ThreadSafeModule {}

impl Drop for ThreadSafeModule {
    fn drop(&mut self) {
        unsafe { LLVMOrcDisposeThreadSafeModule(self.ptr) };
    }
}

/// A symbol string pool reference.
#[repr(C)]
pub struct SymbolStringPoolRef {
    ptr: LLVMOrcSymbolStringPoolRef,
}

impl SymbolStringPoolRef {
    /// Wraps a raw pointer.
    pub unsafe fn from_inner(ptr: LLVMOrcSymbolStringPoolRef) -> Self {
        Self { ptr }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcSymbolStringPoolRef {
        self.ptr
    }

    /// Clear all unreferenced symbol string pool entries.
    ///
    /// This can be called at any time to release unused entries in the
    /// ExecutionSession's string pool. Since it locks the pool (preventing
    /// interning of any new strings) it is recommended that it only be called
    /// infrequently, ideally when the caller has reason to believe that some
    /// entries will have become unreferenced, e.g. after removing a module or
    /// closing a JITDylib.
    pub fn clear_dead_entries(&self) {
        unsafe { LLVMOrcSymbolStringPoolClearDeadEntries(self.as_inner()) };
    }
}

/// A reference-counted unique string interned in a `SymbolStringPool`.
#[derive(PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SymbolStringPoolEntry {
    ptr: LLVMOrcSymbolStringPoolEntryRef,
}

impl fmt::Debug for SymbolStringPoolEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_cstr().fmt(f)
    }
}

impl std::ops::Deref for SymbolStringPoolEntry {
    type Target = CStr;

    fn deref(&self) -> &Self::Target {
        self.as_cstr()
    }
}

impl SymbolStringPoolEntry {
    /// Wraps a raw pointer.
    pub unsafe fn from_inner(ptr: LLVMOrcSymbolStringPoolEntryRef) -> Self {
        Self { ptr }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcSymbolStringPoolEntryRef {
        self.ptr
    }

    /// Convert to a C string.
    pub fn as_cstr(&self) -> &CStr {
        unsafe { CStr::from_ptr(LLVMOrcSymbolStringPoolEntryStr(self.ptr)) }
    }
}

impl Drop for SymbolStringPoolEntry {
    fn drop(&mut self) {
        unsafe { LLVMOrcReleaseSymbolStringPoolEntry(self.ptr) }
    }
}

/// An evaluated symbol.
#[repr(C)]
pub struct EvaluatedSymbol {
    address: u64,
    flags: SymbolFlags,
}

impl EvaluatedSymbol {
    /// Create a new EvaluatedSymbol.
    pub fn new(address: usize) -> Self {
        Self { address: address as u64, flags: SymbolFlags::none() }
    }
}

/// Symbol flags.
#[repr(C)]
pub struct SymbolFlags {
    generic: u8,
    target: u8,
}

impl SymbolFlags {
    /// Create a new, empty SymbolFlags.
    pub fn none() -> Self {
        Self { generic: LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsNone as u8, target: 0 }
    }

    /// Set the `Exported` flag.
    pub fn set_exported(&mut self) {
        self.generic |= LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsExported as u8;
    }

    /// Set the `Exported` flag.
    pub fn with_exported(mut self) -> Self {
        self.set_exported();
        self
    }

    /// Returns `true` if the `Exported` flag is set.
    pub fn is_exported(&self) -> bool {
        (self.generic & LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsExported as u8) != 0
    }

    /// Set the `Weak` flag.
    pub fn set_weak(&mut self) {
        self.generic |= LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsWeak as u8;
    }

    /// Set the `Weak` flag.
    pub fn weak(mut self) -> Self {
        self.set_weak();
        self
    }

    /// Returns `true` if the `Weak` flag is set.
    pub fn is_weak(&self) -> bool {
        (self.generic & LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsWeak as u8) != 0
    }

    /// Set the `Callable` flag.
    pub fn set_callable(&mut self) {
        self.generic |= LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsCallable as u8;
    }

    /// Set the `Callable` flag.
    pub fn callable(mut self) -> Self {
        self.set_callable();
        self
    }

    /// Returns `true` if the `Callable` flag is set.
    pub fn is_callable(&self) -> bool {
        (self.generic & LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsCallable as u8) != 0
    }

    /// Set the `MaterializationSideEffectsOnly` flag.
    pub fn set_materialization_side_effects_only(&mut self) {
        self.generic |=
            LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly
                as u8;
    }

    /// Set the `MaterializationSideEffectsOnly` flag.
    pub fn materialization_side_effects_only(mut self) -> Self {
        self.set_materialization_side_effects_only();
        self
    }

    /// Returns `true` if the `MaterializationSideEffectsOnly` flag is set.
    pub fn is_materialization_side_effects_only(&self) -> bool {
        (self.generic
            & LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly
                as u8)
            != 0
    }

    /// Add a generic flag.
    pub fn set_generic(&mut self, flag: LLVMJITSymbolGenericFlags) {
        self.generic |= flag as u8;
    }

    /// Add a generic flag.
    pub fn with_generic(mut self, flag: LLVMJITSymbolGenericFlags) -> Self {
        self.set_generic(flag);
        self
    }

    /// Add a target flag.
    pub fn set_target(&mut self, flag: LLVMJITSymbolTargetFlags) {
        self.target |= flag;
    }

    /// Add a target flag.
    pub fn with_target(mut self, flag: LLVMJITSymbolTargetFlags) -> Self {
        self.set_target(flag);
        self
    }
}

/// A pair of a symbol name and an evaluated symbol.
#[repr(C)]
pub struct SymbolFlagsMapPair {
    name: SymbolStringPoolEntry,
    evaluated_symbol: EvaluatedSymbol,
}

impl SymbolFlagsMapPair {
    /// Create a new pair.
    pub fn new(name: SymbolStringPoolEntry, evaluated_symbol: EvaluatedSymbol) -> Self {
        Self { name, evaluated_symbol }
    }
}

/// An owned list of symbol flags map pairs.
///
/// Returned by [`MaterializationResponsibility::get_symbols`].
pub struct SymbolFlagsMapPairs<'a>(&'a [SymbolFlagsMapPair]);

impl std::ops::Deref for SymbolFlagsMapPairs<'_> {
    type Target = [SymbolFlagsMapPair];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl Drop for SymbolFlagsMapPairs<'_> {
    fn drop(&mut self) {
        unsafe { LLVMOrcDisposeCSymbolFlagsMap(self.0.as_ptr().cast_mut().cast()) };
    }
}

/// A materialization responsibility.
pub struct MaterializationUnit {
    mu: LLVMOrcMaterializationUnitRef,
}

impl MaterializationUnit {
    /// Create a MaterializationUnit to define the given symbols as pointing to the corresponding
    /// raw addresses.
    pub fn absolute_symbols(syms: Vec<SymbolFlagsMapPair>) -> Self {
        /*
         * This function takes ownership of the elements of the Syms array. The Name
         * fields of the array elements are taken to have been retained for this
         * function. This allows the following pattern...
         *
         *   size_t NumPairs;
         *   LLVMOrcCSymbolMapPairs Sym;
         *   -- Build Syms array --
         *   LLVMOrcMaterializationUnitRef MU =
         *       LLVMOrcAbsoluteSymbols(Syms, NumPairs);
         *
         * ... without requiring cleanup of the elements of the Sym array afterwards.
         *
         * The client is still responsible for deleting the Sym array itself.
         *
         * If a client wishes to reuse elements of the Sym array after this call they
         * must explicitly retain each of the elements for themselves.
         */
        let m = unsafe { Self::absolute_symbols_raw(syms.as_ptr().cast_mut().cast(), syms.len()) };
        // TODO: Is this the best way to deallocate without running the elements' destructors?
        syms.into_iter().for_each(mem::forget);
        m
    }

    /// Create a MaterializationUnit to define the given symbols as pointing to the corresponding
    /// raw addresses.
    pub unsafe fn absolute_symbols_raw(syms: LLVMOrcCSymbolMapPairs, len: usize) -> Self {
        unsafe { Self::from_inner(LLVMOrcAbsoluteSymbols(syms, len)) }
    }

    /// Wraps a raw pointer.
    pub unsafe fn from_inner(mu: LLVMOrcMaterializationUnitRef) -> Self {
        Self { mu }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcMaterializationUnitRef {
        self.mu
    }
}

impl Drop for MaterializationUnit {
    fn drop(&mut self) {
        unsafe { LLVMOrcDisposeMaterializationUnit(self.as_inner()) };
    }
}

/// A materialization responsibility.
pub struct MaterializationResponsibility {
    mr: LLVMOrcMaterializationResponsibilityRef,
}

impl MaterializationResponsibility {
    /// Wraps a raw pointer.
    pub unsafe fn from_inner(mr: LLVMOrcMaterializationResponsibilityRef) -> Self {
        Self { mr }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcMaterializationResponsibilityRef {
        self.mr
    }

    /// Returns the target JITDylib that these symbols are being materialized into.
    pub fn get_target_dylib(&self) -> JITDylibRef {
        unsafe {
            JITDylibRef::from_inner(LLVMOrcMaterializationResponsibilityGetTargetDylib(
                self.as_inner(),
            ))
        }
    }

    /// Returns the ExecutionSession for this MaterializationResponsibility.
    pub fn get_execution_session(&self) -> ExecutionSessionRef {
        unsafe {
            ExecutionSessionRef::from_inner(
                LLVMOrcMaterializationResponsibilityGetExecutionSession(self.as_inner()),
            )
        }
    }

    /// Returns the symbol flags map for this responsibility instance.
    pub fn get_symbols(&self) -> SymbolFlagsMapPairs<'_> {
        /*
         * The length of the array is returned in NumPairs and the caller is responsible
         * for the returned memory and needs to call LLVMOrcDisposeCSymbolFlagsMap.
         *
         * To use the returned symbols beyond the livetime of the
         * MaterializationResponsibility requires the caller to retain the symbols
         * explicitly.
         */
        let mut len = MaybeUninit::uninit();
        let data = unsafe {
            LLVMOrcMaterializationResponsibilityGetSymbols(self.as_inner(), len.as_mut_ptr())
        };
        SymbolFlagsMapPairs(unsafe { std::slice::from_raw_parts(data.cast(), len.assume_init()) })
    }

    /// Returns the initialization pseudo-symbol, if any.
    ///
    /// This symbol will also be present in the SymbolFlagsMap for this
    /// MaterializationResponsibility object.
    pub fn get_initializer_symbol(&self) -> Option<SymbolStringPoolEntry> {
        let ptr =
            unsafe { LLVMOrcMaterializationResponsibilityGetInitializerSymbol(self.as_inner()) };
        (!ptr.is_null()).then(|| unsafe { SymbolStringPoolEntry::from_inner(ptr) })
    }

    /// Returns the names of any symbols covered by this MaterializationResponsibility object that
    /// have queries pending.
    ///
    /// This information can be used to return responsibility for unrequested symbols back to the
    /// JITDylib via the delegate method.
    pub fn get_requested_symbols(&self) -> &[SymbolStringPoolEntry] {
        let mut len = MaybeUninit::uninit();
        let ptr = unsafe {
            LLVMOrcMaterializationResponsibilityGetRequestedSymbols(
                self.as_inner(),
                len.as_mut_ptr(),
            )
        };
        unsafe { std::slice::from_raw_parts(ptr.cast(), len.assume_init()) }
    }

    /// Notifies the target JITDylib that the given symbols have been resolved.
    pub fn notify_resolved(&self, syms: &[SymbolFlagsMapPair]) -> Result<(), LLVMString> {
        cvt(unsafe {
            LLVMOrcMaterializationResponsibilityNotifyResolved(
                self.as_inner(),
                syms.as_ptr().cast_mut().cast(),
                syms.len(),
            )
        })
    }

    /// Notifies the target JITDylib (and any pending queries on that JITDylib)
    /// that all symbols covered by this MaterializationResponsibility instance
    /// have been emitted.
    pub fn notify_emitted(&self) -> Result<(), LLVMString> {
        cvt(unsafe { LLVMOrcMaterializationResponsibilityNotifyEmitted(self.as_inner()) })
    }

    /// Notify all not-yet-emitted covered by this MaterializationResponsibility instance that an
    /// error has occurred.
    ///
    /// This will remove all symbols covered by this MaterializationResponsibility from the target
    /// JITDylib, and send an error to any queries waiting on these symbols.
    pub fn fail_materialization(&self) {
        unsafe { LLVMOrcMaterializationResponsibilityFailMaterialization(self.as_inner()) };
    }

    /// Transfers responsibility to the given MaterializationUnit for all symbols defined by that
    /// MaterializationUnit.
    ///
    /// This allows materializers to break up work based on run-time information (e.g.
    /// by introspecting which symbols have actually been looked up and
    /// materializing only those).
    pub fn replace(&self, mu: MaterializationUnit) -> Result<(), LLVMString> {
        // TODO: mem::forget?
        cvt(unsafe { LLVMOrcMaterializationResponsibilityReplace(self.as_inner(), mu.as_inner()) })
    }

    /// Delegates responsibility for the given symbols to the returned
    /// materialization responsibility. Useful for breaking up work between
    /// threads, or different kinds of materialization processes.
    ///
    /// The caller retains responsibility of the the passed
    /// MaterializationResponsibility.
    pub fn delegate(
        &self,
        syms: &[SymbolStringPoolEntry],
    ) -> Result<MaterializationResponsibility, LLVMString> {
        let mut res = MaybeUninit::uninit();
        cvt(unsafe {
            LLVMOrcMaterializationResponsibilityDelegate(
                self.as_inner(),
                syms.as_ptr().cast_mut().cast(),
                syms.len(),
                res.as_mut_ptr(),
            )
        })?;
        Ok(unsafe { MaterializationResponsibility::from_inner(res.assume_init()) })
    }
}

/// A resource tracker.
///
/// ResourceTrackers allow you to remove code.
pub struct ResourceTracker {
    rt: LLVMOrcResourceTrackerRef,
}

impl ResourceTracker {
    /// Wraps a raw pointer.
    pub unsafe fn from_inner(rt: LLVMOrcResourceTrackerRef) -> Self {
        Self { rt }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcResourceTrackerRef {
        self.rt
    }

    /// Remove all resources associated with this tracker.
    pub fn remove(&self) -> Result<(), LLVMString> {
        cvt(unsafe { LLVMOrcResourceTrackerRemove(self.as_inner()) })
    }

    /// Transfers tracking of all resources associated with this resource tracker to the given
    /// resource tracker.
    pub fn transfer_to(&self, rt: &ResourceTracker) {
        unsafe { LLVMOrcResourceTrackerTransferTo(self.as_inner(), rt.as_inner()) };
    }
}

impl Drop for ResourceTracker {
    fn drop(&mut self) {
        unsafe { LLVMOrcReleaseResourceTracker(self.as_inner()) };
    }
}

/// A JIT execution session reference.
///
/// Returned by [`LLJIT::get_execution_session`].
///
/// ExecutionSession represents the JIT'd program and provides context for the JIT: It contains the
/// JITDylibs, error reporting mechanisms, and dispatches the materializers.
///
/// See [the ORCv2 docs](https://releases.llvm.org/17.0.1/docs/ORCv2.html).
pub struct ExecutionSessionRef {
    es: LLVMOrcExecutionSessionRef,
}

impl ExecutionSessionRef {
    /// Wraps a raw pointer.
    pub unsafe fn from_inner(es: LLVMOrcExecutionSessionRef) -> Self {
        Self { es }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcExecutionSessionRef {
        self.es
    }

    /// Intern a string in the ExecutionSession's SymbolStringPool and return a reference to it.
    pub fn intern(&self, name: &CStr) -> SymbolStringPoolEntry {
        unsafe {
            SymbolStringPoolEntry::from_inner(LLVMOrcExecutionSessionIntern(
                self.as_inner(),
                name.as_ptr(),
            ))
        }
    }

    /// Returns the JITDylib with the given name, if any.
    pub fn get_dylib_by_name(&self, name: &CStr) -> Option<JITDylibRef> {
        unsafe {
            let dylib = LLVMOrcExecutionSessionGetJITDylibByName(self.as_inner(), name.as_ptr());
            (!dylib.is_null()).then(|| JITDylibRef::from_inner(dylib))
        }
    }

    /// Create a "bare" JITDylib.
    ///
    /// The client is responsible for ensuring that the JITDylib's name is unique.
    pub fn create_bare_jit_dylib(&self, name: &CStr) -> JITDylibRef {
        unsafe {
            JITDylibRef::from_inner(LLVMOrcExecutionSessionCreateBareJITDylib(
                self.as_inner(),
                name.as_ptr(),
            ))
        }
    }

    /// Create a JITDylib.
    ///
    /// The client is responsible for ensuring that the JITDylib's name is unique.
    pub fn create_jit_dylib(&self, name: &CStr) -> Result<JITDylibRef, LLVMString> {
        let mut res = MaybeUninit::uninit();
        cvt(unsafe {
            LLVMOrcExecutionSessionCreateJITDylib(self.as_inner(), res.as_mut_ptr(), name.as_ptr())
        })?;
        Ok(unsafe { JITDylibRef::from_inner(res.assume_init()) })
    }

    /// Sets the default error reporter to the ExecutionSession.
    ///
    /// Uses [`tracing::error!`] to log the error message.
    pub fn set_default_error_reporter(&self) {
        self.set_error_reporter(&mut |msg| error!(msg = %msg.to_string_lossy(), "LLVM error"))
    }

    /// Attach a custom error reporter function to the ExecutionSession.
    pub fn set_error_reporter(&self, mut f: &mut dyn FnMut(&CStr)) {
        extern "C" fn shim(ctx: *mut c_void, err: LLVMErrorRef) {
            let f = ctx.cast::<&mut dyn FnMut(&CStr)>();
            let Err(e) = cvt(err) else { return };
            let res = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe { (*f)(&e) }));
            match res {
                Ok(()) => {}
                Err(e) => {
                    error!(msg=?panic_payload(&e), "error reporter closure panicked");
                }
            }
        }

        let ctx = &mut f as *mut _ as *mut c_void;
        unsafe { LLVMOrcExecutionSessionSetErrorReporter(self.as_inner(), shim, ctx) };
    }
}

/// A JIT dynamic library reference.
///
/// JITDylibs provide the symbol tables.
pub struct JITDylibRef {
    dylib: LLVMOrcJITDylibRef,
}

impl JITDylibRef {
    /// Wraps a raw pointer.
    pub unsafe fn from_inner(dylib: LLVMOrcJITDylibRef) -> Self {
        Self { dylib }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcJITDylibRef {
        self.dylib
    }

    /// Return a reference to a newly created resource tracker associated with JD.
    pub fn create_resource_tracker(&self) -> ResourceTracker {
        unsafe {
            ResourceTracker::from_inner(LLVMOrcJITDylibCreateResourceTracker(self.as_inner()))
        }
    }

    /// Return a reference to the default resource tracker for the given JITDylib.
    pub fn get_default_resource_tracker(&self) -> ResourceTracker {
        unsafe {
            ResourceTracker::from_inner(LLVMOrcJITDylibGetDefaultResourceTracker(self.as_inner()))
        }
    }

    /// Add the given MaterializationUnit to the given JITDylib.
    pub fn define(&self, mu: MaterializationUnit) -> Result<(), (LLVMString, MaterializationUnit)> {
        // If this operation succeeds then JITDylib JD will take ownership of MU.
        // If the operation fails then ownership remains with the caller who should call
        // LLVMOrcDisposeMaterializationUnit to destroy it.
        cvt(unsafe { LLVMOrcJITDylibDefine(self.as_inner(), mu.as_inner()) }).map_err(|e| (e, mu))
    }

    /// Calls remove on all trackers associated with this JITDylib.
    pub fn clear(&self) -> Result<(), LLVMString> {
        cvt(unsafe { LLVMOrcJITDylibClear(self.as_inner()) })
    }

    /*
    /// Add a DefinitionGenerator to the given JITDylib.
    pub fn add_generator(&self, dg: DefinitionGenerator) {
        unsafe { LLVMOrcJITDylibAddGenerator(self.as_inner(), dg.as_inner()) };
        // The JITDylib will take ownership of the given generator: The client is no longer responsible for managing its memory.
        mem::forget(dg);
    }
    */
}

/// [`LLVMOrcJITTargetMachineBuilderRef`], used in [`LLJITBuilder`].
pub struct JITTargetMachineBuilder {
    builder: LLVMOrcJITTargetMachineBuilderRef,
}

impl JITTargetMachineBuilder {
    /// Create a JITTargetMachineBuilder from the given TargetMachine template.
    pub fn new(tm: TargetMachine) -> Self {
        let builder =
            unsafe { LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(tm.as_mut_ptr()) };
        mem::forget(tm);
        Self { builder }
    }

    /// Create a JITTargetMachineBuilder by detecting the host.
    pub fn detect_host() -> Result<JITTargetMachineBuilder, LLVMString> {
        let mut res = MaybeUninit::uninit();
        cvt(unsafe { LLVMOrcJITTargetMachineBuilderDetectHost(res.as_mut_ptr()) })?;
        Ok(Self { builder: unsafe { res.assume_init() } })
    }

    /// Wraps a raw pointer.
    pub unsafe fn from_inner(builder: LLVMOrcJITTargetMachineBuilderRef) -> Self {
        Self { builder }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcJITTargetMachineBuilderRef {
        self.builder
    }

    /// Returns the target triple for the given JITTargetMachineBuilder as a string.
    pub fn get_target_triple(&self) -> LLVMString {
        unsafe { llvm_string(LLVMOrcJITTargetMachineBuilderGetTargetTriple(self.as_inner())) }
    }

    /// Sets the target triple for the given JITTargetMachineBuilder to the given string.
    pub fn set_target_triple(&self, triple: &CStr) {
        unsafe { LLVMOrcJITTargetMachineBuilderSetTargetTriple(self.as_inner(), triple.as_ptr()) }
    }
}

impl Drop for JITTargetMachineBuilder {
    fn drop(&mut self) {
        unsafe { LLVMOrcDisposeJITTargetMachineBuilder(self.builder) }
    }
}

/// Lazily-initialized [`LLJIT`] builder.
#[must_use]
pub struct LLJITBuilder {
    builder: lljit::LLVMOrcLLJITBuilderRef,
}

impl LLJITBuilder {
    /// Creates a new default LLJIT builder.
    pub fn new() -> Self {
        Self { builder: ptr::null_mut() }
    }

    /// Wraps a raw pointer.
    pub unsafe fn from_inner(builder: LLVMOrcLLJITBuilderRef) -> Self {
        Self { builder }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcLLJITBuilderRef {
        self.builder
    }

    fn as_inner_init(&mut self) -> LLVMOrcLLJITBuilderRef {
        if self.builder.is_null() {
            self.builder = unsafe { LLVMOrcCreateLLJITBuilder() };
        }
        self.builder
    }

    /// Set the target machine builder by creating it from the given template.
    pub fn set_target_machine(self, tm: TargetMachine) -> Self {
        self.set_target_machine_builder(JITTargetMachineBuilder::new(tm))
    }

    /// Set the target machine builder by detecting the host.
    pub fn set_target_machine_from_host(self) -> Result<Self, LLVMString> {
        JITTargetMachineBuilder::detect_host().map(|jtmb| self.set_target_machine_builder(jtmb))
    }

    /// Set the target machine builder.
    pub fn set_target_machine_builder(mut self, jtmb: JITTargetMachineBuilder) -> Self {
        unsafe {
            LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(self.as_inner_init(), jtmb.as_inner())
        };
        mem::forget(jtmb);
        self
    }

    /*
    pub fn set_object_linking_layer_creator(self, mut f: &mut dyn FnMut(*const (), &CStr)) -> Self {
        extern "C" fn shim(
            ctx: *mut c_void,
            es: LLVMOrcExecutionSessionRef,
            triple: *const c_char,
        ) -> LLVMOrcObjectLayerRef {
            let f = ctx.cast::<&mut dyn FnMut(*const (), &CStr)>();
            let name = unsafe { CStr::from_ptr(triple) };
            unsafe { (*f)(es.cast(), name) };
            ptr::null_mut()
        }

        let ctx = &mut f as *mut _ as *mut c_void;
        unsafe { LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(self.as_inner_init(), shim, ctx) };
        self
    }
    */

    /// Builds the JIT.
    pub fn build(self) -> Result<LLJIT, LLVMString> {
        // This operation takes ownership of the Builder argument: clients should not
        // dispose of the builder after calling this function (even if the function
        // returns an error).
        let mut res = MaybeUninit::uninit();
        let r = cvt(unsafe { LLVMOrcCreateLLJIT(res.as_mut_ptr(), self.as_inner()) });
        mem::forget(self);
        r.map(|()| unsafe { LLJIT::from_inner(res.assume_init()) })
    }
}

impl Drop for LLJITBuilder {
    fn drop(&mut self) {
        unsafe { LLVMOrcDisposeLLJITBuilder(self.builder) };
    }
}

/// An ORC JIT.
///
/// Manages the memory of all JIT'd code and all modules that are transferred to it.
///
/// See [the ORCv2 docs](https://releases.llvm.org/17.0.1/docs/ORCv2.html).
pub struct LLJIT {
    jit: LLVMOrcLLJITRef,
}

impl LLJIT {
    /// Creates a new LLJIT builder.
    pub fn builder() -> LLJITBuilder {
        LLJITBuilder::new()
    }

    /// Creates a new ORC JIT with a target machine for the host.
    pub fn new() -> Result<Self, LLVMString> {
        LLJITBuilder::new().set_target_machine_from_host()?.build()
    }

    /// Creates a new default ORC JIT.
    pub fn new_empty() -> Result<Self, LLVMString> {
        LLJITBuilder::new().build()
    }

    /// Wraps a raw pointer.
    pub unsafe fn from_inner(jit: LLVMOrcLLJITRef) -> Self {
        Self { jit }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcLLJITRef {
        self.jit
    }

    /// Return the target triple for this LLJIT instance.
    pub fn get_triple_string(&self) -> &CStr {
        // This string is owned by the LLJIT instance and does not need to be freed by the caller.
        unsafe { CStr::from_ptr(LLVMOrcLLJITGetTripleString(self.jit)) }
    }

    /// Return the data layout for this LLJIT instance.
    pub fn get_data_layout_string(&self) -> &CStr {
        // This string is owned by the LLJIT instance and does not need to be freed by the caller.
        unsafe { CStr::from_ptr(LLVMOrcLLJITGetDataLayoutStr(self.jit)) }
    }

    /// Returns the global prefix character according to the LLJIT's DataLayout.
    pub fn get_global_prefix(&self) -> c_char {
        unsafe { LLVMOrcLLJITGetGlobalPrefix(self.jit) }
    }

    /// Add an IR module to the main JITDylib.
    pub fn add_module(&self, tsm: ThreadSafeModule) -> Result<(), LLVMString> {
        let jd = self.get_main_jit_dylib();
        self.add_module_with_dylib(tsm, jd)
    }

    /// Add an IR module to the given JITDylib.
    pub fn add_module_with_dylib(
        &self,
        tsm: ThreadSafeModule,
        jd: JITDylibRef,
    ) -> Result<(), LLVMString> {
        let res = cvt(unsafe {
            LLVMOrcLLJITAddLLVMIRModule(self.as_inner(), jd.as_inner(), tsm.as_inner())
        });
        mem::forget(tsm);
        res
    }

    /// Add an IR module to the given ResourceTracker's JITDylib.
    pub fn add_module_with_rt(
        &self,
        tsm: ThreadSafeModule,
        jd: ResourceTracker,
    ) -> Result<(), LLVMString> {
        let res = cvt(unsafe {
            LLVMOrcLLJITAddLLVMIRModuleWithRT(self.as_inner(), jd.as_inner(), tsm.as_inner())
        });
        mem::forget(tsm);
        res
    }

    /// Gets the execution session.
    pub fn get_execution_session(&self) -> ExecutionSessionRef {
        unsafe { ExecutionSessionRef::from_inner(LLVMOrcLLJITGetExecutionSession(self.as_inner())) }
    }

    /// Return a reference to the Main JITDylib.
    pub fn get_main_jit_dylib(&self) -> JITDylibRef {
        unsafe { JITDylibRef::from_inner(LLVMOrcLLJITGetMainJITDylib(self.as_inner())) }
    }

    /// Mangles the given string according to the LLJIT instance's DataLayout, then interns the
    /// result in the SymbolStringPool and returns a reference to the pool entry.
    pub fn mangle_and_intern(&self, unmangled_name: &CStr) -> SymbolStringPoolEntry {
        unsafe {
            SymbolStringPoolEntry::from_inner(LLVMOrcLLJITMangleAndIntern(
                self.as_inner(),
                unmangled_name.as_ptr(),
            ))
        }
    }

    /// Look up the given symbol in the main JITDylib of the given LLJIT instance.
    pub fn lookup(&self, name: &CStr) -> Result<usize, LLVMString> {
        self.lookup_unmangled(&self.mangle_and_intern(name))
    }

    /// Look up the given symbol in the main JITDylib of the given LLJIT instance.
    ///
    /// The name should be mangled.
    pub fn lookup_unmangled(&self, unmangled_name: &CStr) -> Result<usize, LLVMString> {
        let mut res = MaybeUninit::uninit();
        cvt(unsafe {
            LLVMOrcLLJITLookup(self.as_inner(), res.as_mut_ptr(), unmangled_name.as_ptr())
        })?;
        Ok(unsafe { res.assume_init() }.try_into().unwrap())
    }

    /// Returns a non-owning reference to the LLJIT instance's IR transform layer.
    pub fn get_ir_transform_layer(&self) -> IRTransformLayerRef {
        unsafe { IRTransformLayerRef::from_inner(LLVMOrcLLJITGetIRTransformLayer(self.as_inner())) }
    }

    // get_*_layer...

    // Experimental interface for `libLLVMOrcDebugging.a`.
    /*
    /// Install the plugin that submits debug objects to the executor.
    /// Executors must expose the llvm_orc_registerJITLoaderGDBWrapper symbol.
    pub fn enable_debug_support(&self) -> Result<(), LLVMString> {
        cvt(unsafe { LLVMOrcLLJITEnableDebugSupport(self.as_inner()) })
    }
    */
}

impl Drop for LLJIT {
    fn drop(&mut self) {
        if let Err(e) = cvt(unsafe { LLVMOrcDisposeLLJIT(self.jit) }) {
            error!("Failed to dispose JIT: {e}");
        }
    }
}

/*
pub struct ObjectLayerRef {
    ptr: LLVMOrcObjectLayerRef,
}

impl ObjectLayerRef {
    /// Wraps a raw pointer.
    pub unsafe fn from_inner(ptr: LLVMOrcObjectLayerRef) -> Self {
        Self { ptr }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcObjectLayerRef {
        self.ptr
    }
}

pub struct ObjectTransformLayerRef {
    ptr: LLVMOrcObjectTransformLayerRef,
}

impl ObjectTransformLayerRef {
    /// Wraps a raw pointer.
    pub unsafe fn from_inner(ptr: LLVMOrcObjectTransformLayerRef) -> Self {
        Self { ptr }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcObjectTransformLayerRef {
        self.ptr
    }
}
*/

/// A reference to an IR transform layer.
pub struct IRTransformLayerRef {
    ptr: LLVMOrcIRTransformLayerRef,
}

impl IRTransformLayerRef {
    /// Wraps a raw pointer.
    pub unsafe fn from_inner(ptr: LLVMOrcIRTransformLayerRef) -> Self {
        Self { ptr }
    }

    /// Unwraps the raw pointer.
    pub fn as_inner(&self) -> LLVMOrcIRTransformLayerRef {
        self.ptr
    }

    /// IDK.
    pub fn emit(&self, mr: &MaterializationResponsibility, tsm: &ThreadSafeModule) {
        unsafe { LLVMOrcIRTransformLayerEmit(self.as_inner(), mr.as_inner(), tsm.as_inner()) };
    }

    /// Set the transform function of this transform layer.
    pub fn set_transform(&self, mut f: &mut dyn FnMut(&ThreadSafeModule) -> Result<(), String>) {
        extern "C" fn shim(
            ctx: *mut c_void,
            m: *mut LLVMOrcThreadSafeModuleRef,
            _mr: LLVMOrcMaterializationResponsibilityRef,
        ) -> LLVMErrorRef {
            let f = ctx.cast::<&mut dyn FnMut(&ThreadSafeModule) -> Result<(), String>>();
            let m = unsafe { ThreadSafeModule::from_inner(*m) };
            let res = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe { (*f)(&m) }));
            mem::forget(m);
            cvt_cb_res(res)
        }

        let ctx = &mut f as *mut _ as *mut c_void;
        unsafe { LLVMOrcIRTransformLayerSetTransform(self.as_inner(), shim, ctx) };
    }
}

fn cvt(ptr: LLVMErrorRef) -> Result<(), LLVMString> {
    if ptr.is_null() {
        Ok(())
    } else {
        Err(unsafe { llvm_string(LLVMGetErrorMessage(ptr)) })
    }
}

fn cvt_cb_res(res: Result<Result<(), String>, Box<dyn std::any::Any + Send>>) -> LLVMErrorRef {
    let msg = match res {
        Ok(Ok(())) => return ptr::null_mut(),
        Ok(Err(e)) => e,
        Err(e) => format!("callback panicked at {:?}", panic_payload(&e)),
    };
    unsafe { LLVMCreateStringError(CString::new(msg).unwrap_or_default().as_ptr()) }
}

fn panic_payload(any: &dyn std::any::Any) -> Option<&str> {
    if let Some(s) = any.downcast_ref::<&str>() {
        Some(*s)
    } else if let Some(s) = any.downcast_ref::<String>() {
        Some(s.as_str())
    } else {
        None
    }
}

unsafe fn llvm_string(ptr: *const c_char) -> LLVMString {
    // `LLVMString::new` is private
    std::mem::transmute(ptr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::{passes::PassBuilderOptions, targets::Target};

    #[test]
    #[ignore = "ci fails idk"]
    fn e2e() {
        let (tsm, tscx) = ThreadSafeModule::create("test");
        let fn_name = "my_fn";
        tsm.with_module(&mut |m| {
            let cx = tscx.get_context();
            let bcx = cx.create_builder();
            let ty = cx.i64_type().fn_type(&[], false);

            let f = m.add_function(fn_name, ty, Some(inkwell::module::Linkage::External));
            let bb = cx.append_basic_block(f, "entry");
            bcx.position_at_end(bb);
            bcx.build_int_compare(
                inkwell::IntPredicate::EQ,
                cx.i64_type().const_zero(),
                cx.i64_type().const_all_ones(),
                "a",
            )
            .unwrap();
            bcx.build_return(Some(&cx.i64_type().const_int(69, false))).unwrap();

            eprintln!("--before--");
            eprintln!("{}", m.print_to_string().to_string_lossy());

            m.verify().map_err(|e| e.to_string())?;

            Target::initialize_native(&Default::default()).unwrap();
            let triple = TargetMachine::get_default_triple();
            let cpu = TargetMachine::get_host_cpu_name();
            let features = TargetMachine::get_host_cpu_features();
            let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
            let machine = target
                .create_target_machine(
                    &triple,
                    &cpu.to_string_lossy(),
                    &features.to_string_lossy(),
                    Default::default(),
                    inkwell::targets::RelocMode::Default,
                    inkwell::targets::CodeModel::Default,
                )
                .ok_or_else(|| String::from("failed to create target machine"))?;

            m.run_passes("default<O3>", &machine, PassBuilderOptions::create()).unwrap();

            eprintln!("--after--");
            eprintln!("{}", m.print_to_string().to_string_lossy());

            Ok(())
        })
        .unwrap();

        let jit = LLJIT::new_empty().unwrap();
        jit.add_module(tsm).unwrap();
        let address = jit
            .lookup_unmangled(dbg!(&jit.mangle_and_intern(&CString::new(fn_name).unwrap())))
            .unwrap();
        eprintln!("address: {address:#x}");
        let f = unsafe { std::mem::transmute::<usize, extern "C" fn() -> u64>(address) };
        let r = f();
        assert_eq!(r, 69);
    }
}

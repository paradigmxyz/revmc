#include <llvm/IR/Attributes.h>
#include <llvm/IR/ConstantRangeList.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm-c/Core.h>
#include <llvm-c/Orc.h>
#include <llvm-c/LLJIT.h>

using namespace llvm;

extern "C" LLVMAttributeRef revmc_llvm_create_initializes_attr(LLVMContextRef C,
                                                                int64_t Lower,
                                                                int64_t Upper) {
    auto &Ctx = *unwrap(C);
    unsigned BitWidth = 64;
    ConstantRange CR(APInt(BitWidth, Lower, true), APInt(BitWidth, Upper, true));
    return wrap(Attribute::get(Ctx, Attribute::Initializes, ArrayRef(CR)));
}

/// Remove a JITDylib from the ExecutionSession.
extern "C" LLVMErrorRef revmc_llvm_execution_session_remove_jit_dylib(
    LLVMOrcExecutionSessionRef ES, LLVMOrcJITDylibRef JD) {
    auto *Session = reinterpret_cast<orc::ExecutionSession *>(ES);
    auto *Dylib = reinterpret_cast<orc::JITDylib *>(JD);
    return wrap(Session->removeJITDylib(*Dylib));
}

/// Add a JITDylib to the link order of another JITDylib.
extern "C" void revmc_llvm_jit_dylib_add_to_link_order(
    LLVMOrcJITDylibRef JD, LLVMOrcJITDylibRef Other) {
    auto *Dylib = reinterpret_cast<orc::JITDylib *>(JD);
    auto *OtherDylib = reinterpret_cast<orc::JITDylib *>(Other);
    Dylib->addToLinkOrder(*OtherDylib);
}

/// Enable concurrent compilation on an LLJIT builder so that multiple threads
/// can compile modules through the same LLJIT instance safely.
/// When enabled, LLJIT uses ConcurrentIRCompiler (a fresh TargetMachine per
/// compilation) instead of SimpleCompiler (single shared TargetMachine).
extern "C" void revmc_llvm_lljit_builder_set_support_concurrent_compilation(
    LLVMOrcLLJITBuilderRef Builder) {
    auto *B = reinterpret_cast<orc::LLJITBuilder *>(Builder);
    B->setSupportConcurrentCompilation(true);
}

/// Synchronous symbol lookup in a specific JITDylib.
/// `Name` is unmangled — LLJIT applies the data layout prefix internally.
extern "C" LLVMErrorRef revmc_llvm_lljit_lookup_in(LLVMOrcLLJITRef J,
                                                    LLVMOrcJITDylibRef JD,
                                                    LLVMOrcExecutorAddress *Result,
                                                    const char *Name) {
    auto *Jit = reinterpret_cast<orc::LLJIT *>(J);
    auto *Dylib = reinterpret_cast<orc::JITDylib *>(JD);
    auto Addr = Jit->lookup(*Dylib, Name);
    if (!Addr)
        return wrap(Addr.takeError());
    *Result = Addr->getValue();
    return LLVMErrorSuccess;
}

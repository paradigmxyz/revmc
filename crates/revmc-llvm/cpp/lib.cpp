#include <llvm-c/LLJIT.h>
#include <llvm-c/Orc.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Debugging/PerfSupportPlugin.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/IR/Attributes.h>

#include <llvm/Support/Debug.h>
#include <llvm/Support/circular_raw_ostream.h>

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
extern "C" LLVMErrorRef
revmc_llvm_execution_session_remove_jit_dylib(LLVMOrcExecutionSessionRef ES,
                                              LLVMOrcJITDylibRef JD) {
  auto *Session = reinterpret_cast<orc::ExecutionSession *>(ES);
  auto *Dylib = reinterpret_cast<orc::JITDylib *>(JD);
  return wrap(Session->removeJITDylib(*Dylib));
}

/// Add a JITDylib to the link order of another JITDylib.
extern "C" void
revmc_llvm_jit_dylib_add_to_link_order(LLVMOrcJITDylibRef JD,
                                       LLVMOrcJITDylibRef Other) {
  auto *Dylib = reinterpret_cast<orc::JITDylib *>(JD);
  auto *OtherDylib = reinterpret_cast<orc::JITDylib *>(Other);
  Dylib->addToLinkOrder(*OtherDylib);
}

/// Use ConcurrentIRCompiler (thread-safe, fresh TargetMachine per compilation)
/// while keeping the default InPlaceTaskDispatcher (no background threads).
///
/// setSupportConcurrentCompilation(true) would also switch the dispatcher to
/// DynamicThreadPoolTaskDispatcher, spawning background threads. We only want
/// the thread-safe compiler, not the thread pool.
extern "C" void revmc_llvm_lljit_builder_set_concurrent_compiler(
    LLVMOrcLLJITBuilderRef Builder) {
  auto *B = reinterpret_cast<orc::LLJITBuilder *>(Builder);
  B->setCompileFunctionCreator(
      [](orc::JITTargetMachineBuilder JTMB)
          -> Expected<std::unique_ptr<orc::IRCompileLayer::IRCompiler>> {
        return std::make_unique<orc::ConcurrentIRCompiler>(std::move(JTMB));
      });
}

/// Synchronous symbol lookup in a specific JITDylib.
/// `Name` is unmangled — LLJIT applies the data layout prefix internally.
extern "C" LLVMErrorRef
revmc_llvm_lljit_lookup_in(LLVMOrcLLJITRef J, LLVMOrcJITDylibRef JD,
                           LLVMOrcExecutorAddress *Result, const char *Name) {
  auto *Jit = reinterpret_cast<orc::LLJIT *>(J);
  auto *Dylib = reinterpret_cast<orc::JITDylib *>(JD);
  auto Addr = Jit->lookup(*Dylib, Name);
  if (!Addr)
    return wrap(Addr.takeError());
  *Result = Addr->getValue();
  return LLVMErrorSuccess;
}

/// Silence the unconditional `dbgs() <<` in PerfSupportPlugin's
/// writeUnwindRecord by redirecting the dbgs() stream to nulls().
/// In debug builds dbgs() is a circular_raw_ostream wrapping errs(), so we
/// swap its underlying stream. In release builds dbgs() IS errs() and we
/// can't redirect it, but this is only used with assertions-enabled LLVM.
// TODO: Remove once the upstream LLVM fix for the PerfSupportPlugin dbgs() log
// lands in a release.
static void suppress_dbgs() {
#ifndef NDEBUG
  static_cast<circular_raw_ostream &>(dbgs()).setStream(nulls());
#endif
}

/// Install PerfSupportPlugin on the LLJIT's ObjectLinkingLayer.
/// Writes perf jitdump records so `perf record -k 1` / `perf inject --jit`
/// can resolve JIT-compiled symbols with full debug info and unwind info.
/// Returns an error if the object layer is not JITLink-based.
extern "C" LLVMErrorRef
revmc_llvm_lljit_enable_perf_support(LLVMOrcLLJITRef J) {
  suppress_dbgs();

  auto *Jit = reinterpret_cast<orc::LLJIT *>(J);
  auto *OLL = dyn_cast<orc::ObjectLinkingLayer>(&Jit->getObjLinkingLayer());
  if (!OLL)
    return wrap(make_error<StringError>("PerfSupportPlugin requires JITLink",
                                        inconvertibleErrorCode()));
  auto &ES = Jit->getExecutionSession();
  auto &EPC = ES.getExecutorProcessControl();
  auto ProcessJD = Jit->getProcessSymbolsJITDylib();
  if (!ProcessJD)
    return wrap(make_error<StringError>(
        "PerfSupportPlugin requires process symbols JITDylib",
        inconvertibleErrorCode()));
  auto Plugin = orc::PerfSupportPlugin::Create(
      EPC, *ProcessJD, /*EmitDebugInfo=*/true, /*EmitUnwindInfo=*/true);
  if (!Plugin)
    return wrap(Plugin.takeError());
  OLL->addPlugin(std::move(*Plugin));
  return LLVMErrorSuccess;
}

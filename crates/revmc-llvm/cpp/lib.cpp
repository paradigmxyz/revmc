#include <llvm-c/LLJIT.h>
#include <llvm-c/Orc.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Debugging/PerfSupportPlugin.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Target/TargetMachine.h>

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

/// Opaque capture context shared between the Rust caller and the DualOutputCompiler.
/// The Rust side requests capture before a JIT lookup; the compiler checks the flag
/// during `operator()` and writes verbose assembly text into the buffer.
///
/// Thread-safety: LLJIT uses InPlaceTaskDispatcher, so compilation runs inline
/// on the calling thread. The request/read/clear cycle is single-threaded.
struct RevmcAsmCaptureCtx {
  bool requested = false;
  std::string buffer;
};

namespace {

/// Custom IRCompiler that optionally emits verbose assembly text alongside
/// object code.
///
/// When capture is requested, the compiler first runs
/// addPassesToEmitFile(AssemblyFile) to capture asm with LLVM's register
/// allocation and scheduling comments, then runs addPassesToEmitMC for the
/// object code (matching ORC's SimpleCompiler behavior).
///
/// When not requested, only the object path runs — identical to ConcurrentIRCompiler.
class DualOutputCompiler : public orc::IRCompileLayer::IRCompiler {
public:
  DualOutputCompiler(orc::JITTargetMachineBuilder JTMB,
                     RevmcAsmCaptureCtx *Capture)
      : IRCompiler(orc::irManglingOptionsFromTargetOptions(JTMB.getOptions())),
        JTMB(std::move(JTMB)), Capture(Capture) {}

  Expected<std::unique_ptr<MemoryBuffer>> operator()(Module &M) override {
    auto TM = JTMB.createTargetMachine();
    if (!TM)
      return TM.takeError();

    auto &TMRef = **TM;

    if (M.getDataLayout().isDefault())
      M.setDataLayout(TMRef.createDataLayout());

    // Emit assembly if capture was requested.
    if (Capture && Capture->requested) {
      TMRef.Options.MCOptions.AsmVerbose = true;

      SmallString<0> AsmBuf;
      raw_svector_ostream AsmOS(AsmBuf);
      legacy::PassManager PMAsm;

      if (!TMRef.addPassesToEmitFile(PMAsm, AsmOS, nullptr,
                                     CodeGenFileType::AssemblyFile)) {
        PMAsm.run(M);
        Capture->buffer = std::string(AsmBuf.str());
      }

      Capture->requested = false;
    }

    // Emit object code (matching ORC SimpleCompiler behavior).
    SmallVector<char, 0> ObjBuf;
    raw_svector_ostream ObjOS(ObjBuf);
    legacy::PassManager PMObj;
    MCContext *Ctx = nullptr;

    if (TMRef.addPassesToEmitMC(PMObj, Ctx, ObjOS))
      return make_error<StringError>("Target does not support MC emission",
                                     inconvertibleErrorCode());

    PMObj.run(M);

    auto ModId = M.getModuleIdentifier();
    return std::make_unique<SmallVectorMemoryBuffer>(
        std::move(ObjBuf), ModId + "-jitted-objectbuffer",
        /*RequiresNullTerminator=*/false);
  }

private:
  orc::JITTargetMachineBuilder JTMB;
  RevmcAsmCaptureCtx *Capture;
};

} // namespace

/// Install DualOutputCompiler (thread-safe, fresh TargetMachine per compilation,
/// optional assembly text capture) while keeping the default
/// InPlaceTaskDispatcher (no background threads).
///
/// The returned capture context pointer is valid for the lifetime of the LLJIT.
extern "C" RevmcAsmCaptureCtx *
revmc_llvm_lljit_builder_set_dual_compiler(LLVMOrcLLJITBuilderRef Builder) {
  auto *B = reinterpret_cast<orc::LLJITBuilder *>(Builder);
  // Leak the context — it lives as long as the process-global LLJIT.
  auto *Capture = new RevmcAsmCaptureCtx();
  B->setCompileFunctionCreator(
      [Capture](orc::JITTargetMachineBuilder JTMB)
          -> Expected<std::unique_ptr<orc::IRCompileLayer::IRCompiler>> {
        return std::make_unique<DualOutputCompiler>(std::move(JTMB), Capture);
      });
  return Capture;
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

/// Install PerfSupportPlugin on the LLJIT's ObjectLinkingLayer.
/// Writes perf jitdump records so `perf record -k 1` / `perf inject --jit`
/// can resolve JIT-compiled symbols with full debug info and unwind info.
/// Returns an error if the object layer is not JITLink-based.
extern "C" LLVMErrorRef
revmc_llvm_lljit_enable_perf_support(LLVMOrcLLJITRef J) {
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

extern "C" void revmc_llvm_asm_capture_request(RevmcAsmCaptureCtx *Ctx) {
  Ctx->requested = true;
  Ctx->buffer.clear();
}

extern "C" const char *revmc_llvm_asm_capture_data(const RevmcAsmCaptureCtx *Ctx,
                                                   size_t *Len) {
  *Len = Ctx->buffer.size();
  return Ctx->buffer.data();
}


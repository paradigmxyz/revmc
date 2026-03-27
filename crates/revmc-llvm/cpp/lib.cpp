#include <llvm-c/LLJIT.h>
#include <llvm-c/Orc.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Debugging/PerfSupportPlugin.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCObjectFileInfo.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCParser/MCAsmParser.h>
#include <llvm/MC/MCParser/MCTargetAsmParser.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
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

/// Write callback used by `revmc_llvm_emit_module` to deliver output into
/// caller-owned buffers (e.g. Rust `Vec<u8>`). Called once with the complete
/// output after each pipeline run.
using RevmcWriteFn = void (*)(const char *Data, size_t Len, void *Ctx);

/// Assemble `.s` text into object code using LLVM's MC layer.
/// Much cheaper than running the full backend (ISel/RA/scheduling) again.
static Expected<SmallVector<char, 0>>
assembleToObject(TargetMachine &TM, StringRef AsmText) {
  const auto &T = TM.getTarget();
  auto Triple = TM.getTargetTriple();

  std::string FeatStr(TM.getTargetFeatureString());
  std::unique_ptr<MCSubtargetInfo> STI(
      T.createMCSubtargetInfo(Triple.str(), TM.getTargetCPU(), FeatStr));
  if (!STI)
    return make_error<StringError>("failed to create MCSubtargetInfo",
                                   inconvertibleErrorCode());

  std::unique_ptr<MCRegisterInfo> MRI(T.createMCRegInfo(Triple.str()));
  MCTargetOptions MCOptions = TM.Options.MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(
      T.createMCAsmInfo(*MRI, Triple.str(), MCOptions));
  std::unique_ptr<MCInstrInfo> MCII(T.createMCInstrInfo());

  // Set up source manager with the assembly text.
  SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(AsmText), SMLoc());

  MCObjectFileInfo MOFI;
  MCContext Ctx(Triple, MAI.get(), MRI.get(), STI.get(), &SrcMgr);
  MOFI.initMCObjectFileInfo(Ctx, /*PIC=*/false);
  Ctx.setObjectFileInfo(&MOFI);

  // Create object streamer that writes to ObjBuf.
  SmallVector<char, 0> ObjBuf;
  raw_svector_ostream ObjOS(ObjBuf);

  std::unique_ptr<MCCodeEmitter> CE(T.createMCCodeEmitter(*MCII, Ctx));
  std::unique_ptr<MCAsmBackend> MAB(T.createMCAsmBackend(*STI, *MRI, MCOptions));
  std::unique_ptr<MCObjectWriter> OW = MAB->createObjectWriter(ObjOS);
  std::unique_ptr<MCStreamer> ObjStreamer(
      T.createMCObjectStreamer(Triple, Ctx, std::move(MAB), std::move(OW),
                               std::move(CE), *STI));

  // Parse and assemble.
  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(SrcMgr, Ctx, *ObjStreamer, *MAI));
  std::unique_ptr<MCTargetAsmParser> TAP(
      T.createMCAsmParser(*STI, *Parser, *MCII, MCOptions));
  if (!TAP)
    return make_error<StringError>("failed to create target asm parser",
                                   inconvertibleErrorCode());
  Parser->setTargetParser(*TAP);

  if (Parser->Run(/*NoInitialTextSection=*/false))
    return make_error<StringError>("assembly parsing failed",
                                   inconvertibleErrorCode());

  // Finalize — flushes fixups and writes the object file.
  ObjStreamer->finish();

  return std::move(ObjBuf);
}

/// Emit object code from a module, optionally emitting verbose assembly text.
///
/// Output is delivered through caller-provided write callbacks, each invoked
/// once with the complete output.
///
/// `ObjWrite`/`ObjCtx` receive the object code bytes.
/// If `AsmWrite` is non-null, verbose assembly (with register allocation
/// comments) is also emitted via `AsmWrite`/`AsmCtx`.
///
/// When both are requested, the expensive backend (ISel/RA/scheduling) runs
/// only once to produce verbose `.s`, which is then assembled to `.o` via
/// LLVM's MC layer — much cheaper than running the full backend twice.
///
/// Returns null on success, or an error message string (caller must free).
extern "C" char *
revmc_llvm_emit_module(TargetMachine *TM, LLVMModuleRef MRef,
                       RevmcWriteFn ObjWrite, void *ObjCtx,
                       RevmcWriteFn AsmWrite, void *AsmCtx) {
  auto *M = unwrap(MRef);

  // When both asm and object are requested, run the backend once (asm), then
  // assemble the text to object code via the MC layer.
  if (AsmWrite) {
    TM->Options.MCOptions.AsmVerbose = true;

    SmallString<0> AsmBuf;
    raw_svector_ostream AsmOS(AsmBuf);
    legacy::PassManager PM;

    if (TM->addPassesToEmitFile(PM, AsmOS, nullptr,
                                CodeGenFileType::AssemblyFile)) {
      return strdup("target does not support assembly emission");
    }
    PM.run(*M);
    AsmWrite(AsmBuf.data(), AsmBuf.size(), AsmCtx);

    // Assemble .s → .o without rerunning the backend.
    auto ObjBuf = assembleToObject(*TM, AsmBuf.str());
    if (!ObjBuf) {
      return strdup(toString(ObjBuf.takeError()).c_str());
    }
    ObjWrite(ObjBuf->data(), ObjBuf->size(), ObjCtx);
    return nullptr;
  }

  // Object-only fast path: direct MC emission.
  {
    SmallVector<char, 0> ObjBuf;
    raw_svector_ostream ObjOS(ObjBuf);
    legacy::PassManager PM;
    MCContext *Ctx = nullptr;

    if (TM->addPassesToEmitMC(PM, Ctx, ObjOS)) {
      return strdup("target does not support MC emission");
    }
    PM.run(*M);
    ObjWrite(ObjBuf.data(), ObjBuf.size(), ObjCtx);
  }

  return nullptr;
}


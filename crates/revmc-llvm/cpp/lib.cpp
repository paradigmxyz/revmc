#include <llvm/IR/Attributes.h>
#include <llvm/IR/ConstantRangeList.h>
#include <llvm-c/Core.h>

using namespace llvm;

extern "C" LLVMAttributeRef revmc_llvm_create_initializes_attr(LLVMContextRef C,
                                                                int64_t Lower,
                                                                int64_t Upper) {
    auto &Ctx = *unwrap(C);
    unsigned BitWidth = 64;
    ConstantRange CR(APInt(BitWidth, Lower, true), APInt(BitWidth, Upper, true));
    return wrap(Attribute::get(Ctx, Attribute::Initializes, ArrayRef(CR)));
}

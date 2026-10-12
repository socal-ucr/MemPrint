// MemPrintResize: routes every heap allocation through libmpr with a stable site id, so that
// MemPrint's per-site reservations can be applied to it (resize/runtime/mpr.h).
//
//   MPR_PASS_MODE=apply MPR_PASS_RESERVATIONS=FILE MPR_PASS_SITEMAP=FILE \
//       clang -O2 -g -fpass-plugin=libMemPrintResize.so ...  -L... -lmpr
//   opt -load-pass-plugin=libMemPrintResize.so -passes=mpr-resize [-mpr-mode=...] ...
// (clang parses -mllvm options before it loads a -fpass-plugin, hence the environment variables)
//
// The site id hashes the call's debug location chain (file, line, column, and every inlinedAt
// above it), so an allocation inside an inlined std::vector growth resolves to the program's own
// call site. It runs at the end of the optimisation pipeline, after inlining. Without debug
// information the id hashes the function's name and the call's position in it.
//
// Modes: profile (the default) only routes the calls; MemPrint's Pin tool (-sites) then profiles
// the program per id. apply also embeds a reservation file (memprint.sites) in the binary as
// __mpr_embedded_reservations, which libmpr reads when MPR_RESERVATIONS is not set.
//
// std::vector growth: in profile and apply modes the result of every std::vector<...>::_M_check_len
// call (the new capacity of a growing vector) is passed through __mpr_capacity, which can raise it
// to a predicted final size.
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <vector>

using namespace llvm;

static cl::opt< std::string > Mode("mpr-mode", cl::init("profile"), cl::desc("MemPrintResize: profile | apply | off"));
static cl::opt< std::string > Reservations("mpr-reservations", cl::init(""),
                                           cl::desc("MemPrintResize apply: reservation file to embed"));
static cl::opt< std::string > SiteMap("mpr-sitemap", cl::init(""),
                                      cl::desc("MemPrintResize: append 'id location' lines to this file"));

namespace {

// a cl::opt, or the environment variable when the option was not given
std::string Setting(const cl::opt< std::string >& opt, const char* env)
{
    if (opt.getNumOccurrences()) return opt;
    const char* v = getenv(env);
    return v ? v : std::string(opt);
}

enum Kind { MPR_MALLOC = 1, MPR_CALLOC = 2, MPR_REALLOC = 4, MPR_MEMALIGN = 8, MPR_NEW = 16 };

// What a call to an allocation function becomes. sizeArg/alignArg: argument indices (-1: none).
struct Rule
{
    const char* entry; // __mpr_alloc, __mpr_calloc, __mpr_realloc or __mpr_posix_memalign
    int kind;
    int sizeArg, alignArg;
};
const std::map< std::string, Rule > Rules = {
    {"malloc", {"__mpr_alloc", MPR_MALLOC, 0, -1}},
    {"valloc", {"__mpr_alloc", MPR_MEMALIGN, 0, -1}},
    {"aligned_alloc", {"__mpr_alloc", MPR_MEMALIGN, 1, 0}},
    {"memalign", {"__mpr_alloc", MPR_MEMALIGN, 1, 0}},
    {"calloc", {"__mpr_calloc", MPR_CALLOC, -1, -1}},
    {"realloc", {"__mpr_realloc", MPR_REALLOC, -1, -1}},
    {"posix_memalign", {"__mpr_posix_memalign", MPR_MEMALIGN, -1, -1}},
    {"_Znwm", {"__mpr_alloc", MPR_NEW, 0, -1}},
    {"_Znam", {"__mpr_alloc", MPR_NEW, 0, -1}},
    {"_ZnwmSt11align_val_t", {"__mpr_alloc", MPR_NEW, 0, 1}},
    {"_ZnamSt11align_val_t", {"__mpr_alloc", MPR_NEW, 0, 1}},
    // nothrow: NULL on failure, as malloc
    {"_ZnwmRKSt9nothrow_t", {"__mpr_alloc", MPR_MALLOC, 0, -1}},
    {"_ZnamRKSt9nothrow_t", {"__mpr_alloc", MPR_MALLOC, 0, -1}},
};

uint64_t fnv(uint64_t h, StringRef s)
{
    for (unsigned char c : s)
        h = (h ^ c) * 0x100000001b3ULL;
    return h;
}

uint64_t fnvInt(uint64_t h, uint64_t v)
{
    for (int i = 0; i < 8; i++, v >>= 8)
        h = (h ^ (v & 0xff)) * 0x100000001b3ULL;
    return h;
}

// The id and a readable location of a call: its debug location chain, innermost first.
std::pair< uint64_t, std::string > SiteOf(CallBase& call, StringRef callee, unsigned position)
{
    // the enclosing function's (mangled) name: inlined library code (std::allocator::allocate) has the
    // same location for every element type and caller
    uint64_t h = fnv(fnv(0xcbf29ce484222325ULL, callee), call.getFunction()->getName());
    std::string where;
    raw_string_ostream os(where);
    if (DILocation* loc = call.getDebugLoc().get())
    {
        for (DILocation* l = loc; l; l = l->getInlinedAt())
        {
            StringRef file = l->getFilename();
            h = fnv(h, file), h = fnvInt(h, l->getLine()), h = fnvInt(h, l->getColumn());
            StringRef fn = l->getScope() ? l->getScope()->getSubprogram()->getName() : "";
            os << (l == loc ? "" : " < ") << file << ":" << l->getLine() << ":" << l->getColumn() << " " << fn;
        }
    }
    else
    {
        StringRef fn = call.getFunction()->getName();
        h = fnv(h, fn), h = fnvInt(h, position);
        os << fn << " #" << position << " (no debug location)";
    }
    return {h & 0x7fffffffffffffffULL, os.str()};
}

struct MemPrintResize : PassInfoMixin< MemPrintResize >
{
    // early: at the start of the pipeline, before inlining, only the std::vector growth points
    // (_M_check_len is still a call there); late: at its end, the allocation calls
    bool early = false;
    explicit MemPrintResize(bool early = false) : early(early) {}

    PreservedAnalyses run(Module& M, ModuleAnalysisManager&)
    {
        std::string mode = Setting(Mode, "MPR_PASS_MODE"), reservations = Setting(Reservations, "MPR_PASS_RESERVATIONS");
        std::string siteMap = Setting(SiteMap, "MPR_PASS_SITEMAP");
        if (mode == "off") return PreservedAnalyses::all();
        LLVMContext& C = M.getContext();
        Type* I64      = Type::getInt64Ty(C);
        Type* I32      = Type::getInt32Ty(C);
        Type* Ptr      = PointerType::getUnqual(C);
        Type* SizeT    = I64;
        FunctionCallee alloc = M.getOrInsertFunction("__mpr_alloc", Ptr, SizeT, I64, I32, SizeT);
        FunctionCallee calloc = M.getOrInsertFunction("__mpr_calloc", Ptr, SizeT, SizeT, I64);
        FunctionCallee realloc = M.getOrInsertFunction("__mpr_realloc", Ptr, Ptr, SizeT, I64);
        FunctionCallee posix = M.getOrInsertFunction("__mpr_posix_memalign", I32, Ptr, SizeT, SizeT, I64);
        FunctionCallee capacity = M.getOrInsertFunction("__mpr_capacity", SizeT, SizeT, SizeT, SizeT, I64);

        std::string mapText;
        raw_string_ostream map(mapText);
        unsigned rewritten = 0, grown = 0;
        for (Function& F : M)
        {
            if (F.isDeclaration() || F.getName().starts_with("__mpr_")) continue;
            std::vector< CallBase* > calls;
            for (BasicBlock& BB : F)
                for (Instruction& I : BB)
                    if (auto* call = dyn_cast< CallBase >(&I))
                        if (Function* callee = call->getCalledFunction())
                            if (early ? callee->getName().contains("_M_check_len") : Rules.count(callee->getName().str()) > 0)
                                calls.push_back(call);
            unsigned position = 0;
            for (CallBase* call : calls)
            {
                position++;
                StringRef name = call->getCalledFunction()->getName();
                auto site      = SiteOf(*call, name, position);
                IRBuilder<> B(call);
                Value* id = ConstantInt::get(I64, site.first);
                Value* repl = nullptr;
                if (name.contains("_M_check_len"))
                {
                    // the growing vector's new capacity (elements); the element size is not known
                    // here, so it is passed as 1 and the runtime's model is in elements
                    if (!call->getType()->isIntegerTy(64)) continue;
                    B.SetInsertPoint(call->getNextNode());
                    Value* len = B.CreateCall(capacity, {call, ConstantInt::get(SizeT, 0), ConstantInt::get(SizeT, 1), id});
                    call->replaceUsesWithIf(len, [&](Use& u) { return u.getUser() != len; });
                    grown++;
                    map << "0x" << Twine::utohexstr(site.first) << " capacity " << site.second << "\n";
                    continue;
                }
                const Rule& r = Rules.at(name.str());
                auto arg = [&](int i) { return B.CreateZExtOrTrunc(call->getArgOperand(i), SizeT); };
                if (StringRef(r.entry) == "__mpr_alloc")
                    repl = B.CreateCall(alloc, {arg(r.sizeArg), id, ConstantInt::get(I32, r.kind),
                                                r.alignArg >= 0 ? arg(r.alignArg) : ConstantInt::get(SizeT, 0)});
                else if (StringRef(r.entry) == "__mpr_calloc")
                    repl = B.CreateCall(calloc, {arg(0), arg(1), id});
                else if (StringRef(r.entry) == "__mpr_realloc")
                    repl = B.CreateCall(realloc, {call->getArgOperand(0), arg(1), id});
                else
                    repl = B.CreateCall(posix, {call->getArgOperand(0), arg(1), arg(2), id});
                if (auto* invoke = dyn_cast< InvokeInst >(call))
                {
                    // operator new under an invoke: the runtime does not throw, so a plain call
                    // followed by the normal destination
                    B.CreateBr(invoke->getNormalDest());
                    invoke->getUnwindDest()->removePredecessor(invoke->getParent());
                }
                if (repl->getType() != call->getType()) repl = B.CreateBitOrPointerCast(repl, call->getType());
                repl->takeName(call);
                call->replaceAllUsesWith(repl);
                call->eraseFromParent();
                rewritten++;
                map << "0x" << Twine::utohexstr(site.first) << " " << name << " " << site.second << "\n";
            }
        }

        if (!early && mode == "apply" && !reservations.empty())
        {
            auto buf = MemoryBuffer::getFile(reservations);
            if (!buf)
                C.emitError("MemPrintResize: cannot read " + reservations);
            else if (!M.getGlobalVariable("__mpr_embedded_reservations"))
            {
                Constant* text = ConstantDataArray::getString(C, (*buf)->getBuffer(), true);
                auto* gv       = new GlobalVariable(M, text->getType(), true, GlobalValue::WeakAnyLinkage, text,
                                                    "__mpr_embedded_reservations");
                (void)gv;
            }
        }
        if (!siteMap.empty() && (rewritten || grown))
        {
            std::error_code ec;
            raw_fd_ostream out(siteMap, ec, sys::fs::OF_Append | sys::fs::OF_Text);
            if (!ec) out << map.str();
        }
        if (rewritten || grown)
            errs() << "MemPrintResize: " << M.getSourceFileName() << ": " << rewritten << " allocation calls, " << grown
                   << " vector growth points\n";
        return (rewritten || grown) ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo()
{
    return {LLVM_PLUGIN_API_VERSION, "MemPrintResize", "0.1", [](PassBuilder& PB) {
                PB.registerPipelineParsingCallback([](StringRef name, ModulePassManager& MPM,
                                                      ArrayRef< PassBuilder::PipelineElement >) {
                    if (name == "mpr-resize-growth")
                        MPM.addPass(MemPrintResize(true));
                    else if (name == "mpr-resize")
                        MPM.addPass(MemPrintResize());
                    else
                        return false;
                    return true;
                });
                PB.registerPipelineStartEPCallback(
                    [](ModulePassManager& MPM, OptimizationLevel) { MPM.addPass(MemPrintResize(true)); });
                PB.registerOptimizerLastEPCallback(
                    [](ModulePassManager& MPM, OptimizationLevel, ThinOrFullLTOPhase) { MPM.addPass(MemPrintResize()); });
            }};
}

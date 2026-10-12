// MemPrintResize: routes every heap allocation through libmpr with a stable site id, so that
// MemPrint's per-site reservations can be applied to it (resize/runtime/mpr.h).
//
//   MPR_PASS_MODE=apply MPR_PASS_RESERVATIONS=FILE MPR_PASS_SITEMAP=FILE \
//       clang -O2 -g -fpass-plugin=libMemPrintResize.so ...  -L... -lmpr
//   opt -load-pass-plugin=libMemPrintResize.so -passes=mpr-resize [-mpr-mode=...] ...
// (clang parses -mllvm options before it loads a -fpass-plugin, hence the environment variables)
//
// Site ids. A site id hashes the call's debug location chain (every inlinedAt level), with each
// level's line taken relative to the first line of the function it is in, as sample-based profiles
// do: inserting lines elsewhere in the file, or above the function, leaves the id unchanged, and
// files are named without their directory, so the id does not depend on where it is built. It runs
// at the end of the optimisation pipeline, after inlining, so an allocation inside inlined STL code
// resolves to the program's own call site. Without debug information the id hashes the function's
// name and the call's position in it.
//
// Modes: profile (the default) routes the calls; MemPrint's Pin tool (-sites) then profiles the
// program per id. apply also embeds a reservation file in the binary as
// __mpr_embedded_reservations, which libmpr reads when MPR_RESERVATIONS is not set.
//
// What the pass rewrites:
//   allocation calls      malloc, calloc, realloc, aligned allocation, operator new -> __mpr_*
//   address-taken uses    an allocation function stored or passed as a pointer -> a thunk with its own id
//   zeroing               memset(p, 0, n) / bzero / llvm.memset with a zero value, n >= 4 KB or not a
//                         constant -> __mpr_zero (libmpr leaves untouched pages of its blocks alone)
//   growth points         (early, before inlining) a container's new capacity -> __mpr_capacity:
//                         std::vector (libstdc++ _M_check_len, libc++ __recommend), std::string
//                         (libstdc++ _M_create, libc++ __recommend), hash tables (_M_rehash's bucket count)
//   growth contexts       calls from the program's own functions to container members that may grow
//                         them are bracketed by __mpr_growth(id) / __mpr_growth(0), so that a capacity
//                         is predicted per call site of the program, not per template instantiation
#include "llvm/ADT/SmallString.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <map>
#include <set>
#include <string>
#include <vector>

using namespace llvm;

static cl::opt< std::string > Mode("mpr-mode", cl::init("profile"), cl::desc("MemPrintResize: profile | apply | off"));
static cl::opt< std::string > Reservations("mpr-reservations", cl::init(""),
                                           cl::desc("MemPrintResize apply: reservation file to embed"));
static cl::opt< std::string > SiteMap("mpr-sitemap", cl::init(""),
                                      cl::desc("MemPrintResize: append 'id kind location' lines to this file"));

namespace {

// a cl::opt, or the environment variable when the option was not given
std::string Setting(const cl::opt< std::string >& opt, const char* env)
{
    if (opt.getNumOccurrences()) return opt;
    const char* v = getenv(env);
    return v ? v : std::string(opt);
}

enum Kind { MPR_MALLOC = 1, MPR_CALLOC = 2, MPR_REALLOC = 4, MPR_MEMALIGN = 8, MPR_NEW = 16 };
const uint64_t ZERO_MIN = 4096; // zeroing of fewer (constant) bytes stays as it is

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

// The id and a readable location of a call: its debug location chain, innermost first. Each level
// is hashed as (file, function, line - the function's first line, column).
std::pair< uint64_t, std::string > SiteOf(Instruction& call, StringRef callee, unsigned position)
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
            DISubprogram* sp = l->getScope() ? l->getScope()->getSubprogram() : nullptr;
            StringRef fn     = sp ? sp->getName() : "";
            unsigned first   = sp ? sp->getLine() : 0;
            unsigned offset  = l->getLine() >= first ? l->getLine() - first : l->getLine();
            // the file's base name: the id must not depend on the directory the program is built in
            StringRef file = l->getFilename();
            file           = file.substr(file.find_last_of('/') + 1);
            h = fnv(h, file), h = fnv(h, sp ? sp->getLinkageName() : ""), h = fnv(h, fn);
            h = fnvInt(h, offset), h = fnvInt(h, l->getColumn());
            os << (l == loc ? "" : " < ") << l->getFilename() << ":" << l->getLine() << ":" << l->getColumn() << " " << fn;
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

// A system header (the C++ library's own code): libstdc++, libc++, /usr/include
bool SystemFile(StringRef file, StringRef dir)
{
    std::string path = (dir.empty() || file.starts_with("/")) ? file.str() : (dir + "/" + file).str();
    return StringRef(path).contains("/include/c++/") || StringRef(path).starts_with("/usr/include/") ||
           StringRef(path).contains("/c++/v1/") || StringRef(path).starts_with("/usr/lib/gcc/");
}

// The program's own code: a function whose definition is not in a system header
bool UserFunction(Function& F)
{
    DISubprogram* sp = F.getSubprogram();
    return sp && sp->getFile() && !SystemFile(sp->getFilename(), sp->getDirectory());
}

// The size of a DIType in bytes, through typedefs and qualifiers (0: unknown)
uint64_t TypeSize(DIType* t)
{
    for (int i = 0; t && i < 16; i++)
    {
        if (t->getSizeInBits()) return t->getSizeInBits() / 8;
        auto* d = dyn_cast< DIDerivedType >(t);
        if (!d) return 0;
        t = d->getBaseType();
    }
    return 0;
}

// The element size of the container whose member F is: the size of the class's first template
// type parameter (std::vector<T>'s T, std::basic_string<CharT>'s CharT), from the debug information.
// 0 when it is not known (no debug information, or a class declared without its parameters).
uint64_t ElementSize(Function& F)
{
    DISubprogram* sp = F.getSubprogram();
    for (DISubprogram* s : {sp, sp ? sp->getDeclaration() : nullptr})
    {
        if (!s) continue;
        auto* cls = dyn_cast_or_null< DICompositeType >(s->getScope());
        if (!cls) continue;
        for (DITemplateParameter* p : cls->getTemplateParams())
            if (auto* tp = dyn_cast< DITemplateTypeParameter >(p))
                if (uint64_t n = TypeSize(tp->getType())) return n;
    }
    return 0;
}

// The base name and the whole demangled name of a mangled C++ function name
std::pair< std::string, std::string > Demangled(StringRef mangled)
{
    ItaniumPartialDemangler d;
    if (d.partialDemangle(mangled.str().c_str())) return {"", ""};
    std::pair< std::string, std::string > out;
    size_t n = 0;
    if (char* b = d.getFunctionBaseName(nullptr, &n)) out.first = b, free(b);
    out.second = demangle(mangled.str());
    return out;
}

// Containers whose growth goes through a growth point, and their members that may grow them
const char* ContainerPrefixes[] = {"std::vector<", "std::__cxx11::basic_string<", "std::basic_string<",
                                   "std::_Hashtable<", "std::unordered_map<", "std::unordered_set<",
                                   "std::__detail::_Map_base<", "std::__detail::_Insert_base<",
                                   "std::__detail::_Insert<", "std::__1::vector<", "std::__1::basic_string<",
                                   "std::__1::__hash_table<", "std::__1::unordered_map<", "std::__1::unordered_set<"};
// The containers' growth paths (out of line in libstdc++ and libc++): a call to one of them from the
// program's code is a container growing there. The public members (push_back, insert, operator[]) are
// left alone: they run on every call, and when they are inlined their growth path is called from the
// program's code anyway.
const std::set< std::string > GrowingMembers = {
    "_M_realloc_insert", "_M_realloc_append", "_M_default_append", "_M_fill_insert", "_M_range_insert",
    "_M_insert_aux", "_M_fill_assign", "_M_assign_aux", "_M_create", "_M_mutate", "_M_replace_aux",
    "_M_rehash", "_M_insert_unique_node", "_M_insert_multi_node", "__push_back_slow_path",
    "__emplace_back_slow_path", "__append", "__grow_by", "__grow_by_and_replace", "__rehash"};

bool GrowingCall(Function* callee)
{
    if (!callee || !callee->getName().starts_with("_Z")) return false;
    auto [base, cls] = Demangled(callee->getName());
    if (base.empty() || !GrowingMembers.count(base)) return false;
    for (const char* p : ContainerPrefixes)
        if (StringRef(cls).starts_with(p)) return true;
    return false;
}

// A growth point: a call whose result (or an argument) is a container's new capacity.
// arg: -1 the result; else the index of the argument that is the capacity (byRef: a pointer to it,
// read before the call and written back). fixedElem: the element size when it is not the class's
// template parameter (a hash table's bucket: a pointer).
struct GrowthPoint
{
    int arg;
    bool byRef;
    uint64_t fixedElem;
};

bool FindGrowthPoint(Function* callee, GrowthPoint* g)
{
    if (!callee) return false;
    StringRef n = callee->getName();
    if (n.contains("12_M_check_lenEmPKc") || (n.contains("11__recommendEm") && callee->getReturnType()->isIntegerTy(64)))
        return *g = {-1, false, 0}, true;   // libstdc++ vector, libc++ vector and string: the result
    if (n.contains("9_M_createERmm"))
        return *g = {1, true, 0}, true;     // libstdc++ string: _M_create(size_type& capacity, size_type old)
    if (n.starts_with("_ZNSt10_Hashtable") && n.contains("9_M_rehashEm"))
        return *g = {1, false, 8}, true;    // libstdc++ hash table: _M_rehash(size_type buckets, state)
    if (n.contains("__hash_table") && (n.contains("8__rehashILb") || n.contains("8__rehashEm")))
        return *g = {1, false, 8}, true;    // libc++ hash table: __rehash(size_type buckets)
    return false;
}

struct MemPrintResize : PassInfoMixin< MemPrintResize >
{
    // early: at the start of the pipeline, before inlining, only the growth points (the calls that
    // compute a new capacity are still calls there); late: at its end, everything else
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
        Type* Void     = Type::getVoidTy(C);
        FunctionCallee alloc = M.getOrInsertFunction("__mpr_alloc", Ptr, SizeT, I64, I32, SizeT);
        FunctionCallee calloc = M.getOrInsertFunction("__mpr_calloc", Ptr, SizeT, SizeT, I64);
        FunctionCallee realloc = M.getOrInsertFunction("__mpr_realloc", Ptr, Ptr, SizeT, I64);
        FunctionCallee posix = M.getOrInsertFunction("__mpr_posix_memalign", I32, Ptr, SizeT, SizeT, I64);
        FunctionCallee capacity = M.getOrInsertFunction("__mpr_capacity", SizeT, SizeT, SizeT, SizeT, I64);
        FunctionCallee zero = M.getOrInsertFunction("__mpr_zero", Void, Ptr, SizeT, I64);
        FunctionCallee growth = M.getOrInsertFunction("__mpr_growth", Void, I64);

        std::string mapText;
        raw_string_ostream map(mapText);
        unsigned rewritten = 0, grown = 0, contexts = 0, zeroed = 0, thunks = 0;
        auto note = [&](uint64_t id, StringRef kind, const std::string& where) {
            map << "0x" << Twine::utohexstr(id) << " " << kind << " " << where << "\n";
        };

        if (early)
        {
            for (Function& F : M)
            {
                if (F.isDeclaration() || F.getName().starts_with("__mpr_")) continue;
                std::vector< std::pair< CallBase*, GrowthPoint > > points;
                for (BasicBlock& BB : F)
                    for (Instruction& I : BB)
                        if (auto* call = dyn_cast< CallBase >(&I))
                        {
                            GrowthPoint g;
                            if (FindGrowthPoint(call->getCalledFunction(), &g)) points.push_back({call, g});
                        }
                unsigned position = 0;
                for (auto [call, g] : points)
                {
                    position++;
                    auto site     = SiteOf(*call, call->getCalledFunction()->getName(), position);
                    uint64_t elem = g.fixedElem ? g.fixedElem : ElementSize(F);
                    Value* id     = ConstantInt::get(I64, site.first);
                    Value* elemV  = ConstantInt::get(SizeT, elem ? elem : 1);
                    if (g.arg < 0)
                    {
                        // the new capacity is the result (elements; libmpr's models are in bytes)
                        if (!call->getType()->isIntegerTy(64) || isa< InvokeInst >(call)) continue;
                        IRBuilder<> B(call->getNextNode());
                        Value* len = B.CreateCall(capacity, {call, ConstantInt::get(SizeT, 0), elemV, id});
                        call->replaceUsesWithIf(len, [&](Use& u) { return u.getUser() != len; });
                    }
                    else if (g.byRef)
                    {
                        IRBuilder<> B(call);
                        Value* ref = call->getArgOperand(g.arg);
                        Value* len = B.CreateLoad(I64, ref);
                        B.CreateStore(B.CreateCall(capacity, {len, ConstantInt::get(SizeT, 0), elemV, id}), ref);
                    }
                    else
                    {
                        IRBuilder<> B(call);
                        Value* len = B.CreateZExtOrTrunc(call->getArgOperand(g.arg), SizeT);
                        Value* cap = B.CreateCall(capacity, {len, ConstantInt::get(SizeT, 0), elemV, id});
                        call->setArgOperand(g.arg, B.CreateZExtOrTrunc(cap, call->getArgOperand(g.arg)->getType()));
                    }
                    grown++;
                    note(site.first, ("capacity elem=" + Twine(elem)).str(), site.second);
                }
            }
        }
        else
        {
            // address-taken allocation functions: each use gets a thunk with its own site id (a
            // use in a constant, e.g. a table of function pointers, shares one per module)
            for (auto& rule : Rules)
            {
                Function* fn = M.getFunction(rule.first);
                if (!fn) continue;
                std::vector< Use* > uses;
                for (Use& u : fn->uses())
                {
                    auto* call = dyn_cast< CallBase >(u.getUser());
                    if (call && call->isCallee(&u)) continue;
                    uses.push_back(&u);
                }
                Function* shared = nullptr;
                for (Use* u : uses)
                {
                    auto* user = dyn_cast< Instruction >(u->getUser());
                    if (!user && isa< GlobalValue >(u->getUser()) == false && !isa< Constant >(u->getUser())) continue;
                    uint64_t id;
                    std::string where;
                    if (user)
                        std::tie(id, where) = SiteOf(*user, ("&" + rule.first), 0);
                    else
                        id = fnv(fnv(0xcbf29ce484222325ULL, M.getSourceFileName()), "&" + rule.first) & 0x7fffffffffffffffULL,
                        where = M.getSourceFileName() + " (a constant)";
                    if (!user && shared) continue;
                    std::string name = "__mpr_thunk_" + Twine::utohexstr(id).str();
                    Function* thunk  = M.getFunction(name);
                    if (!thunk)
                    {
                        thunk = Function::Create(fn->getFunctionType(), GlobalValue::InternalLinkage, name, M);
                        BasicBlock* body = BasicBlock::Create(C, "", thunk);
                        IRBuilder<> B(body);
                        std::vector< Value* > args;
                        for (Argument& a : thunk->args())
                            args.push_back(&a);
                        CallInst* inner = B.CreateCall(fn, args);
                        if (fn->getReturnType()->isVoidTy())
                            B.CreateRetVoid();
                        else
                            B.CreateRet(inner);
                        thunks++;
                        note(id, ("thunk " + rule.first), where);
                    }
                    if (user)
                        u->set(thunk);
                    else
                        shared = thunk;
                }
                // constant users (initializers): every remaining non-call use, at once
                if (shared)
                    fn->replaceUsesWithIf(shared, [&](Use& u) {
                        auto* call = dyn_cast< CallBase >(u.getUser());
                        return !(call && call->isCallee(&u)) && !isa< Instruction >(u.getUser()) &&
                               u.getUser() != shared;
                    });
            }

            for (Function& F : M)
            {
                if (F.isDeclaration() || F.getName().starts_with("__mpr_") && !F.getName().starts_with("__mpr_thunk_")) continue;
                std::vector< CallBase* > calls, growing, inlinedCaps;
                std::vector< Instruction* > zeroing;
                bool user = UserFunction(F);
                for (BasicBlock& BB : F)
                    for (Instruction& I : BB)
                    {
                        if (auto* ms = dyn_cast< MemSetInst >(&I))
                        {
                            auto* v   = dyn_cast< ConstantInt >(ms->getValue());
                            auto* len = dyn_cast< ConstantInt >(ms->getLength());
                            if (v && v->isZero() && !ms->isVolatile() && (!len || len->getZExtValue() >= ZERO_MIN))
                                zeroing.push_back(ms);
                            continue;
                        }
                        auto* call = dyn_cast< CallBase >(&I);
                        if (!call) continue;
                        Function* callee = call->getCalledFunction();
                        if (!callee) continue;
                        StringRef name = callee->getName();
                        if (Rules.count(name.str()))
                            calls.push_back(call);
                        else if (name == "__mpr_capacity" && call->getDebugLoc() && call->getDebugLoc()->getInlinedAt())
                            inlinedCaps.push_back(call);
                        else if ((name == "memset" || name == "bzero") && isa< CallInst >(call))
                        {
                            Value* len = call->getArgOperand(name == "memset" ? 2 : 1);
                            auto* v    = name == "memset" ? dyn_cast< ConstantInt >(call->getArgOperand(1)) : nullptr;
                            auto* n    = dyn_cast< ConstantInt >(len);
                            if ((name == "bzero" || (v && v->isZero())) && (!n || n->getZExtValue() >= ZERO_MIN))
                                zeroing.push_back(call);
                        }
                        else if (user && GrowingCall(callee))
                            growing.push_back(call);
                    }
                unsigned position = 0;
                for (CallBase* call : calls)
                {
                    position++;
                    StringRef name = call->getCalledFunction()->getName();
                    auto site      = SiteOf(*call, name, position);
                    IRBuilder<> B(call);
                    Value* id = ConstantInt::get(I64, site.first);
                    Value* repl = nullptr;
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
                    note(site.first, name, site.second);
                }
                for (Instruction* z : zeroing)
                {
                    position++;
                    auto site = SiteOf(*z, "zero", position);
                    IRBuilder<> B(z);
                    Value *dst, *len;
                    if (auto* ms = dyn_cast< MemSetInst >(z))
                        dst = ms->getDest(), len = ms->getLength();
                    else
                    {
                        auto* call = cast< CallBase >(z);
                        bool bz    = call->getCalledFunction()->getName() == "bzero";
                        dst = call->getArgOperand(0), len = call->getArgOperand(bz ? 1 : 2);
                    }
                    B.CreateCall(zero, {dst, B.CreateZExtOrTrunc(len, SizeT), ConstantInt::get(I64, site.first)});
                    if (!z->use_empty()) z->replaceAllUsesWith(dst); // memset returns its destination
                    z->eraseFromParent();
                    zeroed++;
                    note(site.first, "zero", site.second);
                }
                // a growth point inlined into its caller: the call site is known statically, from the
                // inlined location chain, so it becomes the growth point's id (no growth context needed)
                for (CallBase* call : inlinedCaps)
                {
                    position++;
                    auto site = SiteOf(*call, "capacity", position);
                    call->setArgOperand(3, ConstantInt::get(I64, site.first));
                    grown++;
                    note(site.first, "capacity-site", site.second);
                }
                for (CallBase* call : growing)
                {
                    position++;
                    auto site = SiteOf(*call, call->getCalledFunction()->getName(), position);
                    Value* id = ConstantInt::get(I64, site.first);
                    IRBuilder<>(call).CreateCall(growth, {id});
                    Value* none = ConstantInt::get(I64, 0);
                    if (auto* invoke = dyn_cast< InvokeInst >(call))
                    {
                        // cleared on both edges (the unwind destination's landing pad comes first)
                        BasicBlock* normal = invoke->getNormalDest();
                        if (!normal->getSinglePredecessor()) normal = SplitEdge(invoke->getParent(), normal);
                        IRBuilder<>(&*normal->getFirstInsertionPt()).CreateCall(growth, {none});
                        BasicBlock* unwind = invoke->getUnwindDest();
                        IRBuilder<>(&*unwind->getFirstInsertionPt()).CreateCall(growth, {none});
                    }
                    else
                        IRBuilder<>(call->getNextNode()).CreateCall(growth, {none});
                    contexts++;
                    note(site.first, "growth", site.second);
                }
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
                // one copy in the program however many modules carry it: a COMDAT the linker keeps once
                auto* gv = new GlobalVariable(M, text->getType(), true, GlobalValue::WeakODRLinkage, text,
                                              "__mpr_embedded_reservations");
                if (Triple(M.getTargetTriple()).supportsCOMDAT())
                    gv->setComdat(M.getOrInsertComdat("__mpr_embedded_reservations"));
            }
        }
        unsigned changed = rewritten + grown + contexts + zeroed + thunks;
        if (!siteMap.empty() && changed)
        {
            std::error_code ec;
            raw_fd_ostream out(siteMap, ec, sys::fs::OF_Append | sys::fs::OF_Text);
            if (!ec) out << map.str();
        }
        if (changed && getenv("MPR_PASS_VERBOSE"))
            errs() << "MemPrintResize: " << M.getSourceFileName() << ": " << rewritten << " allocation calls, " << thunks
                   << " thunks, " << zeroed << " zeroings, " << grown << " growth points, " << contexts
                   << " growth contexts\n";
        return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo()
{
    return {LLVM_PLUGIN_API_VERSION, "MemPrintResize", "0.2", [](PassBuilder& PB) {
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

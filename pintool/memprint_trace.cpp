/*
 * Copyright (C) 2009-2021 Intel Corporation.
 * SPDX-License-Identifier: MIT
 */

/*
 * MemPrint memory-footprint tracer.
 *
 * Records every memory operand (address, size) into a Pin trace buffer and,
 * when the buffer fills, folds the references into maps of address -> largest
 * access size.  The footprint of a set of references is the sum of the
 * largest size seen at each unique address.
 *
 * Two modes:
 *
 *   splitter  Instruments every memory reference.  Writes the exact
 *             footprint (SamplingInterval = 1) and, for every interval k in
 *             -intervals, -bins disjoint bins that each hold an independent
 *             1-in-k subsample of the references (one CSV per interval and bin).
 *
 *   sampler   Samples 1-in-i memory references at instrumentation time and
 *             Poisson-bootstraps every sampled reference into distinct bins
 *             out of -s (lambda = s / r), so each bin holds roughly a
 *             1-in-(i*r) subsample (exactly i / (E[min(Poisson(lambda), s)] / s)).
 *             Writes the union of the bins plus one CSV per bin.
 *
 * Optional:
 *
 *   -track_frees 1  free/realloc/munmap remove the released address range
 *                   from every footprint, so footprints are live memory.
 *   -snapshot N     every N memory references (time), append the current
 *                   footprints to a timeline CSV.
 *   -stop N         after N references, write all outputs and detach.
 *
 * Time counts memory references executed: one per reference in the
 * splitter, i per sampled reference in the sampler (an estimate).
 *
 * Output (in -outdir, which is created if missing):
 *   <Prefix>_<name>_<interval>_<pid>[_<args>].csv
 *   <Prefix>_<name>_<interval>_<pid>[_<args>]_SubSample_<binInterval>_bin_<j>.csv
 *   <Prefix>_<name>_<interval>_<pid>[_<args>]_timeline.csv        (-snapshot)
 * with Prefix = Buffered (splitter) or Sampled (sampler). Summary files have
 * the columns FunctionName,MemUsageObs,UniqueAddresses,CountObs,SamplingInterval;
 * the timeline has Time,SamplingInterval,Bin,MemUsageObs,UniqueAddresses,CountObs,FreedBytes,
 * Singletons,Doubletons. Bin -1 is the exact footprint in the splitter and the union of the
 * bins in the sampler; Bin -2 (splitter) is the union of one interval's bins, a
 * 1-in-(interval/bins) sample. Singletons/Doubletons count the addresses sampled
 * exactly once/twice, from which Chao1 estimates the addresses never sampled.
 */

#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <set>
#include <unordered_map>
#include <vector>
#include "pin.H"

using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::ostringstream;
using std::string;
using std::unordered_map;
using std::vector;

#define MAX_THREADS 64
#define NUM_BUF_PAGES 1024
#define PAGE_BITS 12

/* ===================================================================== */
/* Knobs                                                                 */
/* ===================================================================== */

KNOB< string > KnobMode(KNOB_MODE_WRITEONCE, "pintool", "mode", "splitter", "splitter | sampler");
KNOB< UINT32 > KnobSamplingInterval(KNOB_MODE_WRITEONCE, "pintool", "i", "100", "sampler: sample 1-in-i memory references");
KNOB< UINT32 > KnobNumSplits(KNOB_MODE_WRITEONCE, "pintool", "s", "100", "sampler: number of bootstrap bins");
KNOB< UINT32 > KnobSampleReplication(KNOB_MODE_WRITEONCE, "pintool", "r", "10", "sampler: replication factor (lambda = s/r)");
KNOB< string > KnobIntervals(KNOB_MODE_WRITEONCE, "pintool", "intervals",
                             "100,250,500,750,1000,2500,5000,7500,10000,25000,50000,75000,100000",
                             "splitter: comma-separated sampling intervals");
KNOB< UINT32 > KnobBins(KNOB_MODE_WRITEONCE, "pintool", "bins", "20", "splitter: bins per sampling interval");
KNOB< string > KnobOutDir(KNOB_MODE_WRITEONCE, "pintool", "outdir", "traces", "output directory");
KNOB< string > KnobName(KNOB_MODE_WRITEONCE, "pintool", "name", "",
                        "trace name, e.g. 2mm-MEDIUM (default: binary name, with the args appended after the pid)");
KNOB< string > KnobSeed(KNOB_MODE_WRITEONCE, "pintool", "seed", "", "RNG seed base (default: process id)");
KNOB< BOOL > KnobTrackFrees(KNOB_MODE_WRITEONCE, "pintool", "track_frees", "0",
                            "remove freed (free/realloc/munmap) address ranges from the footprints");
KNOB< UINT64 > KnobSnapshot(KNOB_MODE_WRITEONCE, "pintool", "snapshot", "0",
                            "write a timeline snapshot every N memory references (0: off)");
KNOB< UINT64 > KnobStop(KNOB_MODE_WRITEONCE, "pintool", "stop", "0",
                        "after N memory references, write the outputs and detach (0: run to the end)");

/* ===================================================================== */
/* Footprint                                                             */
/* ===================================================================== */

// Largest access size and number of samples per address, with running
// totals: bytes, and the number of addresses sampled exactly once (singletons)
// and twice (doubletons), which estimate how many addresses were never sampled
// (Chao1). When indexed, addresses are also grouped by page so that a freed
// range can be erased.
class Footprint
{
  public:
    UINT64 bytes = 0;      // sum of the largest access size per address
    UINT64 freed = 0;      // bytes removed by Erase
    UINT64 singletons = 0; // addresses sampled exactly once
    UINT64 doubletons = 0; // addresses sampled exactly twice

    VOID SetIndexed(BOOL on) { indexed = on; }
    size_t Unique() const { return entries.size(); }

    VOID Record(ADDRINT address, UINT32 size)
    {
        Entry& e = entries[address];
        if (e.count == 0 && indexed) pages[address >> PAGE_BITS].push_back(address);
        Count(e.count, -1);
        Count(++e.count, +1);
        if (e.size < size)
        {
            bytes += size - e.size;
            e.size = size;
        }
    }

    // Remove every address in [lo, hi).
    VOID Erase(ADDRINT lo, ADDRINT hi)
    {
        if (hi <= lo || pages.empty()) return;
        ADDRINT first = lo >> PAGE_BITS, last = (hi - 1) >> PAGE_BITS;
        if (last - first + 1 < pages.size())
        {
            for (ADDRINT page = first; page <= last; page++)
            {
                auto it = pages.find(page);
                if (it != pages.end()) ErasePage(it, lo, hi);
            }
        }
        else
        {
            for (auto it = pages.begin(); it != pages.end();)
            {
                auto next = std::next(it);
                if (it->first >= first && it->first <= last) ErasePage(it, lo, hi);
                it = next;
            }
        }
    }

  private:
    struct Entry
    {
        UINT32 size  = 0; // largest access size
        UINT32 count = 0; // times sampled
    };
    typedef unordered_map< ADDRINT, vector< ADDRINT > > PageIndex;

    unordered_map< ADDRINT, Entry > entries;
    PageIndex pages; // page -> addresses recorded in it (when indexed)
    BOOL indexed = FALSE;

    VOID Count(UINT32 count, INT32 delta)
    {
        if (count == 1) singletons += delta;
        if (count == 2) doubletons += delta;
    }

    VOID ErasePage(PageIndex::iterator page, ADDRINT lo, ADDRINT hi)
    {
        vector< ADDRINT >& addresses = page->second;
        size_t keep = 0;
        for (ADDRINT address : addresses)
        {
            if (address < lo || address >= hi)
            {
                addresses[keep++] = address;
                continue;
            }
            auto it = entries.find(address);
            if (it == entries.end()) continue;
            bytes -= it->second.size;
            freed += it->second.size;
            Count(it->second.count, -1);
            entries.erase(it);
        }
        addresses.resize(keep);
        if (keep == 0) pages.erase(page);
    }
};

/* ===================================================================== */
/* Global state                                                          */
/* ===================================================================== */

enum Mode { SPLITTER, SAMPLER };
Mode mode;

BUFFER_ID bufId;

struct MEMREF
{
    ADDRINT pc;
    ADDRINT ea;
    UINT32 size;
    BOOL read;
};

// A released address range, applied once the thread's buffer has been
// processed up to `position` (the fill pointer when the release happened).
struct Release
{
    VOID* position;
    ADDRINT lo, hi;
};

// Per-thread state; each slot is only touched by its own thread.
struct ThreadData
{
    UINT32 seed = 0;           // 32-bit LCG state
    vector< Release > pending; // releases not yet applied
    // Allocator call in progress (outermost call only)
    UINT32 allocDepth = 0;
    ADDRINT allocSp = 0; // stack pointer at the outermost call
    ADDRINT allocSize = 0, allocPtr = 0, allocOut = 0;
    ADDRINT munmapLo = 0, munmapHi = 0;
};
ThreadData threads[MAX_THREADS];

// Footprints shared by all threads; guarded by stateLock.
PIN_LOCK stateLock;
Footprint footprint;     // splitter: exact footprint; sampler: union of the bins
vector< Footprint > bins; // splitter: [interval * numBins + bin]; sampler: [bin]
vector< Footprint > unions; // splitter with -snapshot: union of each interval's bins
vector< UINT64 > binObservations;
UINT64 observations = 0; // references processed
UINT64 timeNow = 0;      // memory references executed (estimated in the sampler)
UINT64 nextSnapshot = 0;
UINT64 lastSnapshot = ~(UINT64)0; // time of the last snapshot written
BOOL finished = FALSE;   // outputs written (end of run or -stop)

// Live heap blocks (address -> size), for -track_frees.
PIN_LOCK allocLock;
unordered_map< ADDRINT, ADDRINT > allocations;

vector< UINT32 > intervals;          // splitter
vector< UINT64 > intervalThresholds; // splitter: accept a reference into interval k iff lcg < threshold[k]
UINT32 numBins;                      // bins per interval (splitter) or bootstrap bins (sampler)
double lambda;                       // sampler
UINT64 sampledBinInterval;           // sampler: effective sampling interval of one bootstrap bin

string outputFileNamePrefix;
ofstream outputFile;
ofstream timelineFile;

static const char* CSV_HEADER      = "FunctionName,MemUsageObs,UniqueAddresses,CountObs,SamplingInterval";
static const char* TIMELINE_HEADER =
    "Time,SamplingInterval,Bin,MemUsageObs,UniqueAddresses,CountObs,FreedBytes,Singletons,Doubletons";

/* ===================================================================== */
/* Random numbers                                                        */
/* ===================================================================== */

static inline UINT32 PIN_FAST_ANALYSIS_CALL lcg_step(UINT32& s)
{
    __asm__ __volatile__("imull $1664525, %[s]\n\t"
                         "add   $1013904223, %[s]\n\t"
                         : [s] "+r"(s)
                         :
                         : "cc");
    return s;
}

// Knuth's Poisson sampler on the thread's LCG stream.
UINT32 sample_poisson(double lambda, THREADID tid)
{
    UINT32 k = 0;
    double p = 1.0;
    double L = exp(-lambda);
    do
    {
        ++k;
        double u = (lcg_step(threads[tid].seed) & 0x7FFFFFFF) / static_cast< double >(0x80000000);
        p *= u;
    } while (p > L);
    return k - 1;
}

static inline ADDRINT PIN_FAST_ANALYSIS_CALL ShouldSample(THREADID tid, UINT32 rate)
{
    return ((lcg_step(threads[tid].seed) % rate) == 0) ? 1 : 0;
}

/* ===================================================================== */
/* Instrumentation                                                       */
/* ===================================================================== */

static VOID InsertRecord(INS ins, UINT32 memOp, UINT32 refSize, BOOL isRead)
{
    if (mode == SAMPLER)
    {
        INS_InsertIfCall(ins, IPOINT_BEFORE, AFUNPTR(ShouldSample), IARG_FAST_ANALYSIS_CALL, IARG_THREAD_ID, IARG_UINT32,
                         KnobSamplingInterval.Value(), IARG_END);
        INS_InsertFillBufferThen(ins, IPOINT_BEFORE, bufId, IARG_INST_PTR, offsetof(struct MEMREF, pc), IARG_MEMORYOP_EA, memOp,
                                 offsetof(struct MEMREF, ea), IARG_UINT32, refSize, offsetof(struct MEMREF, size), IARG_BOOL,
                                 isRead, offsetof(struct MEMREF, read), IARG_END);
    }
    else
    {
        INS_InsertFillBuffer(ins, IPOINT_BEFORE, bufId, IARG_INST_PTR, offsetof(struct MEMREF, pc), IARG_MEMORYOP_EA, memOp,
                             offsetof(struct MEMREF, ea), IARG_UINT32, refSize, offsetof(struct MEMREF, size), IARG_BOOL, isRead,
                             offsetof(struct MEMREF, read), IARG_END);
    }
}

VOID Trace(TRACE trace, VOID* v)
{
    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl))
    {
        for (INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins))
        {
            if (!INS_IsStandardMemop(ins) && !INS_HasMemoryVector(ins)) continue;

            UINT32 memoryOperands = INS_MemoryOperandCount(ins);
            for (UINT32 memOp = 0; memOp < memoryOperands; memOp++)
            {
                UINT32 refSize = INS_MemoryOperandSize(ins, memOp);
                // An operand that is both read and written is recorded once for each.
                if (INS_MemoryOperandIsRead(ins, memOp)) InsertRecord(ins, memOp, refSize, TRUE);
                if (INS_MemoryOperandIsWritten(ins, memOp)) InsertRecord(ins, memOp, refSize, FALSE);
            }
        }
    }
}

/* ===================================================================== */
/* Deallocation tracking (-track_frees)                                  */
/* ===================================================================== */

// Queue the release of [lo, hi) at the thread's current buffer position.
static VOID QueueRelease(THREADID tid, CONTEXT* ctxt, ADDRINT lo, ADDRINT hi)
{
    if (hi > lo) threads[tid].pending.push_back({PIN_GetBufferPointer(ctxt, bufId), lo, hi});
}

static ADDRINT ForgetAllocation(ADDRINT ptr)
{
    PIN_GetLock(&allocLock, 1);
    ADDRINT size = 0;
    auto it      = allocations.find(ptr);
    if (it != allocations.end())
    {
        size = it->second;
        allocations.erase(it);
    }
    PIN_ReleaseLock(&allocLock);
    return size;
}

static VOID RememberAllocation(ADDRINT ptr, ADDRINT size)
{
    if (ptr == 0) return;
    PIN_GetLock(&allocLock, 1);
    allocations[ptr] = size;
    PIN_ReleaseLock(&allocLock);
}

// Entry analysis: only the outermost allocator call of a thread is tracked,
// so allocator functions calling each other are not counted twice. A nested
// call has a lower stack pointer than the outermost one; an entry at the same
// or a higher one is a new outermost call (the previous one jumped back to
// its entry, or returned without its exit being seen).
static VOID AllocEnter(THREADID tid, ADDRINT sp, ADDRINT size, ADDRINT ptr, ADDRINT out)
{
    ThreadData& t = threads[tid];
    if (t.allocDepth > 0 && sp >= t.allocSp) t.allocDepth = 0;
    if (t.allocDepth++ > 0) return;
    t.allocSp   = sp;
    t.allocSize = size;
    t.allocPtr  = ptr;
    t.allocOut  = out;
}

static BOOL AllocLeave(THREADID tid)
{
    ThreadData& t = threads[tid];
    if (t.allocDepth == 0) return FALSE;
    return --t.allocDepth == 0;
}

// malloc(size), memalign(align, size), aligned_alloc(align, size), valloc(size)
static VOID MallocExit(THREADID tid, ADDRINT ret)
{
    if (AllocLeave(tid)) RememberAllocation(ret, threads[tid].allocSize);
}

// calloc(n, size)
static VOID CallocEnter(THREADID tid, ADDRINT sp, ADDRINT n, ADDRINT size) { AllocEnter(tid, sp, n * size, 0, 0); }

// posix_memalign(&out, align, size)
static VOID PosixMemalignExit(THREADID tid, ADDRINT ret)
{
    if (!AllocLeave(tid) || ret != 0) return;
    ADDRINT ptr = 0;
    PIN_SafeCopy(&ptr, (VOID*)threads[tid].allocOut, sizeof(ptr));
    RememberAllocation(ptr, threads[tid].allocSize);
}

// free(ptr): released at entry, because glibc's free leaves through a tail
// jump that Pin cannot see. Free's own bookkeeping writes into small blocks
// (16 bytes of tcache links) are therefore counted again.
static VOID FreeEnter(THREADID tid, CONTEXT* ctxt, ADDRINT ptr)
{
    ADDRINT size = ForgetAllocation(ptr);
    QueueRelease(tid, ctxt, ptr, ptr + size);
}

// realloc(ptr, size): releases the old block if it moved, else the tail it shrank by.
static VOID ReallocExit(THREADID tid, CONTEXT* ctxt, ADDRINT ret)
{
    if (!AllocLeave(tid)) return;
    ADDRINT ptr = threads[tid].allocPtr, size = threads[tid].allocSize;
    if (ptr == 0)
    {
        RememberAllocation(ret, size);
        return;
    }
    if (ret == 0 && size != 0) return; // failed: the old block is untouched
    ADDRINT oldSize = ForgetAllocation(ptr);
    if (ret == ptr)
        QueueRelease(tid, ctxt, ptr + size, ptr + oldSize);
    else
        QueueRelease(tid, ctxt, ptr, ptr + oldSize);
    RememberAllocation(ret, size);
}

static VOID InstrumentAllocator(IMG img, const char* name, AFUNPTR enter, IARGLIST enterArgs, AFUNPTR exit,
                                IARGLIST exitArgs)
{
    static std::set< ADDRINT > instrumented; // aliases (e.g. memalign, aligned_alloc) share an address
    RTN rtn = RTN_FindByName(img, name);
    // Skip PLT stubs: they jump to the real function and never return.
    if (!RTN_Valid(rtn) || SEC_Name(RTN_Sec(rtn)).find(".plt") == 0) return;
    if (!instrumented.insert(RTN_Address(rtn)).second) return;
    RTN_Open(rtn);
    RTN_InsertCall(rtn, IPOINT_BEFORE, enter, IARG_THREAD_ID, IARG_REG_VALUE, REG_STACK_PTR, IARG_IARGLIST, enterArgs,
                   IARG_END);
    RTN_InsertCall(rtn, IPOINT_AFTER, exit, IARG_THREAD_ID, IARG_IARGLIST, exitArgs, IARG_END);
    RTN_Close(rtn);
}

static IARGLIST Args(std::initializer_list< std::pair< IARG_TYPE, UINT32 > > args)
{
    IARGLIST list = IARGLIST_Alloc();
    for (const auto& arg : args)
    {
        if (arg.first == IARG_FUNCARG_ENTRYPOINT_VALUE)
            IARGLIST_AddArguments(list, IARG_FUNCARG_ENTRYPOINT_VALUE, arg.second, IARG_END);
        else
            IARGLIST_AddArguments(list, arg.first, IARG_END);
    }
    return list;
}

// Constant arguments for AllocEnter: (size arg, ptr arg, out arg) with -1 meaning "pass 0".
static IARGLIST EnterArgs(INT32 sizeArg, INT32 ptrArg, INT32 outArg)
{
    IARGLIST list = IARGLIST_Alloc();
    for (INT32 arg : {sizeArg, ptrArg, outArg})
    {
        if (arg < 0)
            IARGLIST_AddArguments(list, IARG_ADDRINT, (ADDRINT)0, IARG_END);
        else
            IARGLIST_AddArguments(list, IARG_FUNCARG_ENTRYPOINT_VALUE, (UINT32)arg, IARG_END);
    }
    return list;
}

VOID ImageLoad(IMG img, VOID* v)
{
    IARGLIST ret = Args({{IARG_FUNCRET_EXITPOINT_VALUE, 0}});
    IARGLIST ctxtRet = Args({{IARG_CONTEXT, 0}, {IARG_FUNCRET_EXITPOINT_VALUE, 0}});

    InstrumentAllocator(img, "malloc", AFUNPTR(AllocEnter), EnterArgs(0, -1, -1), AFUNPTR(MallocExit), ret);
    InstrumentAllocator(img, "valloc", AFUNPTR(AllocEnter), EnterArgs(0, -1, -1), AFUNPTR(MallocExit), ret);
    InstrumentAllocator(img, "memalign", AFUNPTR(AllocEnter), EnterArgs(1, -1, -1), AFUNPTR(MallocExit), ret);
    InstrumentAllocator(img, "aligned_alloc", AFUNPTR(AllocEnter), EnterArgs(1, -1, -1), AFUNPTR(MallocExit), ret);
    InstrumentAllocator(img, "calloc", AFUNPTR(CallocEnter),
                        Args({{IARG_FUNCARG_ENTRYPOINT_VALUE, 0}, {IARG_FUNCARG_ENTRYPOINT_VALUE, 1}}), AFUNPTR(MallocExit),
                        ret);
    InstrumentAllocator(img, "posix_memalign", AFUNPTR(AllocEnter), EnterArgs(2, -1, 0), AFUNPTR(PosixMemalignExit), ret);
    InstrumentAllocator(img, "realloc", AFUNPTR(AllocEnter), EnterArgs(1, 0, -1), AFUNPTR(ReallocExit), ctxtRet);

    RTN free = RTN_FindByName(img, "free");
    if (RTN_Valid(free) && SEC_Name(RTN_Sec(free)).find(".plt") != 0)
    {
        RTN_Open(free);
        RTN_InsertCall(free, IPOINT_BEFORE, AFUNPTR(FreeEnter), IARG_THREAD_ID, IARG_CONTEXT, IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                       IARG_END);
        RTN_Close(free);
    }
}

// munmap(addr, len) is caught at the system call, whoever makes it.
VOID SyscallEntry(THREADID tid, CONTEXT* ctxt, SYSCALL_STANDARD std, VOID* v)
{
    ThreadData& t = threads[tid];
    t.munmapLo = t.munmapHi = 0;
    if (PIN_GetSyscallNumber(ctxt, std) != SYS_munmap) return;
    ADDRINT lo  = PIN_GetSyscallArgument(ctxt, std, 0);
    ADDRINT len = PIN_GetSyscallArgument(ctxt, std, 1);
    t.munmapLo  = lo;
    t.munmapHi  = (lo + len + 4095) & ~(ADDRINT)4095; // whole pages are unmapped
}

VOID SyscallExit(THREADID tid, CONTEXT* ctxt, SYSCALL_STANDARD std, VOID* v)
{
    ThreadData& t = threads[tid];
    if (t.munmapHi > t.munmapLo && PIN_GetSyscallReturn(ctxt, std) == 0) QueueRelease(tid, ctxt, t.munmapLo, t.munmapHi);
    t.munmapLo = t.munmapHi = 0;
}

/* ===================================================================== */
/* Analysis                                                              */
/* ===================================================================== */

static VOID SplitReference(THREADID tid, ADDRINT address, UINT32 size)
{
    footprint.Record(address, size);

    // Keep the reference with probability bins/interval and put it in a uniform bin,
    // so every bin is a 1-in-interval subsample.
    UINT32& seed = threads[tid].seed;
    for (UINT32 k = 0; k < intervals.size(); k++)
    {
        if (lcg_step(seed) < intervalThresholds[k])
        {
            UINT32 bin = (lcg_step(seed) / 256) % numBins;
            bins[k * numBins + bin].Record(address, size);
            ++binObservations[k * numBins + bin];
            if (!unions.empty()) unions[k].Record(address, size);
        }
    }
}

static VOID BootstrapReference(THREADID tid, ADDRINT address, UINT32 size)
{
    UINT32 times = sample_poisson(lambda, tid);
    if (times == 0) return;
    if (times > numBins) times = numBins; // cannot place in more distinct bins than exist
    footprint.Record(address, size);

    // Place the reference in `times` distinct bins.
    vector< bool > used(numBins, false);
    UINT32 count = 0;
    while (count < times)
    {
        UINT32 bin = lcg_step(threads[tid].seed) % numBins;
        if (used[bin]) continue;
        used[bin] = true;
        count++;
        bins[bin].Record(address, size);
        ++binObservations[bin];
    }
}

static VOID ApplyRelease(ADDRINT lo, ADDRINT hi)
{
    footprint.Erase(lo, hi);
    for (Footprint& bin : bins)
        bin.Erase(lo, hi);
    for (Footprint& u : unions)
        u.Erase(lo, hi);
}

static UINT64 MainInterval() { return (mode == SPLITTER) ? 1 : KnobSamplingInterval.Value(); }

static UINT64 BinInterval(UINT32 mapIndex) { return (mode == SPLITTER) ? intervals[mapIndex / numBins] : sampledBinInterval; }

static VOID WriteSnapshot()
{
    if (timeNow == lastSnapshot) return;
    lastSnapshot = timeNow;
    // Hand-rolled formatting: streams and snprintf in Pin's C runtime cost
    // 10-30 us per row, which dominated runs with frequent snapshots.
    string text;
    text.reserve((bins.size() + 1) * 64);
    auto num = [&](UINT64 v) {
        char digits[24];
        int n = 0;
        do
        {
            digits[n++] = char('0' + v % 10);
            v /= 10;
        } while (v);
        while (n) text += digits[--n];
    };
    auto add = [&](UINT64 interval, INT64 bin, const Footprint& f, UINT64 count) {
        num(timeNow), text += ',', num(interval), text += ',';
        if (bin < 0)
            text += '-', num((UINT64)-bin);
        else
            num((UINT64)bin);
        text += ',', num(f.bytes), text += ',', num(f.Unique()), text += ',', num(count), text += ',', num(f.freed);
        text += ',', num(f.singletons), text += ',', num(f.doubletons), text += '\n';
    };
    add(MainInterval(), -1, footprint, observations);
    for (UINT32 k = 0; k < unions.size(); k++)
    {
        UINT64 count = 0;
        for (UINT32 j = 0; j < numBins; j++)
            count += binObservations[k * numBins + j];
        add(intervals[k] / numBins, -2, unions[k], count);
    }
    for (UINT32 m = 0; m < bins.size(); m++)
        add(BinInterval(m), m % numBins, bins[m], binObservations[m]);
    timelineFile.write(text.data(), text.size());
}

static VOID WriteOutputs();

// Advance time by one processed reference; returns FALSE once -stop is reached.
static BOOL Tick()
{
    timeNow += MainInterval();
    if (KnobSnapshot.Value() && timeNow >= nextSnapshot)
    {
        WriteSnapshot();
        while (nextSnapshot <= timeNow)
            nextSnapshot += KnobSnapshot.Value();
    }
    if (KnobStop.Value() && timeNow >= KnobStop.Value())
    {
        WriteOutputs();
        PIN_Detach();
        return FALSE;
    }
    return TRUE;
}

VOID* BufferFull(BUFFER_ID id, THREADID tid, const CONTEXT* ctxt, VOID* buf, UINT64 numElements, VOID* v)
{
    if (!buf) return buf;
    vector< Release >& pending = threads[tid].pending;
    size_t nextRelease         = 0;

    PIN_GetLock(&stateLock, tid + 1);
    struct MEMREF* reference = (struct MEMREF*)buf;
    for (UINT64 i = 0; i < numElements && !finished; i++, reference++)
    {
        for (; nextRelease < pending.size() && pending[nextRelease].position <= (VOID*)reference; nextRelease++)
            ApplyRelease(pending[nextRelease].lo, pending[nextRelease].hi);

        ADDRINT address = reference->ea;
        UINT32 size     = reference->size;
        if (address == 0) continue;

        ++observations;
        if (mode == SPLITTER)
            SplitReference(tid, address, size);
        else
            BootstrapReference(tid, address, size);
        if (!Tick()) break;
    }
    for (; nextRelease < pending.size() && !finished; nextRelease++)
        ApplyRelease(pending[nextRelease].lo, pending[nextRelease].hi);
    pending.clear();
    PIN_ReleaseLock(&stateLock);
    return buf;
}

VOID ThreadStart(THREADID tid, CONTEXT* ctxt, INT32 flags, VOID* v)
{
    if (tid >= MAX_THREADS)
    {
        cerr << "memprint_trace: thread id " << tid << " exceeds MAX_THREADS (" << MAX_THREADS << ")" << endl;
        PIN_ExitProcess(1);
    }
    UINT32 base       = KnobSeed.Value().empty() ? PIN_GetPid() : (UINT32)strtoul(KnobSeed.Value().c_str(), NULL, 10);
    threads[tid].seed = base * 31 + tid * 19;
}

/* ===================================================================== */
/* Output                                                                */
/* ===================================================================== */

static VOID WriteTotal(ofstream& out, UINT64 bytes, size_t uniqueAddresses, UINT64 observations, UINT64 interval)
{
    out << "Total," << bytes << "," << uniqueAddresses << "," << observations << "," << interval << endl;
}

// Default trace name: binary basename, with its arguments (joined by '_') as suffix.
static VOID DefaultName(string& binaryName, string& argSuffix)
{
    ifstream cmdline("/proc/self/cmdline");
    string arg;
    while (std::getline(cmdline, arg, '\0'))
    {
        if (binaryName.empty())
            binaryName = arg;
        else
            argSuffix += "_" + arg;
    }
    size_t lastSlash = binaryName.find_last_of('/');
    if (lastSlash != string::npos) binaryName = binaryName.substr(lastSlash + 1);
}

// mkdir -p
static VOID MakeDirs(const string& path)
{
    for (size_t slash = path.find('/', 1); slash != string::npos; slash = path.find('/', slash + 1))
        mkdir(path.substr(0, slash).c_str(), 0775);
    mkdir(path.c_str(), 0775);
}

BOOL InitOutputFiles()
{
    string outDir = KnobOutDir.Value();
    MakeDirs(outDir);

    string name = KnobName.Value(), argSuffix;
    if (name.empty()) DefaultName(name, argSuffix);
    ostringstream prefix;
    prefix << outDir << "/" << (mode == SPLITTER ? "Buffered_" : "Sampled_") << name << "_" << MainInterval() << "_"
           << PIN_GetPid() << argSuffix;
    outputFileNamePrefix = prefix.str();

    string filename = outputFileNamePrefix + ".csv";
    outputFile.open(filename);
    if (!outputFile.is_open())
    {
        cerr << "memprint_trace: cannot write " << filename << endl;
        return FALSE;
    }
    outputFile << CSV_HEADER << endl;

    if (KnobSnapshot.Value())
    {
        timelineFile.open(outputFileNamePrefix + "_timeline.csv");
        timelineFile << TIMELINE_HEADER << "\n";
        nextSnapshot = KnobSnapshot.Value();
    }
    return TRUE;
}

// Write the summary files (and the last snapshot). Called with stateLock held
// or after all threads have finished.
static VOID WriteOutputs()
{
    if (finished) return;
    finished = TRUE;

    for (UINT32 m = 0; m < bins.size(); m++)
    {
        ostringstream name;
        name << outputFileNamePrefix << "_SubSample_" << BinInterval(m) << "_bin_" << m % numBins << ".csv";
        ofstream out(name.str());
        out << CSV_HEADER << endl;
        WriteTotal(out, bins[m].bytes, bins[m].Unique(), binObservations[m], BinInterval(m));
    }
    WriteTotal(outputFile, footprint.bytes, footprint.Unique(), observations, MainInterval());
    outputFile.close();

    if (timelineFile.is_open())
    {
        WriteSnapshot();
        timelineFile.close();
    }
}

VOID Fini(INT32 code, VOID* v)
{
    PIN_GetLock(&stateLock, 0);
    for (ThreadData& t : threads)
    {
        for (const Release& r : t.pending)
            ApplyRelease(r.lo, r.hi);
        t.pending.clear();
    }
    WriteOutputs();
    PIN_ReleaseLock(&stateLock);
}

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

INT32 Usage(const string& message = "")
{
    if (!message.empty()) cerr << "memprint_trace: " << message << endl;
    cerr << "MemPrint memory-footprint tracer (splitter / sampler)." << endl;
    cerr << endl << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}

// Effective sampling interval of one bootstrap bin: a sampled reference lands in
// min(Poisson(lambda), bins) distinct bins, so a given bin receives it with
// probability E[min(Poisson(lambda), bins)] / bins (~ 1/r when lambda << bins).
static UINT64 BootstrapBinInterval(UINT32 interval, UINT32 bins, double lambda)
{
    double pmf = exp(-lambda), expected = 0, tail = 1;
    for (UINT32 k = 0; k < bins; k++)
    {
        expected += k * pmf;
        tail -= pmf;
        pmf *= lambda / (k + 1);
    }
    expected += bins * tail;
    return (UINT64)llround(interval * bins / expected);
}

static BOOL ParseIntervals(const string& list)
{
    std::istringstream in(list);
    string item;
    while (std::getline(in, item, ','))
    {
        UINT32 value = (UINT32)strtoul(item.c_str(), NULL, 10);
        if (value == 0) return FALSE;
        intervals.push_back(value);
    }
    return !intervals.empty();
}

int main(int argc, char* argv[])
{
    PIN_InitSymbols();
    if (PIN_Init(argc, argv)) return Usage();

    if (KnobMode.Value() == "splitter")
    {
        mode    = SPLITTER;
        numBins = KnobBins.Value();
        if (numBins == 0) return Usage("-bins must be positive");
        if (!ParseIntervals(KnobIntervals.Value())) return Usage("-intervals must be a list of positive integers");
        for (UINT32 interval : intervals)
        {
            if (interval < numBins) return Usage("every interval must be >= -bins");
            intervalThresholds.push_back(((UINT64)numBins << 32) / interval);
        }
        bins.resize(intervals.size() * numBins);
        if (KnobSnapshot.Value()) unions.resize(intervals.size());
    }
    else if (KnobMode.Value() == "sampler")
    {
        mode    = SAMPLER;
        numBins = KnobNumSplits.Value();
        if (KnobSamplingInterval.Value() == 0 || numBins == 0 || KnobSampleReplication.Value() == 0)
            return Usage("-i, -s and -r must be positive");
        lambda             = static_cast< double >(numBins) / KnobSampleReplication.Value();
        sampledBinInterval = BootstrapBinInterval(KnobSamplingInterval.Value(), numBins, lambda);
        bins.resize(numBins);
    }
    else
        return Usage("unknown -mode " + KnobMode.Value());
    binObservations.assign(bins.size(), 0);

    PIN_InitLock(&stateLock);
    PIN_InitLock(&allocLock);
    if (KnobTrackFrees.Value())
    {
        footprint.SetIndexed(TRUE);
        for (Footprint& bin : bins)
            bin.SetIndexed(TRUE);
        for (Footprint& u : unions)
            u.SetIndexed(TRUE);
        IMG_AddInstrumentFunction(ImageLoad, 0);
        PIN_AddSyscallEntryFunction(SyscallEntry, 0);
        PIN_AddSyscallExitFunction(SyscallExit, 0);
    }

    bufId = PIN_DefineTraceBuffer(sizeof(struct MEMREF), NUM_BUF_PAGES, BufferFull, 0);
    if (bufId == BUFFER_ID_INVALID)
    {
        cerr << "Error: could not allocate initial buffer" << endl;
        return 1;
    }

    if (!InitOutputFiles()) PIN_ExitProcess(1);

    TRACE_AddInstrumentFunction(Trace, 0);
    PIN_AddThreadStartFunction(ThreadStart, 0);
    PIN_AddFiniFunction(Fini, 0);

    PIN_StartProgram();
    return 0;
}

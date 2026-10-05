/*
 * Copyright (C) 2009-2021 Intel Corporation.
 * SPDX-License-Identifier: MIT
 */

/*
 * MemPrint memory-footprint tracer.
 *
 * Records every memory operand (address, size) into a Pin trace buffer and,
 * when the buffer fills, folds the references into per-thread maps of
 * address -> largest access size.  The footprint of a set of references is
 * the sum of the largest size seen at each unique address.
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
 * Output (in -outdir, which is created if missing):
 *   <Prefix>_<name>_<interval>_<pid>[_<args>].csv
 *   <Prefix>_<name>_<interval>_<pid>[_<args>]_SubSample_<binInterval>_bin_<j>.csv
 * with Prefix = Buffered (splitter) or Sampled (sampler) and the columns
 *   FunctionName,MemUsageObs,UniqueAddresses,CountObs,SamplingInterval
 */

#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
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

typedef unordered_map< ADDRINT, UINT32 > FootprintMap; // address -> largest access size

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

// Per-thread state; each thread only touches its own slot.
struct ThreadData
{
    UINT64 observations = 0;     // references seen by BufferFull
    FootprintMap footprint;      // splitter: exact footprint
    vector< FootprintMap > bins; // splitter: [interval * numBins + bin]; sampler: [bin]
    vector< UINT64 > binObservations;
};
ThreadData* threadData[MAX_THREADS];

static UINT32 seed[MAX_THREADS]; // 32-bit LCG state per thread

vector< UINT32 > intervals;          // splitter
vector< UINT64 > intervalThresholds; // splitter: accept a reference into interval k iff lcg < threshold[k]
UINT32 numBins;             // bins per interval (splitter) or bootstrap bins (sampler)
double lambda;              // sampler
UINT64 sampledBinInterval;  // sampler: effective sampling interval of one bootstrap bin

string outputFileNamePrefix;
ofstream outputFile;

static const char* CSV_HEADER = "FunctionName,MemUsageObs,UniqueAddresses,CountObs,SamplingInterval";

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
    UINT32 k   = 0;
    double p   = 1.0;
    double L   = exp(-lambda);
    do
    {
        ++k;
        double u = (lcg_step(seed[tid]) & 0x7FFFFFFF) / static_cast< double >(0x80000000);
        p *= u;
    } while (p > L);
    return k - 1;
}

static inline ADDRINT PIN_FAST_ANALYSIS_CALL ShouldSample(THREADID tid, UINT32 rate)
{
    return ((lcg_step(seed[tid]) % rate) == 0) ? 1 : 0;
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
/* Analysis                                                              */
/* ===================================================================== */

static inline VOID Record(FootprintMap& map, ADDRINT address, UINT32 size)
{
    UINT32& largest = map[address];
    if (largest < size) largest = size;
}

static VOID SplitReference(ThreadData* td, THREADID tid, ADDRINT address, UINT32 size)
{
    Record(td->footprint, address, size);

    // Keep the reference with probability bins/interval and put it in a uniform bin,
    // so every bin is a 1-in-interval subsample.
    for (UINT32 k = 0; k < intervals.size(); k++)
    {
        if (lcg_step(seed[tid]) < intervalThresholds[k])
        {
            UINT32 bin = (lcg_step(seed[tid]) / 256) % numBins;
            Record(td->bins[k * numBins + bin], address, size);
            ++td->binObservations[k * numBins + bin];
        }
    }
}

static VOID BootstrapReference(ThreadData* td, THREADID tid, ADDRINT address, UINT32 size)
{
    UINT32 times = sample_poisson(lambda, tid);
    if (times == 0) return;
    if (times > numBins) times = numBins; // cannot place in more distinct bins than exist

    // Place the reference in `times` distinct bins.
    vector< bool > used(numBins, false);
    UINT32 count = 0;
    while (count < times)
    {
        UINT32 bin = lcg_step(seed[tid]) % numBins;
        if (used[bin]) continue;
        used[bin] = true;
        count++;
        Record(td->bins[bin], address, size);
        ++td->binObservations[bin];
    }
}

VOID* BufferFull(BUFFER_ID id, THREADID tid, const CONTEXT* ctxt, VOID* buf, UINT64 numElements, VOID* v)
{
    if (!buf) return buf;
    ThreadData* td = threadData[tid];

    struct MEMREF* reference = (struct MEMREF*)buf;
    for (UINT64 i = 0; i < numElements; i++, reference++)
    {
        ADDRINT address = reference->ea;
        UINT32 size     = reference->size;
        if (address == 0) continue;

        ++td->observations;
        if (mode == SPLITTER)
            SplitReference(td, tid, address, size);
        else
            BootstrapReference(td, tid, address, size);
    }
    return buf;
}

VOID ThreadStart(THREADID tid, CONTEXT* ctxt, INT32 flags, VOID* v)
{
    if (tid >= MAX_THREADS)
    {
        cerr << "memprint_trace: thread id " << tid << " exceeds MAX_THREADS (" << MAX_THREADS << ")" << endl;
        PIN_ExitProcess(1);
    }
    UINT32 base = KnobSeed.Value().empty() ? PIN_GetPid() : (UINT32)strtoul(KnobSeed.Value().c_str(), NULL, 10);
    seed[tid]   = base * 31 + tid * 19;

    ThreadData* td      = new ThreadData;
    size_t numMaps      = (mode == SPLITTER) ? intervals.size() * numBins : numBins;
    td->bins.resize(numMaps);
    td->binObservations.assign(numMaps, 0);
    threadData[tid] = td;
}

/* ===================================================================== */
/* Output                                                                */
/* ===================================================================== */

// Footprint accumulated across threads.
struct Footprint
{
    FootprintMap map;
    UINT64 bytes = 0;

    VOID Merge(const FootprintMap& other)
    {
        for (const auto& entry : other)
        {
            UINT32& largest = map[entry.first];
            if (largest < entry.second)
            {
                bytes += entry.second - largest;
                largest = entry.second;
            }
        }
    }
};

static VOID WriteTotal(ofstream& out, UINT64 bytes, size_t uniqueAddresses, UINT64 observations, UINT64 interval)
{
    out << "Total," << bytes << "," << uniqueAddresses << "," << observations << "," << interval << endl;
}

static VOID WriteBin(UINT64 binInterval, UINT32 bin, UINT32 mapIndex, Footprint* total)
{
    ostringstream name;
    name << outputFileNamePrefix << "_SubSample_" << binInterval << "_bin_" << bin << ".csv";
    ofstream out(name.str());
    out << CSV_HEADER << endl;

    Footprint binFootprint;
    UINT64 binObservations = 0;
    for (UINT32 t = 0; t < MAX_THREADS; t++)
    {
        if (!threadData[t]) continue;
        if (total) total->Merge(threadData[t]->bins[mapIndex]);
        binFootprint.Merge(threadData[t]->bins[mapIndex]);
        binObservations += threadData[t]->binObservations[mapIndex];
    }
    WriteTotal(out, binFootprint.bytes, binFootprint.map.size(), binObservations, binInterval);
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

static UINT32 MainInterval() { return (mode == SPLITTER) ? 1 : KnobSamplingInterval.Value(); }

VOID InitOutputFile()
{
    string outDir = KnobOutDir.Value();
    mkdir(outDir.c_str(), 0775);

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
        cerr << "Error: Could not open file " << filename << endl;
        PIN_ExitApplication(1);
    }
    outputFile << CSV_HEADER << endl;
}

VOID Fini(INT32 code, VOID* v)
{
    Footprint total;
    UINT64 observations = 0;

    if (mode == SPLITTER)
    {
        for (UINT32 k = 0; k < intervals.size(); k++)
            for (UINT32 j = 0; j < numBins; j++)
                WriteBin(intervals[k], j, k * numBins + j, NULL);

        for (UINT32 t = 0; t < MAX_THREADS; t++)
            if (threadData[t]) total.Merge(threadData[t]->footprint);
    }
    else
    {
        for (UINT32 j = 0; j < numBins; j++)
            WriteBin(sampledBinInterval, j, j, &total);
    }

    for (UINT32 t = 0; t < MAX_THREADS; t++)
        if (threadData[t]) observations += threadData[t]->observations;

    WriteTotal(outputFile, total.bytes, total.map.size(), observations, MainInterval());
    outputFile.close();
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
    }
    else if (KnobMode.Value() == "sampler")
    {
        mode    = SAMPLER;
        numBins = KnobNumSplits.Value();
        if (KnobSamplingInterval.Value() == 0 || numBins == 0 || KnobSampleReplication.Value() == 0)
            return Usage("-i, -s and -r must be positive");
        lambda             = static_cast< double >(numBins) / KnobSampleReplication.Value();
        sampledBinInterval = BootstrapBinInterval(KnobSamplingInterval.Value(), numBins, lambda);
    }
    else
        return Usage("unknown -mode " + KnobMode.Value());

    bufId = PIN_DefineTraceBuffer(sizeof(struct MEMREF), NUM_BUF_PAGES, BufferFull, 0);
    if (bufId == BUFFER_ID_INVALID)
    {
        cerr << "Error: could not allocate initial buffer" << endl;
        return 1;
    }

    InitOutputFile();

    TRACE_AddInstrumentFunction(Trace, 0);
    PIN_AddThreadStartFunction(ThreadStart, 0);
    PIN_AddFiniFunction(Fini, 0);

    PIN_StartProgram();
    return 0;
}

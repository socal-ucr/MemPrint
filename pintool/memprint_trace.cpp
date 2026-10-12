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
 *   spatial   Selects 1-in-i *addresses* by a salted hash and records every
 *             access to them, immediately and in order across threads (no
 *             trace buffer: selected accesses are rare). Each address is selected with probability 1/i
 *             whatever its access pattern, so the footprint is estimated by
 *             the selected footprint x i, with no model (binomial error
 *             ~ 1/sqrt(addresses / i)). The selected addresses are also split
 *             into -s hash buckets (1-in-i*s each) for an error bar.
 *
 * Optional:
 *
 *   -track_frees 1  free/realloc/munmap remove the released address range
 *                   from every footprint, so footprints are live memory.
 *   -footprint bytes|starts|pages  the bytes touched (union of every access's byte
 *                   range), or the paper's sum of the largest access size per
 *                   start address, or the 4 KB pages with a touched byte (x 4096:
 *                   what the kernel backs; splitter and -window/-pages_only only).
 *                   Default: bytes with -track_frees or
 *                   -window, else starts. Spatial and windowed sampling with
 *                   bytes select aligned 8-byte words and keep the bytes of
 *                   each access inside selected words, so selected bytes x i
 *                   is unbiased. With pages, the splitter's exact footprint
 *                   (Bin -1) counts pages and its sampled bins stay in bytes;
 *                   the windowed Estimate is the resident pages of live blocks
 *                   and of the program's writable segments and main stack.
 *   -snapshot N     every N memory references (time), append the current
 *                   footprints to a timeline CSV.
 *   -clock seconds  -snapshot N is every N milliseconds of wall time instead
 *                   (under Pin: instrumented time, slower than a native run and
 *                   not uniformly so). Every snapshot row ends with Seconds, the
 *                   wall time since the tool started, whatever the clock.
 *   -stop N         after N references, write all outputs and detach.
 *   -window W -period P  (spatial) watch accesses only for W of every P
 *                   memory references; between windows only a per-block
 *                   reference count and the allocator run. Live allocations
 *                   are tracked all the time and the windows measure how much
 *                   of each one is touched; see "Windows" below.
 *
 * Time counts memory references executed (with -clock seconds too: windows,
 * -period and -stop stay in references): one per reference in the
 * splitter, i per sampled reference in the sampler (an estimate). With
 * -window, references are counted per basic block (a REP string instruction
 * counts once).
 *
 * Windows (-window): live allocations (heap blocks and anonymous mmaps) are
 * tracked all the time, and the footprint is estimated at every snapshot as
 *
 *   touched bytes of fresh blocks
 *   + touched bytes of reused blocks x density
 *   + selected footprint outside blocks x i.
 *
 * A block is "fresh" when its pages were mapped for it (an anonymous mmap, or
 * a malloc served by a new mmap), so they are untouched when it is allocated;
 * otherwise "reused" (e.g. small blocks from the heap). Touched bytes are the
 * block's bytes on resident pages, read from /proc/self/pagemap at every
 * snapshot: exact to the 4 KB page for fresh blocks, between windows too
 * (transparent huge pages are disabled for the process). With -footprint
 * starts, the footprint counts the largest access at every start address, so
 * overlapping accesses (e.g. unaligned copies) count more than the bytes they
 * touch; density is that ratio for reused blocks, measured in the windows.
 * With -footprint bytes (the default here) the footprint is the bytes touched
 * and density is 1. With starts, windows select 1-in-i
 * 64-byte chunks and see every access in them, so they know both the
 * footprint and the bytes covered in the selected chunks; the ratio does not
 * depend on how much of the run the windows covered. With bytes, they select
 * 8-byte words, like spatial sampling. Memory outside blocks
 * (stack, globals) is only seen in windows.
 *
 * Each snapshot is a row of <prefix>_windowed.csv: Time, Watching (in a
 * window), Windows (completed), AllocatedBytes and Blocks (live), FreshBytes,
 * FreshResident, ReusedBytes, ReusedResident, then the windows' selected
 * footprint x i in fresh blocks, reused blocks and outside blocks (FreshEst,
 * ReusedEst, OtherEst), the bytes covered x i in fresh and reused blocks
 * (FreshCovered, ReusedCovered), Density (ReusedEst / ReusedCovered, 1 until
 * 64 KB are covered), Estimate, Window and Period, ..., and Seconds.
 *
 * Output (in -outdir, which is created if missing):
 *   <Prefix>_<name>_<interval>_<pid>[_<args>].csv
 *   <Prefix>_<name>_<interval>_<pid>[_<args>]_SubSample_<binInterval>_bin_<j>.csv
 *   <Prefix>_<name>_<interval>_<pid>[_<args>]_timeline.csv        (-snapshot)
 * with Prefix = Buffered (splitter), Sampled (sampler) or Spatial (spatial). Summary files have
 * the columns FunctionName,MemUsageObs,UniqueAddresses,CountObs,SamplingInterval;
 * the timeline has Time,SamplingInterval,Bin,MemUsageObs,UniqueAddresses,CountObs,FreedBytes,
 * Singletons,Doubletons,Tripletons,Quadrupletons,Discovered,Seconds. Bin -1 is the exact footprint in the splitter and the union of the
 * bins in the sampler; Bin -2 (splitter) is the union of one interval's bins, a
 * 1-in-(interval/bins) sample. Singletons..Quadrupletons count the addresses
 * sampled exactly 1..4 times, from which Chao1/iChao1 estimate the addresses
 * never sampled. Discovered counts how often an address entered the sample
 * (again after a free); its growth between snapshots, per new sample, tells
 * a growing footprint (mostly new addresses) from a plateau (mostly repeats).
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <elf.h>
#include <string.h>
#include <sys/syscall.h>
#include <map>
#include <set>
#include <sys/mman.h>
#include <sys/prctl.h>
#ifndef PR_SET_THP_DISABLE
#define PR_SET_THP_DISABLE 41 // Linux 3.15
#endif
#include <fcntl.h>
#include <unistd.h>
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

KNOB< string > KnobMode(KNOB_MODE_WRITEONCE, "pintool", "mode", "splitter", "splitter | sampler | spatial");
KNOB< UINT32 > KnobSamplingInterval(KNOB_MODE_WRITEONCE, "pintool", "i", "100",
                                    "sampler: sample 1-in-i memory references; spatial: select 1-in-i addresses");
KNOB< UINT32 > KnobNumSplits(KNOB_MODE_WRITEONCE, "pintool", "s", "100",
                             "sampler: number of bootstrap bins; spatial: number of hash buckets");
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
KNOB< UINT32 > KnobBufferPages(KNOB_MODE_WRITEONCE, "pintool", "buffer_pages", "0",
                              "trace buffer pages per thread (0: 1024, or 16 with -snapshot/-track_frees so that "
                              "threads' buffered accesses and frees interleave closely; unused in spatial mode)");
KNOB< UINT64 > KnobWindow(KNOB_MODE_WRITEONCE, "pintool", "window", "0",
                          "spatial: watch accesses for N memory references of every -period (0: always; implies -track_frees)");
KNOB< UINT64 > KnobPeriod(KNOB_MODE_WRITEONCE, "pintool", "period", "0", "spatial with -window: window period in memory references");
KNOB< UINT64 > KnobWindowAlloc(KNOB_MODE_WRITEONCE, "pintool", "window_alloc", "1048576",
                               "spatial with -window: an allocation of at least N bytes opens a window, at most one "
                               "per period on top of the periodic ones (0: off)");
KNOB< string > KnobWindowSwitch(KNOB_MODE_WRITEONCE, "pintool", "window_switch", "auto",
                                "windowed: flush (discard all translated code at every window change) | version (two "
                                "instrumented versions of the code, switched at trace entry) | auto (flush, until a window "
                                "change retranslates more than -version_traces traces; then version)");
KNOB< UINT64 > KnobVersionTraces(KNOB_MODE_WRITEONCE, "pintool", "version_traces", "10000",
                                 "windowed, -window_switch auto: traces retranslated after one window change that make "
                                 "the tool switch to versions");
KNOB< BOOL > KnobEdgeCorrection(KNOB_MODE_WRITEONCE, "pintool", "edge_correction", "1",
                                 "windowed: count half a page less for every touched page inside a block that borders an "
                                 "untouched one (it is probably partly used)");
KNOB< BOOL > KnobBlockSlack(KNOB_MODE_WRITEONCE, "pintool", "block_slack", "0",
                            "windowed: measure heap blocks over their whole glibc chunk (requested size plus slack)");
KNOB< BOOL > KnobPagesOnly(KNOB_MODE_WRITEONCE, "pintool", "pages_only", "0",
                          "spatial with -snapshot: no windows at all; estimate from the pages of live blocks and of "
                          "the program's writable segments and stack");
KNOB< string > KnobDirtyEvery(KNOB_MODE_WRITEONCE, "pintool", "dirty_every", "",
                              "spatial with -window: clear the kernel's soft-dirty page bits every N memory references "
                              "(default: every 10 snapshots; 0: off, reused blocks are measured by residency)");
KNOB< string > KnobFootprint(KNOB_MODE_WRITEONCE, "pintool", "footprint", "",
                            "bytes (bytes touched) | starts (largest access per start address, the paper's) | pages "
                            "(4 KB pages touched x 4096; splitter, -window, -pages_only); "
                            "default: bytes with -track_frees or -window, else starts");
KNOB< string > KnobClock(KNOB_MODE_WRITEONCE, "pintool", "clock", "refs",
                         "refs | seconds: -snapshot counts memory references, or milliseconds of wall time");
KNOB< BOOL > KnobSites(KNOB_MODE_WRITEONCE, "pintool", "sites", "0",
                       "splitter with -track_frees: per allocation site, requested against touched bytes (*_sites.csv, "
                       "*_blocks.csv)");
KNOB< UINT32 > KnobSiteContext(KNOB_MODE_WRITEONCE, "pintool", "site_context", "2",
                                "-sites: caller frames above each large block's allocation call written with it (*_blocks.csv "
                                "context), so that analysis/memprint/sites.py can split a site reached through a wrapper");
KNOB< INT32 > KnobSkipZeroing(KNOB_MODE_WRITEONCE, "pintool", "skip_zeroing", "-1",
                              "splitter: zeros that memset/bzero (and libmpr's __mpr_zero) write over at least 4 KB do not "
                              "count as touching untouched bytes (-1: on with -sites)");
KNOB< UINT64 > KnobSitesBlockMin(KNOB_MODE_WRITEONCE, "pintool", "sites_block_min", "65536",
                                 "-sites: blocks of at least this many bytes are also written one by one (*_blocks.csv)");
KNOB< UINT64 > KnobStop(KNOB_MODE_WRITEONCE, "pintool", "stop", "0",
                        "after N memory references, write the outputs and detach (0: run to the end)");

/* ===================================================================== */
/* Footprint                                                             */
/* ===================================================================== */

// Largest access size and number of samples per address, with running
// totals: bytes, and the number of addresses sampled exactly 1, 2, 3 and 4
// times, from which Chao1/iChao1 estimate how many addresses were never
// sampled. When indexed, addresses are also grouped by page so that a freed
// range can be erased.
// Footprint definition (-footprint): the paper's sum of the largest access size per start
// address ("starts"), or the bytes touched, the union of [address, address + size) ("bytes").
// Overlapping accesses (different widths or offsets) count once per start address in the
// first, once per byte in the second. -footprint pages also counts bytes (countBytes), and the
// splitter's exact footprint reports the pages that hold them (SetPages).
BOOL countBytes = FALSE;
BOOL countPages = FALSE;

class Footprint
{
  public:
    UINT64 bytes = 0;      // footprint: see countBytes
    UINT64 freed = 0;      // bytes removed by Erase
    UINT64 sampled[5] = {0}; // sampled[k]: addresses sampled exactly k times (k = 1..4)
    UINT64 discovered = 0;   // times an address entered the sample (again after a free)

    VOID SetIndexed(BOOL on) { indexed = on; }
    VOID SetPages(BOOL on) { pages4k = on; } // countBytes: bytes and freed count whole pages with a touched byte
    template < typename F > VOID ForEach(F f) const
    {
        for (const auto& e : entries)
            f(e.first, e.second.size);
    }
    size_t Unique() const { return entries.size(); }

    VOID Record(ADDRINT address, UINT32 size)
    {
        Entry& e = entries[address];
        if (e.count == 0)
        {
            ++discovered;
            if (indexed) pages[address >> PAGE_BITS].push_back(address);
        }
        Count(e.count, -1);
        Count(++e.count, +1);
        if (countBytes)
        {
            UINT64 newPages = 0, added = Touch(address, address + size, &newPages);
            bytes += pages4k ? newPages << PAGE_BITS : added;
        }
        else if (e.size < size)
            bytes += size - e.size;
        if (e.size < size) e.size = size;
    }

    // Remove every address in [lo, hi).
    VOID Erase(ADDRINT lo, ADDRINT hi)
    {
        if (hi <= lo) return;
        if (countBytes) Untouch(lo, hi);
        if (pages.empty()) return;
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

    // countBytes: of [lo, hi), the bytes touched, the end offset of the last touched byte (0: none)
    // and the pages with a touched byte (-sites)
    VOID Summarize(ADDRINT lo, ADDRINT hi, UINT64* bytesTouched, ADDRINT* highWater, UINT64* pagesTouched) const
    {
        *bytesTouched = 0, *highWater = 0, *pagesTouched = 0;
        if (!countBytes || hi <= lo) return;
        for (ADDRINT page = lo >> PAGE_BITS; (page << PAGE_BITS) < hi; page++)
        {
            auto it = touched.find(page);
            if (it == touched.end()) continue;
            ADDRINT start = page << PAGE_BITS, a = std::max(lo, start), b = std::min(hi, start + (1 << PAGE_BITS));
            UINT64 n = 0;
            ADDRINT last = 0;
            for (UINT32 w = 0; w < (1 << (PAGE_BITS - 6)); w++)
            {
                ADDRINT wlo = start + w * 64, whi = wlo + 64;
                if (whi <= a || wlo >= b || !it->second.word[w]) continue;
                UINT64 mask = ~0ULL;
                if (wlo < a) mask &= ~0ULL << (a - wlo);
                if (whi > b) mask &= ~0ULL >> (whi - b);
                UINT64 bits = it->second.word[w] & mask;
                if (!bits) continue;
                n += __builtin_popcountll(bits);
                last = wlo + 64 - __builtin_clzll(bits);
            }
            if (n == 0) continue;
            *bytesTouched += n, ++*pagesTouched;
            *highWater = last - lo;
        }
    }

    // countBytes: the runs of pages of [lo, hi) holding a touched byte, as (first page, pages) from
    // lo's page; at most maxRuns, the closest runs merged when there are more (-sites)
    VOID Runs(ADDRINT lo, ADDRINT hi, vector< std::pair< UINT64, UINT64 > >& runs, size_t maxRuns) const
    {
        runs.clear();
        if (!countBytes || hi <= lo) return;
        ADDRINT first = lo >> PAGE_BITS, last = (hi - 1) >> PAGE_BITS;
        for (ADDRINT page = first; page <= last; page++)
        {
            auto it = touched.find(page);
            if (it == touched.end()) continue;
            BOOL any = FALSE;
            for (UINT64 w : it->second.word)
                if (w) { any = TRUE; break; }
            if (!any) continue;
            UINT64 rel = page - first;
            if (!runs.empty() && runs.back().first + runs.back().second == rel)
                runs.back().second++;
            else
                runs.push_back({rel, 1});
        }
        while (runs.size() > maxRuns) // merge the two runs with the smallest gap between them
        {
            size_t best = 0;
            for (size_t i = 1; i + 1 < runs.size(); i++)
                if (runs[i + 1].first - (runs[i].first + runs[i].second) < runs[best + 1].first - (runs[best].first + runs[best].second))
                    best = i;
            runs[best].second = runs[best + 1].first + runs[best + 1].second - runs[best].first;
            runs.erase(runs.begin() + best + 1);
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
    BOOL pages4k = FALSE; // see SetPages

    // countBytes: bytes touched, one bit per byte, per page
    struct Bits
    {
        UINT64 word[(1 << PAGE_BITS) / 64] = {0};
    };
    unordered_map< ADDRINT, Bits > touched;
    ADDRINT lastPage = ~(ADDRINT)0;
    Bits* lastBits   = NULL;

    // Set (or clear) the bits of [lo, hi) within one page; returns how many changed.
    static UINT64 Mark(Bits& b, ADDRINT lo, ADDRINT hi, BOOL set)
    {
        UINT64 changed = 0;
        for (ADDRINT a = lo; a < hi;)
        {
            UINT32 w = (a >> 6) & ((1 << (PAGE_BITS - 6)) - 1), bit = a & 63;
            UINT32 n    = (UINT32)std::min< ADDRINT >(64 - bit, hi - a);
            UINT64 mask = (n == 64 ? ~0ULL : ((1ULL << n) - 1)) << bit;
            UINT64 now  = set ? (b.word[w] | mask) : (b.word[w] & ~mask);
            changed += __builtin_popcountll(b.word[w] ^ now);
            b.word[w] = now;
            a += n;
        }
        return changed;
    }

    // Mark [lo, hi) touched; returns the bytes newly touched, and adds the pages newly touched to
    // *newPages (a page is in `touched` iff it has a touched byte: Untouch erases empty ones).
    UINT64 Touch(ADDRINT lo, ADDRINT hi, UINT64* newPages)
    {
        UINT64 added = 0;
        while (lo < hi)
        {
            ADDRINT page = lo >> PAGE_BITS, end = std::min(hi, (page + 1) << PAGE_BITS);
            if (page != lastPage)
            {
                size_t before = touched.size();
                lastBits = &touched[page], lastPage = page;
                *newPages += touched.size() - before;
            }
            added += Mark(*lastBits, lo, end, TRUE);
            lo = end;
        }
        return added;
    }

    // Clear [lo, hi): the bytes are released.
    VOID Untouch(ADDRINT lo, ADDRINT hi)
    {
        ADDRINT first = lo >> PAGE_BITS, last = (hi - 1) >> PAGE_BITS;
        lastPage = ~(ADDRINT)0; // pages may be erased
        auto clear = [&](unordered_map< ADDRINT, Bits >::iterator it) {
            ADDRINT start = it->first << PAGE_BITS;
            UINT64 n      = Mark(it->second, std::max(lo, start), std::min(hi, start + (1 << PAGE_BITS)), FALSE);
            if (!pages4k) bytes -= n, freed += n;
            for (UINT64 w : it->second.word)
                if (w) return;
            touched.erase(it);
            if (pages4k) bytes -= 1 << PAGE_BITS, freed += 1 << PAGE_BITS;
        };
        if (last - first + 1 < touched.size())
        {
            for (ADDRINT page = first; page <= last; page++)
            {
                auto it = touched.find(page);
                if (it != touched.end()) clear(it);
            }
        }
        else
        {
            for (auto it = touched.begin(); it != touched.end();)
            {
                auto next = std::next(it);
                if (it->first >= first && it->first <= last) clear(it);
                it = next;
            }
        }
    }

    VOID Count(UINT32 count, INT32 delta)
    {
        if (count >= 1 && count <= 4) sampled[count] += delta;
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
            if (!countBytes) bytes -= it->second.size, freed += it->second.size; // (bytes: Untouch)
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

enum Mode { SPLITTER, SAMPLER, SPATIAL };
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
    BOOL probe = FALSE; // -sites: not a release, a look at the block at lo (calloc's zeroing, see SiteProbe)
    BOOL zero  = FALSE; // -skip_zeroing: not a release; [lo, hi) is the range the thread's next memset clears (or empty)
};

// Per-thread state; each slot is only touched by its own thread (except refs, read by
// TotalRefs).
struct alignas(64) ThreadData // own cache lines: counters are written by their thread on every block
{
    UINT32 seed = 0;           // 32-bit LCG state
    UINT64 refs = 0;           // spatial: memory references executed by this thread
    UINT64 checkAt = 0;        // windowed: refs at which to next advance time
    UINT32 epoch   = 0;        // windowed: window changes this thread's code has seen
    vector< Release > pending; // releases not yet applied
    // Allocator call in progress (outermost call only)
    UINT32 allocDepth = 0;
    ADDRINT allocSp = 0; // stack pointer at the outermost call
    ADDRINT allocSize = 0, allocPtr = 0, allocOut = 0;
    ADDRINT allocHeader = 0; // realloc: the old chunk's size field at the entry
    ADDRINT allocSite   = 0; // -sites: the outermost call's return address, or the id passed to __mpr_alloc
    BOOL allocSiteId    = FALSE; // allocSite is an id from the MemPrintResize pass, not an address
    ADDRINT pendingSite = 0;     // -sites: the id passed to __mpr_* by a program built with the pass,
    BOOL pendingSiteSet = FALSE; // taken by the next outermost allocator call of the thread
    UINT8 allocKind     = 0; // -sites: SITE_MALLOC, SITE_CALLOC, ...
    // -sites: caller frames above the allocation call (return addresses), for the outermost call
    ADDRINT allocCtx[4] = {0}, pendingCtx[4] = {0}, scratchCtx[4] = {0};
    UINT32 allocCtxN = 0, pendingCtxN = 0, scratchCtxN = 0;
    ADDRINT scratchFor = 0;           // the return address scratchCtx was taken for
    ADDRINT zeroLo = 0, zeroHi = 0;   // -skip_zeroing: the range the memset being processed clears
    // -sites, programs built with the pass: the call site growing a container (__mpr_growth) and the
    // capacity request (__mpr_capacity) the next allocation serves
    ADDRINT growthSite = 0, pendingCap = 0, allocCap = 0, pendingCapElem = 0, allocCapElem = 0;
    BOOL pendingCapCtx = FALSE, pendingCapSet = FALSE, allocCapCtx = FALSE, allocCapSet = FALSE;
    BOOL zeroQueued = FALSE;          // the last range queued is not empty
    BOOL allocMallocLike = FALSE; // the outermost call returns its block (malloc, calloc, memalign, ...)
    BOOL allocFresh = FALSE; // windowed: the allocator mapped new pages (mmap/mremap) for this call
    ADDRINT munmapLo = 0, munmapHi = 0;
    ADDRINT mmapLen = 0; // windowed: anonymous mmap in progress
    BOOL clearDirty = FALSE; // windowed: clear soft-dirty bits when this system call returns
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
UINT64 nextSnapshot = 0;  // in references, or (-clock seconds) microseconds since the start
UINT64 lastSnapshot = ~(UINT64)0; // time of the last snapshot written
BOOL clockSeconds = FALSE;      // -clock seconds
UINT64 startMicros = 0;         // wall time when the tool started
UINT64 snapshotMicros = 0;      // since the start, at the snapshot being written
UINT64 nextClockCheck = 0;      // -clock seconds: Tick reads the clock again at this many references
const UINT64 CLOCK_CHECK_REFS = 1024; // (about 1 ms in the splitter; a clock read costs well under a microsecond)

static UINT64 Micros()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (UINT64)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}
BOOL finished = FALSE;   // outputs written (end of run or -stop)

// Live heap blocks (address -> size), for -track_frees.
PIN_LOCK allocLock;
unordered_map< ADDRINT, ADDRINT > allocations;

/* -sites: requested against touched bytes per allocation site ----------- */
// A block's touched bytes are read from the exact footprint (one bit per byte) when the block is
// released, just before Untouch clears them (ApplyRelease), and at the end for blocks still live.
// The site is the return address of the outermost allocator call, which with operator new hooked
// as an allocator is in the program's code (STL code is instantiated there), or the id the
// MemPrintResize pass passes to __mpr_alloc.
enum { SITE_MALLOC = 1, SITE_CALLOC = 2, SITE_REALLOC = 4, SITE_MEMALIGN = 8, SITE_NEW = 16, SITE_MPR = 32 };
BOOL sitesOn = FALSE;
PIN_LOCK blockSiteLock; // siteLive, dying; never held while waiting for stateLock
struct SiteBlock
{
    ADDRINT site     = 0;
    BOOL siteId      = FALSE;
    ADDRINT size     = 0;
    UINT64 born      = 0; // timeNow at the allocation
    UINT8 kind       = 0;
    ADDRINT grownFrom = 0; // realloc that moved: the old block's size
    BOOL moved       = FALSE; // freed by a realloc that moved it
    UINT64 allocTouched = 0; // calloc: bytes the allocator itself touched (zeroing), see SiteProbe
    ADDRINT ctx[4] = {0};    // caller frames above the allocation call (-site_context)
    UINT32 ctxN = 0;
    UINT32 chain = ~0u;      // a container's buffer: its growth chain (capChains)
};
// A container's growth chain (programs built with the pass): its successive buffers, each larger,
// allocated while the previous one is still live (the copy), for one capacity key: the call site
// that grew it (cap:0x<growth site>) or the growth point (id:0x<point>).
struct CapChain
{
    ADDRINT key;
    BOOL ctx;
    ADDRINT elem, lastPtr;
    UINT64 blocks = 0, finalSize = 0, finalHighWater = 0;
};
vector< CapChain > capChains;                                    // blockSiteLock
std::map< std::pair< ADDRINT, BOOL >, UINT32 > capLatest;        // key -> its latest chain
unordered_map< ADDRINT, SiteBlock > siteLive; // address -> block
unordered_map< ADDRINT, SiteBlock > dying;    // freed, waiting for the release to be applied
struct SiteStats
{
    UINT32 kinds = 0;
    UINT64 count = 0, requested = 0, requestedMax = 0, touched = 0, highWater = 0, pagesTouched = 0, pages = 0;
    UINT64 neverTouched = 0, allocTouched = 0, moves = 0, grown = 0, lifetime = 0;
    double maxHighWaterRatio = 0, minHighWaterRatio = 2;
};
std::map< std::pair< ADDRINT, BOOL >, SiteStats > siteStats; // (site, is an id) -> totals (stateLock)
struct ImageRange
{
    ADDRINT lo, hi;
    ADDRINT bias; // load offset: an address minus it is the address in the file (what addr2line takes)
    string name;
};
vector< ImageRange > images; // for site addresses: image + offset (stateLock)
ofstream blocksFile;          // *_blocks.csv

// Windows (-window): live allocations.
struct Block
{
    ADDRINT size;
    BOOL mapped; // an anonymous mmap (munmap can trim it)
    BOOL fresh;  // its pages were mapped for it, so its resident pages are the ones it touched
    vector< UINT8 > written; // reused blocks: per page, written since the block was allocated (soft-dirty)
    UINT64 sampled = 0, covered = 0; // selected footprint in the block and the bytes it covers (last Attribute)
    BOOL allWritten = FALSE;         // every page of `written` is set, so harvests can skip it
    // Heap blocks: the end of the usable chunk (requested size plus the allocator's slack), which
    // string and copy functions read and write past the requested size. 0: the block's size.
    ADDRINT span = 0;
    ADDRINT Extent() const { return span ? span : size; }
};
BOOL windowed = FALSE;
BOOL pagesOnly = FALSE; // -pages_only: windowed bookkeeping, but never a window
// The program's writable image sections (.data, .bss) of every loaded image, for -pages_only.
vector< std::pair< ADDRINT, ADDRINT > > dataRegions;
volatile BOOL watching = TRUE; // inside a window (read at instrumentation time)
BOOL flushPending = FALSE;     // instrumentation must be redone for a window change
// -window_switch version: every trace exists in a watching and a not-watching version. CheckTime,
// which every thread runs every checkStep references, loads `watching` into a tool register, and
// a version case at each trace entry moves to the other version if they differ, so no translated
// code is discarded.
// Versions cost a test at every trace entry (2mm LARGE with nothing watched: 63 s against 16 s),
// flushing costs retranslating all code that runs (cc1 at 5% watched: 658 s against 127 s), so
// -window_switch auto starts with flushing and switches when a window change was expensive.
BOOL versioned = FALSE, autoSwitch = FALSE;
UINT64 tracesSinceChange = 0; // traces instrumented since the last window change
REG watchReg;
enum { VERSION_IDLE = 0, VERSION_WATCH = 1 };
UINT64 windowStart = 0, windowEnd = 0, nextPeriodic = 0, windowsDone = 0, checkStep = 0;
// Windows opened by large allocations (-window_alloc): how many, when the last one ended, and
// whether the current window is one; references watched in closed windows.
UINT64 triggeredWindows = 0, lastTriggeredEnd = 0, watchedRefs = 0;
BOOL triggeredWindow = FALSE;
volatile UINT32 epoch = 0; // windowed: number of window changes
std::map< ADDRINT, Block > blocks; // guarded by stateLock
// Selected footprint and the bytes its accesses cover, in reused [0] and fresh
// [1] blocks (as of the last Attribute), and selected footprint outside blocks.
UINT64 kindSampled[2] = {0, 0}, kindCovered[2] = {0, 0}, otherSampled = 0;
ofstream windowedFile;
int pagemapFd = -1;       // /proc/self/pagemap
int clearRefsFd = -1;     // /proc/self/clear_refs, to clear soft-dirty bits (-dirty_every)
UINT64 dirtyEvery = 0, nextDirtyClear = 0;
BOOL dirtyBySnapshots = FALSE; // -clock seconds without -dirty_every: clear every 10 snapshots
UINT64 snapshotsSinceClear = 0;
vector< UINT64 > pageBits; // buffer for pagemap entries

vector< UINT32 > intervals;          // splitter
vector< UINT64 > intervalThresholds; // splitter: accept a reference into interval k iff lcg < threshold[k]
UINT32 numBins;                      // bins per interval (splitter) or bootstrap bins (sampler)
double lambda;                       // sampler
UINT64 salt;                         // spatial: per-run hash salt
UINT64 spatialThreshold;             // spatial: an address is selected iff Mix(address ^ salt) < threshold
UINT64 sampledBinInterval;           // sampler: effective sampling interval of one bootstrap bin

string outputFileNamePrefix;
ofstream outputFile;
ofstream timelineFile;

static const char* CSV_HEADER      = "FunctionName,MemUsageObs,UniqueAddresses,CountObs,SamplingInterval";
static const char* TIMELINE_HEADER =
    "Time,SamplingInterval,Bin,MemUsageObs,UniqueAddresses,CountObs,FreedBytes,Singletons,Doubletons,Tripletons,"
    "Quadrupletons,Discovered,Seconds";

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

// splitmix64 finaliser: scrambles the structure of addresses (strides,
// alignment) so that selection is independent of how memory is laid out.
static inline UINT64 Mix(UINT64 z)
{
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// With countBytes, bytes are selected in aligned 8-byte words: a word is selected with
// probability 1/i, so selected bytes x i estimates the bytes touched. (Windows select 64-byte
// chunks to measure density, which countBytes does not need.) An access selects if any word it
// covers does. The tests are chosen by access size at instrumentation time and are branch-free,
// so that Pin inlines them: up to 8 bytes cover at most 2 words, up to 16 bytes at most 3; wider
// accesses loop (not inlined, and rare).
#define CHUNK_BITS 6
#define WORD_BITS 3
static inline BOOL ChunkSelected(ADDRINT chunk) { return Mix(chunk ^ salt) < spatialThreshold; }
static inline BOOL WordSelected(ADDRINT word) { return Mix(word ^ salt) < spatialThreshold; }

static inline ADDRINT Words8Selected(ADDRINT ea, UINT32 size)
{
    return (Mix((ea >> WORD_BITS) ^ salt) < spatialThreshold) | (Mix(((ea + size - 1) >> WORD_BITS) ^ salt) < spatialThreshold);
}

static inline ADDRINT Words16Selected(ADDRINT ea, UINT32 size)
{
    return (Mix((ea >> WORD_BITS) ^ salt) < spatialThreshold) | (Mix(((ea >> WORD_BITS) + 1) ^ salt) < spatialThreshold) |
           (Mix(((ea + size - 1) >> WORD_BITS) ^ salt) < spatialThreshold);
}

static BOOL LongAccessSelected(ADDRINT ea, UINT32 size)
{
    for (ADDRINT word = ea >> WORD_BITS; word <= (ea + size - 1) >> WORD_BITS; word++)
        if (WordSelected(word)) return TRUE;
    return FALSE;
}

// spatial: count every reference, record those to selected start addresses (-footprint starts)
static inline ADDRINT PIN_FAST_ANALYSIS_CALL SelectAddress(THREADID tid, ADDRINT ea)
{
    threads[tid].refs++;
    return Mix(ea ^ salt) < spatialThreshold;
}

// spatial, countBytes: count every reference, record those that touch a selected word
static inline ADDRINT PIN_FAST_ANALYSIS_CALL SelectBytes8(THREADID tid, ADDRINT ea, UINT32 size)
{
    threads[tid].refs++;
    return Words8Selected(ea, size);
}

static inline ADDRINT PIN_FAST_ANALYSIS_CALL SelectBytes16(THREADID tid, ADDRINT ea, UINT32 size)
{
    threads[tid].refs++;
    return Words16Selected(ea, size);
}

static ADDRINT PIN_FAST_ANALYSIS_CALL SelectLong(THREADID tid, ADDRINT ea, UINT32 size)
{
    threads[tid].refs++;
    return LongAccessSelected(ea, size);
}

// The selection test for an access of refSize bytes (spatial mode)
static AFUNPTR SpatialSelector(UINT32 refSize)
{
    if (!countBytes) return AFUNPTR(SelectAddress);
    return refSize <= 8 ? AFUNPTR(SelectBytes8) : refSize <= 16 ? AFUNPTR(SelectBytes16) : AFUNPTR(SelectLong);
}

static inline ADDRINT PIN_FAST_ANALYSIS_CALL ShouldSample(THREADID tid, UINT32 rate)
{
    return ((lcg_step(threads[tid].seed) % rate) == 0) ? 1 : 0;
}

/* ===================================================================== */
/* Instrumentation                                                       */
/* ===================================================================== */

static VOID RecordSelected(THREADID tid, ADDRINT address, UINT32 size);
static VOID ApplyRelease(ADDRINT lo, ADDRINT hi);

// The allocator's own code (glibc malloc.c, found by ImageLoad with -track_frees or -window).
// Its accesses are chunk headers and free-list links, which land outside live blocks and
// would never be released, so they count as time but not toward any footprint.
ADDRINT mallocCodeLo = 0, mallocCodeHi = 0;
// The allocator in use is the first loaded library (not the program) that defines malloc. If it is
// not glibc (e.g. jemalloc or tcmalloc, usually preloaded), its whole image counts as allocator code,
// glibc's chunk headers do not apply, and a freed block releases up to the allocator's own
// malloc_usable_size (usableFn), called at free's entry.
BOOL foreignAllocator = FALSE;
ADDRINT usableFn = 0;
const UINT32 MALLOC_REF = ~0u; // buffered size of an allocator reference
const UINT32 ZERO_REF   = 1u << 30; // -skip_zeroing: flag on the buffered size of a store made by memset/bzero code
BOOL skipZeroing = FALSE;
vector< std::pair< ADDRINT, ADDRINT > > zeroCode; // -skip_zeroing: memset/bzero routines (reserved in main)
UINT64 zeroSkipped = 0;                            // bytes of zeroing left out of the footprint
static BOOL InZeroCode(INS ins)
{
    for (const auto& r : zeroCode)
        if (INS_Address(ins) >= r.first && INS_Address(ins) < r.second) return TRUE;
    return FALSE;
}

static BOOL InMallocCode(INS ins) { return INS_Address(ins) >= mallocCodeLo && INS_Address(ins) < mallocCodeHi; }

// spatial: count a reference made by the allocator (never selected)
static VOID PIN_FAST_ANALYSIS_CALL CountReference(THREADID tid) { threads[tid].refs++; }

static VOID InsertRecord(INS ins, UINT32 memOp, UINT32 refSize, BOOL isRead)
{
    if (InMallocCode(ins))
    {
        // time only: buffered modes record the size MALLOC_REF, which BufferFull counts but does not add
        if (mode == SPATIAL)
        {
            INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(CountReference), IARG_FAST_ANALYSIS_CALL, IARG_THREAD_ID, IARG_END);
            return;
        }
        refSize = MALLOC_REF;
    }
    else if (skipZeroing && !isRead && InZeroCode(ins))
        refSize |= ZERO_REF; // BufferFull drops it if it falls in the range the memset call clears
    if (mode == SPATIAL)
    {
        if (countBytes)
            INS_InsertIfCall(ins, IPOINT_BEFORE, SpatialSelector(refSize), IARG_FAST_ANALYSIS_CALL, IARG_THREAD_ID,
                             IARG_MEMORYOP_EA, memOp, IARG_UINT32, refSize, IARG_END);
        else
            INS_InsertIfCall(ins, IPOINT_BEFORE, AFUNPTR(SelectAddress), IARG_FAST_ANALYSIS_CALL, IARG_THREAD_ID,
                             IARG_MEMORYOP_EA, memOp, IARG_END);
        INS_InsertThenCall(ins, IPOINT_BEFORE, AFUNPTR(RecordSelected), IARG_THREAD_ID, IARG_MEMORYOP_EA, memOp, IARG_UINT32,
                           refSize, IARG_END);
    }
    else if (mode == SAMPLER)
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

// windowed: count a block's references; TRUE when time should advance
static ADDRINT PIN_FAST_ANALYSIS_CALL CountRefs(THREADID tid, UINT32 n)
{
    ThreadData& t = threads[tid];
    t.refs += n;
    return t.refs >= t.checkAt;
}

static ADDRINT CheckTime(THREADID tid, CONTEXT* ctxt);

// windowed: select 1-in-i 64-byte chunks (every address in a selected chunk; with
// countBytes, 8-byte words), without counting (CountRefs counts). Seeing whole chunks gives the bytes the
// selected accesses cover as well as their footprint.
static inline ADDRINT PIN_FAST_ANALYSIS_CALL SelectWatched(ADDRINT ea) { return ChunkSelected(ea >> CHUNK_BITS); }
static inline ADDRINT PIN_FAST_ANALYSIS_CALL SelectWatched8(ADDRINT ea, UINT32 size) { return Words8Selected(ea, size); }
static inline ADDRINT PIN_FAST_ANALYSIS_CALL SelectWatched16(ADDRINT ea, UINT32 size) { return Words16Selected(ea, size); }
static ADDRINT PIN_FAST_ANALYSIS_CALL SelectWatchedLong(ADDRINT ea, UINT32 size) { return LongAccessSelected(ea, size); }

static UINT32 MemoryReferences(INS ins)
{
    if (!INS_IsStandardMemop(ins) && !INS_HasMemoryVector(ins)) return 0;
    UINT32 n = 0;
    for (UINT32 memOp = 0; memOp < INS_MemoryOperandCount(ins); memOp++)
        n += INS_MemoryOperandIsRead(ins, memOp) + INS_MemoryOperandIsWritten(ins, memOp);
    return n;
}

static VOID TraceWindowed(TRACE trace)
{
    tracesSinceChange++;
    BOOL watch = versioned ? TRACE_Version(trace) == VERSION_WATCH : watching;
    if (versioned)
    {
        INS head = BBL_InsHead(TRACE_BblHead(trace));
        INS_InsertVersionCase(head, watchReg, watch ? 0 : 1, watch ? VERSION_IDLE : VERSION_WATCH, IARG_END);
    }
    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl))
    {
        UINT32 n = 0;
        for (INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins))
            n += MemoryReferences(ins);
        if (n == 0) continue;
        BBL_InsertIfCall(bbl, IPOINT_BEFORE, AFUNPTR(CountRefs), IARG_FAST_ANALYSIS_CALL, IARG_THREAD_ID, IARG_UINT32, n, IARG_END);
        if (versioned)
            BBL_InsertThenCall(bbl, IPOINT_BEFORE, AFUNPTR(CheckTime), IARG_THREAD_ID, IARG_CONTEXT, IARG_RETURN_REGS, watchReg,
                               IARG_END);
        else
            BBL_InsertThenCall(bbl, IPOINT_BEFORE, AFUNPTR(CheckTime), IARG_THREAD_ID, IARG_CONTEXT, IARG_END);
        if (!watch) continue;
        for (INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins))
        {
            if (MemoryReferences(ins) == 0 || InMallocCode(ins)) continue; // counted by CountRefs only
            for (UINT32 memOp = 0; memOp < INS_MemoryOperandCount(ins); memOp++)
            {
                UINT32 refSize = INS_MemoryOperandSize(ins, memOp);
                for (int k = INS_MemoryOperandIsRead(ins, memOp) + INS_MemoryOperandIsWritten(ins, memOp); k > 0; k--)
                {
                    if (!countBytes)
                        INS_InsertIfCall(ins, IPOINT_BEFORE, AFUNPTR(SelectWatched), IARG_FAST_ANALYSIS_CALL, IARG_MEMORYOP_EA,
                                         memOp, IARG_END);
                    else
                        INS_InsertIfCall(ins, IPOINT_BEFORE,
                                         refSize <= 8    ? AFUNPTR(SelectWatched8)
                                         : refSize <= 16 ? AFUNPTR(SelectWatched16)
                                                         : AFUNPTR(SelectWatchedLong),
                                         IARG_FAST_ANALYSIS_CALL, IARG_MEMORYOP_EA, memOp, IARG_UINT32, refSize, IARG_END);
                    INS_InsertThenCall(ins, IPOINT_BEFORE, AFUNPTR(RecordSelected), IARG_THREAD_ID, IARG_MEMORYOP_EA, memOp,
                                       IARG_UINT32, refSize, IARG_END);
                }
            }
        }
    }
}

VOID Trace(TRACE trace, VOID* v)
{
    if (windowed)
    {
        TraceWindowed(trace);
        return;
    }
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
    if (hi <= lo) return;
    if (mode == SPATIAL)
    {
        // spatial accesses are recorded when they happen (not buffered), so a
        // release is applied when it happens too: exact order across threads
        PIN_GetLock(&stateLock, tid + 1);
        if (!finished) ApplyRelease(lo, hi);
        PIN_ReleaseLock(&stateLock);
        return;
    }
    threads[tid].pending.push_back({PIN_GetBufferPointer(ctxt, bufId), lo, hi});
}

/* Windows: live blocks (called with stateLock held) ---------------------- */

static VOID MaybeTriggerWindow(ADDRINT size);

static ADDRINT ChunkHeader(ADDRINT ptr);
static VOID ChunkRange(ADDRINT ptr, ADDRINT size, ADDRINT header, ADDRINT* lo, ADDRINT* hi);

// Usable bytes of a heap block from its glibc chunk header, or `size` if the header does not fit
static ADDRINT UsableSize(ADDRINT ptr, ADDRINT size)
{
    if (!KnobBlockSlack.Value()) return size;
    ADDRINT lo, hi;
    ChunkRange(ptr, size, ChunkHeader(ptr), &lo, &hi);
    return lo == ptr ? size : hi - ptr;
}

static VOID AddBlock(ADDRINT ptr, ADDRINT size, BOOL mapped, BOOL fresh, ADDRINT span = 0)
{
    if (ptr == 0 || size == 0) return;
    blocks[ptr] = {size, mapped, fresh};
    if (span > size) blocks[ptr].span = span;
    MaybeTriggerWindow(size);
}

// After releasing stateLock: re-instrument if a window opened or closed.
static VOID ApplyWindowChange()
{
    PIN_GetLock(&stateLock, 1);
    BOOL flush   = flushPending;
    flushPending = FALSE;
    PIN_ReleaseLock(&stateLock);
    if (flush) PIN_RemoveInstrumentation();
}

static VOID DropBlock(ADDRINT ptr) { blocks.erase(ptr); }

// munmap: drop anonymous mappings in [lo, hi), keeping the parts outside it
static VOID UnmapBlocks(ADDRINT lo, ADDRINT hi)
{
    auto it = blocks.upper_bound(lo);
    if (it != blocks.begin()) --it;
    while (it != blocks.end() && it->first < hi)
    {
        auto next     = std::next(it);
        Block b       = it->second;
        ADDRINT start = it->first, end = start + b.Extent();
        if (b.mapped && end > lo)
        {
            blocks.erase(it);
            if (start < lo) blocks[start] = {lo - start, TRUE, b.fresh};
            if (end > hi) blocks[hi] = {end - hi, TRUE, b.fresh};
        }
        it = next;
    }
}

// Split the selected footprint (accesses in windows so far) between reused
// blocks, fresh blocks and the rest, with the bytes covered in the blocks.
static VOID Attribute()
{
    struct Access
    {
        ADDRINT address;
        UINT32 size;
        Block* block;
        bool operator<(const Access& o) const { return address < o.address; }
    };
    vector< Access > inBlocks, outside;
    otherSampled = 0;
    for (auto& kv : blocks)
        kv.second.sampled = kv.second.covered = 0;
    footprint.ForEach([&inBlocks, &outside](ADDRINT address, UINT32 size) {
        auto it = blocks.upper_bound(address);
        if (it != blocks.begin() && address < std::prev(it)->first + std::prev(it)->second.Extent())
            inBlocks.push_back({address, size, &std::prev(it)->second});
        else if (countBytes)
            outside.push_back({address, size, NULL});
        else
            otherSampled += size;
    });
    // bytes covered: the union of [address, address + size); with countBytes that is also the
    // footprint, so density is 1
    std::sort(inBlocks.begin(), inBlocks.end());
    kindSampled[0] = kindSampled[1] = kindCovered[0] = kindCovered[1] = 0;
    ADDRINT coveredTo = 0;
    for (const Access& a : inBlocks)
    {
        BOOL fresh = a.block->fresh;
        if (!countBytes) kindSampled[fresh] += a.size, a.block->sampled += a.size;
        ADDRINT end = a.address + a.size;
        if (end > coveredTo)
        {
            ADDRINT more = end - std::max(a.address, coveredTo);
            kindCovered[fresh] += more, a.block->covered += more;
            if (countBytes) kindSampled[fresh] += more, a.block->sampled += more;
            coveredTo = end;
        }
    }
    std::sort(outside.begin(), outside.end());
    coveredTo = 0;
    for (const Access& a : outside)
    {
        ADDRINT end = a.address + a.size;
        if (end > coveredTo) otherSampled += end - std::max(a.address, coveredTo), coveredTo = end;
    }
}

// Pagemap entries of pages [cachedFirst, cachedFirst + pageBits.size()), read
// in chunks so that blocks visited in address order share reads.
ADDRINT cachedFirst = 0;

static VOID ForgetResidency() { pageBits.clear(); }

// Bytes of [lo, hi) on pages that have been touched (are resident or swapped).
// With `written`, also marks the pages written since the last soft-dirty clear
// (pagemap bit 55) and returns in *writtenBytes the bytes on pages marked so far.
// With `edges`, also adds the number of places inside [lo, hi) where a touched page (written, with
// `written`; else resident) borders an untouched one: the touched page there is probably partly used.
static UINT64 Resident(ADDRINT lo, ADDRINT hi, vector< UINT8 >* written = NULL, UINT64* writtenBytes = NULL,
                       UINT64* edges = NULL)
{
    const ADDRINT page = 1 << PAGE_BITS, chunk = 512;
    ADDRINT first = lo >> PAGE_BITS, last = (hi - 1) >> PAGE_BITS;
    if (first < cachedFirst || last >= cachedFirst + pageBits.size())
    {
        cachedFirst = first;
        pageBits.resize(std::max(last - first + 1, chunk));
        ssize_t want = pageBits.size() * sizeof(UINT64);
        ssize_t got  = -1;
        if (lseek(pagemapFd, first * sizeof(UINT64), SEEK_SET) >= 0) got = read(pagemapFd, pageBits.data(), want);
        // past the end of the address space the read is short
        pageBits.resize(got > 0 ? got / sizeof(UINT64) : 0);
        if (last >= cachedFirst + pageBits.size())
        {
            if (writtenBytes) *writtenBytes = hi - lo;
            return hi - lo;
        }
    }
    if (written) written->resize(last - first + 1, 0);
    UINT64 bytes = 0, wbytes = 0;
    int prev = -1;
    for (ADDRINT p = first; p <= last; p++)
    {
        UINT64 bits = pageBits[p - cachedFirst];
        ADDRINT a = p << PAGE_BITS, overlap = std::min(hi, a + page) - std::max(lo, a);
        int touched = (bits >> 62) != 0; // bit 63: present, bit 62: swapped
        if (written)
        {
            UINT8& w = (*written)[p - first];
            if ((bits >> 55) & 1) w = 1;
            if (w) wbytes += overlap;
            touched = w;
        }
        if (bits >> 62) bytes += overlap;
        if (edges && prev >= 0 && touched != prev) ++*edges;
        prev = touched;
    }
    if (writtenBytes) *writtenBytes = wbytes;
    return bytes;
}

// Record every reused block's written pages (soft-dirty since the last clear).
static VOID Harvest()
{
    ForgetResidency();
    for (auto& kv : blocks)
    {
        Block& b = kv.second;
        if (b.fresh || b.allWritten) continue;
        UINT64 w = 0;
        Resident(kv.first, kv.first + b.Extent(), &b.written, &w);
        b.allWritten = w == b.Extent();
    }
    ForgetResidency();
}

static VOID ClearDirty()
{
    if (write(clearRefsFd, "4", 1) != 1) clearRefsFd = -1;
}

static VOID HarvestAndClear()
{
    Harvest();
    ClearDirty();
}

// Allocations of at least this size clear the soft-dirty bits when the
// allocator is entered, so that writes made to the new block's pages by
// earlier blocks are not counted as its own; smaller blocks share pages with
// other blocks anyway.
#define DIRTY_CLEAR_MIN (16 * 1024)

// The bytes a freed block releases: its whole glibc chunk, [ptr - 8, ptr - 8 + chunk size),
// read from the chunk's size field (ptr - 8) before the allocator changes it. That is the
// block, the size field and the slack after the requested size, which string and copy
// functions read past the end of a block. The prev_size field below belongs to the
// previous chunk. Without glibc's allocator, or if the field does not fit a chunk of this
// size (an mmapped chunk, another allocator), only [ptr, ptr + size).
static ADDRINT ChunkHeader(ADDRINT ptr)
{
    ADDRINT header = 0;
    if (foreignAllocator || mallocCodeHi == 0 || ptr < 8 || PIN_SafeCopy(&header, (VOID*)(ptr - 8), sizeof(header)) != sizeof(header)) return 0;
    return header;
}

static VOID ChunkRange(ADDRINT ptr, ADDRINT size, ADDRINT header, ADDRINT* lo, ADDRINT* hi)
{
    ADDRINT chunk = header & ~(ADDRINT)7;
    BOOL ok       = header && !(header & 2) && chunk % 16 == 0 && chunk >= 32 && chunk >= size + 8 && chunk < size + 8 + 32;
    *lo = ok ? ptr - 8 : ptr;
    *hi = ok ? ptr - 8 + chunk : ptr + size;
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

/* -sites ------------------------------------------------------------------ */

// A block is allocated; called again for the same block (a nested call's exit, then the outer
// call's), it changes nothing.
static VOID SiteBorn(THREADID tid, ADDRINT ptr, ADDRINT size, ADDRINT grownFrom = 0)
{
    if (!sitesOn || ptr == 0) return;
    ThreadData& t = threads[tid];
    PIN_GetLock(&blockSiteLock, tid + 1);
    auto it = siteLive.find(ptr);
    if (it == siteLive.end() || it->second.size != size)
    {
        SiteBlock b;
        b.site = t.allocSite, b.siteId = t.allocSiteId, b.size = size, b.born = timeNow, b.kind = t.allocKind;
        b.grownFrom = grownFrom;
        b.ctxN      = t.allocCtxN;
        for (UINT32 i = 0; i < t.allocCtxN; i++)
            b.ctx[i] = t.allocCtx[i];
        if (t.allocCapSet)
        {
            // the same container growing: the key's latest chain's buffer is still live, and smaller
            auto key = std::make_pair(t.allocCap, t.allocCapCtx);
            auto it  = capLatest.find(key);
            UINT32 c = ~0u;
            if (it != capLatest.end())
            {
                CapChain& last = capChains[it->second];
                if (siteLive.count(last.lastPtr) && size > last.finalSize) c = it->second;
            }
            if (c == ~0u)
            {
                c = capChains.size();
                capChains.push_back(CapChain());
                capChains[c].key = t.allocCap, capChains[c].ctx = t.allocCapCtx, capChains[c].elem = t.allocCapElem;
                capLatest[key] = c;
            }
            CapChain& ch = capChains[c];
            ch.lastPtr = ptr, ch.blocks++, ch.finalSize = size;
            b.chain = c;
        }
        siteLive[ptr] = b;
    }
    PIN_ReleaseLock(&blockSiteLock);
}

// A block is freed (or moved by realloc): summarized when its release is applied (SiteRelease)
static VOID SiteDies(THREADID tid, ADDRINT ptr, BOOL moved)
{
    if (!sitesOn || ptr == 0) return;
    PIN_GetLock(&blockSiteLock, tid + 1);
    auto it = siteLive.find(ptr);
    if (it != siteLive.end())
    {
        it->second.moved = moved;
        dying[ptr]       = it->second;
        siteLive.erase(it);
    }
    PIN_ReleaseLock(&blockSiteLock);
}

// realloc in place: the block goes on with its new size
static VOID SiteResize(THREADID tid, ADDRINT ptr, ADDRINT size)
{
    if (!sitesOn) return;
    PIN_GetLock(&blockSiteLock, tid + 1);
    auto it = siteLive.find(ptr);
    if (it != siteLive.end()) it->second.size = size;
    PIN_ReleaseLock(&blockSiteLock);
}

// "image+0xaddress" (the address in the image file, for addr2line) for a return address, "id:0x..." for an id from the MemPrintResize pass
static string SiteKey(ADDRINT site, BOOL isId)
{
    ostringstream s;
    if (isId)
    {
        s << "id:0x" << std::hex << site;
        return s.str();
    }
    for (const ImageRange& img : images)
        if (site >= img.lo && site < img.hi)
        {
            size_t slash = img.name.rfind('/');
            s << (slash == string::npos ? img.name : img.name.substr(slash + 1)) << "+0x" << std::hex << site - img.bias;
            return s.str();
        }
    s << "0x" << std::hex << site;
    return s.str();
}

// Fold a dead (or, at the end, live) block into its site's totals. stateLock and blockSiteLock held.
static VOID SiteSummarize(ADDRINT ptr, const SiteBlock& b, UINT64 died)
{
    UINT64 touched = 0, pagesTouched = 0;
    ADDRINT highWater = 0;
    footprint.Summarize(ptr, ptr + b.size, &touched, &highWater, &pagesTouched);
    UINT64 pages  = b.size ? ((ptr + b.size - 1) >> PAGE_BITS) - (ptr >> PAGE_BITS) + 1 : 0;
    if (b.chain != ~0u && capChains[b.chain].lastPtr == ptr && capChains[b.chain].finalSize == b.size)
        capChains[b.chain].finalHighWater = highWater;
    SiteStats& s  = siteStats[{b.site, b.siteId}];
    s.kinds |= b.kind;
    s.count++, s.requested += b.size, s.requestedMax = std::max< UINT64 >(s.requestedMax, b.size);
    s.touched += touched, s.highWater += highWater, s.pagesTouched += pagesTouched, s.pages += pages;
    s.neverTouched += touched == 0;
    s.allocTouched += b.allocTouched, s.moves += b.moved, s.grown += b.grownFrom != 0;
    s.lifetime += died > b.born ? died - b.born : 0;
    double ratio = b.size ? double(highWater) / b.size : 0;
    s.maxHighWaterRatio = std::max(s.maxHighWaterRatio, ratio);
    s.minHighWaterRatio = std::min(s.minHighWaterRatio, ratio);
    if (b.size < KnobSitesBlockMin.Value() || !blocksFile.is_open()) return;
    blocksFile << SiteKey(b.site, b.siteId) << ',' << (UINT32)b.kind << ',' << b.size << ',' << b.born << ',' << died
                   << ',' << touched << ',' << highWater << ',' << pagesTouched << ',' << pages << ',' << b.allocTouched
                   << ',' << b.grownFrom << ',' << (b.moved ? 1 : 0) << ',';
    for (UINT32 i = 0; i < b.ctxN; i++)
        blocksFile << (i ? "|" : "") << SiteKey(b.ctx[i], FALSE);
    // the touched page runs (first page:pages from the block's first page), for shaped reservations
    vector< std::pair< UINT64, UINT64 > > runs;
    footprint.Runs(ptr, ptr + b.size, runs, 256);
    blocksFile << ',';
    for (size_t i = 0; i < runs.size(); i++)
        blocksFile << (i ? ";" : "") << runs[i].first << ':' << runs[i].second;
    blocksFile << '\n';
}

// A release [lo, hi) is about to clear the footprint's bits: summarize the freed block it holds.
// The block starts at lo, or 8 bytes in when [lo, hi) is its whole glibc chunk (ChunkRange).
// stateLock held.
static VOID SiteRelease(ADDRINT lo, ADDRINT hi)
{
    if (!sitesOn) return;
    PIN_GetLock(&blockSiteLock, 1);
    for (ADDRINT ptr : {lo, lo + 8})
    {
        auto it = dying.find(ptr);
        if (it == dying.end() || ptr >= hi) continue;
        SiteSummarize(ptr, it->second, timeNow);
        dying.erase(it);
        break;
    }
    PIN_ReleaseLock(&blockSiteLock);
}

// calloc's exit, applied in order with the thread's references: the bytes touched so far in the
// block were touched by calloc itself (zeroing a reused chunk). stateLock held.
static VOID SiteProbe(ADDRINT ptr)
{
    PIN_GetLock(&blockSiteLock, 1);
    auto it = siteLive.find(ptr);
    if (it != siteLive.end())
    {
        UINT64 touched = 0, pagesTouched = 0;
        ADDRINT highWater = 0;
        footprint.Summarize(ptr, ptr + it->second.size, &touched, &highWater, &pagesTouched);
        it->second.allocTouched = touched;
    }
    PIN_ReleaseLock(&blockSiteLock);
}

// End of the run: summarize the blocks still live (and frees whose release was never applied),
// then write *_sites.csv. stateLock held.
static VOID SitesFinish()
{
    if (!sitesOn) return;
    PIN_GetLock(&blockSiteLock, 1);
    for (const auto& e : siteLive)
        SiteSummarize(e.first, e.second, timeNow);
    for (const auto& e : dying)
        SiteSummarize(e.first, e.second, timeNow);
    siteLive.clear(), dying.clear();
    PIN_ReleaseLock(&blockSiteLock);
    if (blocksFile.is_open()) blocksFile.close();
    if (!capChains.empty())
    {
        // per capacity key: its chains; the small ones (final buffer below -sites_block_min) only as a
        // count and the largest final buffer, the others one by one (final buffer's size and high water)
        struct Agg { ADDRINT elem = 0; UINT64 chains = 0, small = 0, smallMax = 0; vector< std::pair< UINT64, UINT64 > > large; };
        std::map< std::pair< ADDRINT, BOOL >, Agg > agg;
        for (const CapChain& c : capChains)
        {
            Agg& a = agg[{c.key, c.ctx}];
            a.elem = c.elem, a.chains++;
            if (c.finalSize < KnobSitesBlockMin.Value())
                a.small++, a.smallMax = std::max(a.smallMax, c.finalSize);
            else if (a.large.size() < 1000)
                a.large.push_back({c.finalSize, c.finalHighWater});
        }
        ofstream cap(outputFileNamePrefix + "_capacity.csv");
        cap << "key,elem,chains,small_chains,small_max_size,final_sizes,final_high_waters\n";
        for (const auto& e : agg)
        {
            ostringstream key;
            key << (e.first.second ? "cap:0x" : "id:0x") << std::hex << e.first.first;
            cap << key.str() << ',' << e.second.elem << ',' << e.second.chains << ',' << e.second.small << ','
                << e.second.smallMax << ',';
            for (size_t i = 0; i < e.second.large.size(); i++)
                cap << (i ? ";" : "") << e.second.large[i].first;
            cap << ',';
            for (size_t i = 0; i < e.second.large.size(); i++)
                cap << (i ? ";" : "") << e.second.large[i].second;
            cap << '\n';
        }
    }
    {
        // the images the site keys name (image+0x<address in the file>), with their paths: an
        // address key is only meaningful for that build (sites.py records each one's build-id)
        ofstream img(outputFileNamePrefix + "_images.csv");
        img << "name,path\n";
        for (const ImageRange& i : images)
            img << i.name.substr(i.name.rfind('/') + 1) << ',' << i.name << '\n';
    }
    ofstream out(outputFileNamePrefix + "_sites.csv");
    out << "site,kinds,count,requested,requested_max,touched,high_water,pages_touched,pages,never_touched,"
           "alloc_touched,moves,grown,lifetime,max_high_water_ratio,min_high_water_ratio"
        << endl;
    for (const auto& e : siteStats)
    {
        const SiteStats& s = e.second;
        out << SiteKey(e.first.first, e.first.second) << ',' << s.kinds << ',' << s.count << ',' << s.requested << ','
            << s.requestedMax << ',' << s.touched << ',' << s.highWater << ',' << s.pagesTouched << ',' << s.pages << ','
            << s.neverTouched << ',' << s.allocTouched << ',' << s.moves << ',' << s.grown << ',' << s.lifetime << ','
            << s.maxHighWaterRatio << ',' << (s.minHighWaterRatio > 1 ? 0 : s.minHighWaterRatio) << endl;
    }
}

static VOID RememberAllocation(THREADID tid, ADDRINT ptr, ADDRINT size)
{
    if (ptr == 0) return;
    SiteBorn(tid, ptr, size);
    PIN_GetLock(&allocLock, 1);
    allocations[ptr] = size;
    PIN_ReleaseLock(&allocLock);
    if (windowed)
    {
        PIN_GetLock(&stateLock, tid + 1);
        if (!finished) AddBlock(ptr, size, FALSE, threads[tid].allocFresh, UsableSize(ptr, size));
        PIN_ReleaseLock(&stateLock);
        ApplyWindowChange();
    }
}

// windowed: realloc in place; FALSE if the block is not tracked (or not windowed)
static BOOL ResizeBlock(THREADID tid, ADDRINT ptr, ADDRINT size)
{
    if (!windowed || size == 0) return FALSE;
    PIN_GetLock(&stateLock, tid + 1);
    auto it  = blocks.find(ptr);
    BOOL got = !finished && it != blocks.end() && !threads[tid].allocFresh;
    if (got) // the page bitmap follows at the next scan
    {
        it->second.size = size, it->second.allWritten = FALSE;
        ADDRINT usable  = UsableSize(ptr, size);
        it->second.span = usable > size ? usable : 0;
    }
    PIN_ReleaseLock(&stateLock);
    return got;
}

static VOID ForgetBlock(THREADID tid, ADDRINT ptr)
{
    if (!windowed) return;
    PIN_GetLock(&stateLock, tid + 1);
    if (!finished) DropBlock(ptr);
    PIN_ReleaseLock(&stateLock);
}

// Entry analysis: only the outermost allocator call of a thread is tracked,
// so allocator functions calling each other are not counted twice. A nested
// call is made from inside one of the instrumented allocation functions and
// runs with a lower stack pointer than the outermost call. Any other entry
// starts a new outermost call: the previous one left without its exit being
// seen. glibc's calloc does that the first time the allocator is used: it
// calls malloc through an initialization function and then jumps to memset
// instead of returning. So when a call nested in a malloc-like call (malloc,
// calloc, memalign, ...) returns a block, the block is recorded at once with
// the outer call's size; if the outer call's exit is seen, it records the same
// block again, which changes nothing.
struct CodeRange
{
    ADDRINT lo, hi;
};
vector< CodeRange > allocatorCode; // instrumented allocation functions (reserved, never reallocated)

static BOOL InAllocator(ADDRINT pc)
{
    for (const CodeRange& r : allocatorCode)
        if (pc >= r.lo && pc < r.hi) return TRUE;
    return FALSE;
}

static VOID NoteReturnSite(ADDRINT ret, int kind);

static VOID AllocEnter(THREADID tid, ADDRINT sp, ADDRINT caller, BOOL mallocLike, ADDRINT size, ADDRINT ptr, ADDRINT out)
{
    ThreadData& t = threads[tid];
    if (t.allocDepth > 0 && (sp >= t.allocSp || !InAllocator(caller))) t.allocDepth = 0;
    if (t.allocDepth++ > 0) return;
    t.allocSp         = sp;
    t.allocMallocLike = mallocLike;
    t.allocSite       = t.pendingSiteSet ? t.pendingSite : caller;
    t.allocSiteId     = t.pendingSiteSet;
    // the caller frames: above the __mpr_* call (pass builds), or above this call (SiteContext)
    t.allocCtxN = t.pendingSiteSet ? t.pendingCtxN : (t.scratchFor == caller ? t.scratchCtxN : 0);
    for (UINT32 i = 0; i < t.allocCtxN; i++)
        t.allocCtx[i] = t.pendingSiteSet ? t.pendingCtx[i] : t.scratchCtx[i];
    t.scratchFor = 0;
    t.allocCapSet = t.pendingCapSet, t.allocCap = t.pendingCap, t.allocCapCtx = t.pendingCapCtx;
    t.allocCapElem  = t.pendingCapElem;
    t.pendingCapSet = FALSE;
    t.pendingSiteSet  = FALSE;
    t.allocKind       = mallocLike ? SITE_MALLOC : (out ? SITE_MEMALIGN : SITE_REALLOC);
    t.allocFresh      = FALSE;
    // At the entry, before calloc zeroes or realloc copies the new block's memory: see DIRTY_CLEAR_MIN.
    if (clearRefsFd >= 0 && size >= DIRTY_CLEAR_MIN)
    {
        PIN_GetLock(&stateLock, tid + 1);
        if (!finished && clearRefsFd >= 0) HarvestAndClear();
        PIN_ReleaseLock(&stateLock);
    }
    t.allocSize   = size;
    t.allocPtr    = ptr;
    t.allocHeader = ptr ? ChunkHeader(ptr) : 0; // realloc: the old chunk, before it changes
    t.allocOut  = out;
    if (foreignAllocator) NoteReturnSite(caller, mallocLike ? 1 : out ? 3 : 2);
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
    ThreadData& t = threads[tid];
    if (AllocLeave(tid))
        RememberAllocation(tid, ret, t.allocSize);
    else if (t.allocDepth > 0 && t.allocMallocLike && ret)
        RememberAllocation(tid, ret, t.allocSize); // in case the outermost call's exit is not seen
}

// calloc(n, size)
static VOID CallocEnter(THREADID tid, ADDRINT sp, ADDRINT caller, BOOL mallocLike, ADDRINT n, ADDRINT size)
{
    AllocEnter(tid, sp, caller, mallocLike, n * size, 0, 0);
    if (threads[tid].allocDepth == 1) threads[tid].allocKind = SITE_CALLOC;
}

// __mpr_alloc(size, site, ...) and the other entry points of libmpr, called by a program built with
// the MemPrintResize pass: the site id it passes is the next block's site (libmpr then calls glibc's
// allocator, or maps the block itself, which -sites does not see)
static VOID Backtrace(const CONTEXT* ctxt, ADDRINT ret, ADDRINT* out, UINT32* n);

static VOID MprEnter(THREADID tid, const CONTEXT* ctxt, ADDRINT ret, ADDRINT site, ADDRINT a, ADDRINT b)
{
    ThreadData& t    = threads[tid];
    t.pendingSite    = site;
    t.pendingSiteSet = TRUE;
    t.pendingCtxN    = 0;
    if (KnobSiteContext.Value() && a * (b ? b : 1) >= KnobSitesBlockMin.Value()) Backtrace(ctxt, ret, t.pendingCtx, &t.pendingCtxN);
}

// The -site_context frames above a call whose return address is `ret`: the application's backtrace
// (libunwind, from the call frame information, so frame pointers are not needed) from the frame
// after `ret`. libmpr takes the same frames with glibc's backtrace() at run time.
static VOID Backtrace(const CONTEXT* ctxt, ADDRINT ret, ADDRINT* out, UINT32* n)
{
    void* frames[16];
    PIN_LockClient();
    int got = PIN_Backtrace(ctxt, frames, 16);
    PIN_UnlockClient();
    *n      = 0;
    UINT32 want = std::min< UINT32 >(KnobSiteContext.Value(), 4);
    for (int i = 0; i < got; i++)
        if ((ADDRINT)frames[i] == ret)
        {
            for (int j = i + 1; j < got && *n < want; j++)
                out[(*n)++] = (ADDRINT)frames[j];
            return;
        }
}

// -sites, before an allocation function's entry: the caller frames of a large request, for
// AllocEnter's outermost call (taken only for blocks -sites writes one by one)
// __mpr_growth(site): the program's call site growing a container (0: none)
static VOID GrowthEnter(THREADID tid, ADDRINT site) { threads[tid].growthSite = site; }

// __mpr_capacity(len, size, elem, point): the next allocation is that container's new buffer
static VOID CapacityEnter(THREADID tid, ADDRINT elem, ADDRINT point)
{
    ThreadData& t     = threads[tid];
    t.pendingCapCtx   = t.growthSite != 0;
    t.pendingCap      = t.growthSite ? t.growthSite : point;
    t.pendingCapElem  = elem;
    t.pendingCapSet   = TRUE;
}

static BOOL InAllocator(ADDRINT pc);

static VOID SiteContext(THREADID tid, const CONTEXT* ctxt, ADDRINT sp, ADDRINT ret, ADDRINT a, ADDRINT b)
{
    ThreadData& t = threads[tid];
    // outermost: as AllocEnter decides (a depth left by a call whose exit was missed does not count)
    BOOL nested = t.allocDepth > 0 && sp < t.allocSp && InAllocator(ret);
    if (nested || t.pendingSiteSet || a * (b ? b : 1) < KnobSitesBlockMin.Value()) return;
    Backtrace(ctxt, ret, t.scratchCtx, &t.scratchCtxN);
    t.scratchFor = ret;
}

// operator new(size, ...): as malloc
static VOID NewEnter(THREADID tid, ADDRINT sp, ADDRINT caller, BOOL mallocLike, ADDRINT size, ADDRINT ptr, ADDRINT out)
{
    AllocEnter(tid, sp, caller, TRUE, size, ptr, out);
    if (threads[tid].allocDepth == 1) threads[tid].allocKind = SITE_NEW;
}

// calloc's exit: as malloc's, and (-sites) a look at the block once the references calloc made
// (zeroing a reused chunk) have been counted
static VOID CallocExit(THREADID tid, CONTEXT* ctxt, ADDRINT ret)
{
    BOOL outer = threads[tid].allocDepth == 1;
    MallocExit(tid, ret);
    if (!sitesOn || !outer || ret == 0) return;
    if (mode == SPATIAL)
    {
        PIN_GetLock(&stateLock, tid + 1);
        if (!finished) SiteProbe(ret);
        PIN_ReleaseLock(&stateLock);
        return;
    }
    Release probe;
    probe.position = PIN_GetBufferPointer(ctxt, bufId), probe.lo = ret, probe.hi = ret + 1, probe.probe = TRUE;
    threads[tid].pending.push_back(probe);
}

// -skip_zeroing: a memset(p, c, n) / bzero(p, n) is entered. In order with the thread's buffered
// references, the range its stores clear is set: [p, p + n) when it writes zeros over at least 4 KB,
// else empty. Stores made by memset code (ZERO_REF) into that range do not touch the footprint.
static VOID ZeroEnter(THREADID tid, const CONTEXT* ctxt, ADDRINT p, ADDRINT c, ADDRINT n)
{
    ThreadData& t = threads[tid];
    BOOL clears   = (c & 0xff) == 0 && n >= 4096;
    if (!clears && !t.zeroQueued) return; // the range is empty already
    Release z;
    z.position = PIN_GetBufferPointer(const_cast< CONTEXT* >(ctxt), bufId), z.zero = TRUE;
    z.lo = p, z.hi = clears ? p + n : p;
    t.pending.push_back(z);
    t.zeroQueued = clears;
}

static VOID BzeroEnter(THREADID tid, const CONTEXT* ctxt, ADDRINT p, ADDRINT n) { ZeroEnter(tid, ctxt, p, 0, n); }

// posix_memalign(&out, align, size)
static VOID PosixMemalignExit(THREADID tid, ADDRINT ret)
{
    if (!AllocLeave(tid) || ret != 0) return;
    ADDRINT ptr = 0;
    PIN_SafeCopy(&ptr, (VOID*)threads[tid].allocOut, sizeof(ptr));
    RememberAllocation(tid, ptr, threads[tid].allocSize);
}

// free(ptr): released at entry, because glibc's free leaves through a tail
// jump that Pin cannot see. Free's own bookkeeping writes into the block
// (free-list links) come after that, but they are made by the allocator's
// code, which no footprint counts (InMallocCode).
static VOID FreeEnter(THREADID tid, CONTEXT* ctxt, ADDRINT ptr)
{
    ADDRINT size = ForgetAllocation(ptr), lo, hi;
    ForgetBlock(tid, ptr);
    SiteDies(tid, ptr, FALSE);
    ChunkRange(ptr, size, size ? ChunkHeader(ptr) : 0, &lo, &hi);
    if (foreignAllocator && usableFn && ptr)
    {
        // the bytes the allocator gave the block, which string and copy functions read past `size`;
        // also the size of a block whose allocation was missed (an exit through a path not hooked)
        size_t usable = 0;
        PIN_CallApplicationFunction(ctxt, tid, CALLINGSTD_DEFAULT, AFUNPTR(usableFn), NULL, PIN_PARG(size_t), &usable,
                                    PIN_PARG(void*), (void*)ptr, PIN_PARG_END());
        if (usable > size && usable < size + (1 << 20)) lo = ptr, hi = ptr + usable;
    }
    QueueRelease(tid, ctxt, lo, hi);
}

// realloc(ptr, size): releases the old block if it moved, else the tail it shrank by.
static VOID ReallocExit(THREADID tid, CONTEXT* ctxt, ADDRINT ret)
{
    if (!AllocLeave(tid)) return;
    ADDRINT ptr = threads[tid].allocPtr, size = threads[tid].allocSize;
    if (ptr == 0)
    {
        RememberAllocation(tid, ret, size);
        return;
    }
    if (ret == 0 && size != 0) return; // failed: the old block is untouched
    ADDRINT oldSize = ForgetAllocation(ptr);
    if (ret == ptr)
        SiteResize(tid, ptr, size);
    else
    {
        SiteDies(tid, ptr, ret != 0);
        SiteBorn(tid, ret, size, oldSize);
    }
    if (ret == ptr && ResizeBlock(tid, ptr, size))
    {
        // in place: the block keeps the pages it has written
        QueueRelease(tid, ctxt, ptr + size, ptr + oldSize);
        PIN_GetLock(&allocLock, 1);
        allocations[ptr] = size;
        PIN_ReleaseLock(&allocLock);
        return;
    }
    ForgetBlock(tid, ptr);
    if (ret == ptr)
        QueueRelease(tid, ctxt, ptr + size, ptr + oldSize);
    else
    {
        ADDRINT lo, hi; // moved: the old chunk is freed (header read at the entry)
        ChunkRange(ptr, oldSize, threads[tid].allocHeader, &lo, &hi);
        QueueRelease(tid, ctxt, lo, hi);
    }
    RememberAllocation(tid, ret, size);
}

// Calls f(name, value, size, dynamic) for every defined function symbol of an ELF file, from its
// symbol table (.symtab) and its dynamic symbol table (.dynsym; dynamic = TRUE).
template < typename F > static VOID ElfFunctions(const string& file, F f)
{
    int fd = open(file.c_str(), O_RDONLY);
    if (fd < 0) return;
    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_size > (off_t)sizeof(Elf64_Ehdr))
    {
        vector< char > data(st.st_size);
        if (read(fd, data.data(), data.size()) == (ssize_t)data.size() && memcmp(data.data(), ELFMAG, SELFMAG) == 0)
        {
            const Elf64_Ehdr* eh = (const Elf64_Ehdr*)data.data();
            const Elf64_Shdr* sh = (const Elf64_Shdr*)(data.data() + eh->e_shoff);
            for (int i = 0; i < eh->e_shnum; i++)
            {
                if ((sh[i].sh_type != SHT_DYNSYM && sh[i].sh_type != SHT_SYMTAB) || sh[i].sh_link >= eh->e_shnum) continue;
                const Elf64_Sym* sym = (const Elf64_Sym*)(data.data() + sh[i].sh_offset);
                const char* str      = data.data() + sh[sh[i].sh_link].sh_offset;
                for (size_t k = 0; k < sh[i].sh_size / sizeof(Elf64_Sym); k++)
                    if (sym[k].st_shndx != SHN_UNDEF && ELF64_ST_TYPE(sym[k].st_info) == STT_FUNC)
                        f(str + sym[k].st_name, sym[k].st_value, sym[k].st_size, sh[i].sh_type == SHT_DYNSYM);
            }
        }
    }
    close(fd);
}

// Value of a defined function `name` in the dynamic symbol table (.dynsym) of an ELF file, 0 if absent.
static ADDRINT DynamicSymbol(const string& file, const char* name)
{
    ADDRINT value = 0;
    ElfFunctions(file, [&](const char* n, ADDRINT v, ADDRINT, BOOL dynamic) {
        if (dynamic && !value && strcmp(n, name) == 0) value = v;
    });
    return value;
}

// Sizes of the allocator's functions by address, from its ELF symbols. Pin may not know the
// allocator's symbols (jemalloc's), and a routine it creates then runs on to the next symbol it
// knows, over other functions (memalign over realloc): its returns are not all memalign's.
std::map< ADDRINT, ADDRINT > functionSizes;

// End of the function at the start of `rtn`.
static ADDRINT RoutineEnd(RTN rtn)
{
    auto it = functionSizes.find(RTN_Address(rtn));
    if (it == functionSizes.end() || it->second == 0) return RTN_Address(rtn) + RTN_Size(rtn);
    return RTN_Address(rtn) + std::min(it->second, (ADDRINT)RTN_Size(rtn));
}

// Hooks `exit` (with `ifCall` as condition, if given) before each return of the function at the start of `rtn`.
static VOID HookReturns(RTN rtn, AFUNPTR ifCall, AFUNPTR exit, IARGLIST exitArgs)
{
    ADDRINT end = RoutineEnd(rtn);
    for (INS ins = RTN_InsHead(rtn); INS_Valid(ins) && INS_Address(ins) < end; ins = INS_Next(ins))
        if (INS_IsRet(ins))
        {
            if (ifCall)
            {
                INS_InsertIfCall(ins, IPOINT_BEFORE, ifCall, IARG_THREAD_ID, IARG_REG_VALUE, REG_STACK_PTR, IARG_END);
                INS_InsertThenCall(ins, IPOINT_BEFORE, exit, IARG_THREAD_ID, IARG_IARGLIST, exitArgs, IARG_END);
            }
            else
                INS_InsertCall(ins, IPOINT_BEFORE, exit, IARG_THREAD_ID, IARG_IARGLIST, exitArgs, IARG_END);
        }
}

// The routine named `name`. Pin keeps one name per address and may not list a library's exported
// aliases at all (jemalloc's malloc, which its own symbol table calls je_malloc), so the image's
// symbols and then its .dynsym are searched; a routine is created at the address if Pin has none.
static RTN FindRoutine(IMG img, const char* name)
{
    auto real = [](RTN r) { return RTN_Valid(r) && SEC_Name(RTN_Sec(r)).find(".plt") != 0; }; // not a PLT stub
    RTN rtn   = RTN_FindByName(img, name);
    if (real(rtn)) return rtn;
    for (SYM sym = IMG_RegsymHead(img); SYM_Valid(sym); sym = SYM_Next(sym))
        if (SYM_Name(sym) == name && real(RTN_FindByAddress(SYM_Address(sym)))) return RTN_FindByAddress(SYM_Address(sym));
    if (IMG_IsMainExecutable(img)) return RTN_Invalid();
    ADDRINT value = DynamicSymbol(IMG_Name(img), name);
    if (!value) return RTN_Invalid();
    ADDRINT address = IMG_LoadOffset(img) + value;
    rtn             = RTN_FindByAddress(address);
    if (RTN_Valid(rtn) && RTN_Address(rtn) == address) return rtn;
    return RTN_CreateAt(address, name);
}

// Hook the exit of every function that `rtn` jumps to (not calls) inside the allocator's code.
// A return instruction executing with the stack pointer of the outermost allocator call's entry
// returns to that call's caller.
static ADDRINT AtOuterReturn(THREADID tid, ADDRINT sp)
{
    return threads[tid].allocDepth == 1 && sp == threads[tid].allocSp;
}

// The exit of the outermost call, through a function that several allocation functions jump to
// (jemalloc's realloc and malloc share one): the exit of the kind of call that was entered.
static VOID TailExit(THREADID tid, CONTEXT* ctxt, ADDRINT ret)
{
    ThreadData& t = threads[tid];
    if (t.allocMallocLike)
        MallocExit(tid, ret);
    else if (t.allocOut)
        PosixMemalignExit(tid, ret);
    else
        ReallocExit(tid, ctxt, ret);
}

static IARGLIST Args(std::initializer_list< std::pair< IARG_TYPE, UINT32 > > args);

static VOID HookTailJumps(RTN rtn, int depth, std::set< ADDRINT >& instrumented)
{
    static IARGLIST tailExitArgs = Args({{IARG_CONTEXT, 0}, {IARG_FUNCRET_EXITPOINT_VALUE, 0}});
    if (depth == 0) return;
    vector< ADDRINT > targets;
    RTN_Open(rtn);
    ADDRINT end = RoutineEnd(rtn);
    for (INS ins = RTN_InsHead(rtn); INS_Valid(ins) && INS_Address(ins) < end; ins = INS_Next(ins))
        if (INS_IsDirectControlFlow(ins) && !INS_IsCall(ins))
        {
            ADDRINT t = INS_DirectControlFlowTargetAddress(ins);
            if ((t < RTN_Address(rtn) || t >= end) && t >= mallocCodeLo && t < mallocCodeHi)
                targets.push_back(t);
        }
    RTN_Close(rtn);
    for (ADDRINT t : targets)
    {
        RTN target = RTN_FindByAddress(t);
        if (!RTN_Valid(target) || RTN_Address(target) != t || !instrumented.insert(t).second) continue;
        RTN_Open(target);
        if (allocatorCode.size() < allocatorCode.capacity())
            allocatorCode.push_back({RTN_Address(target), RoutineEnd(target)});
        // Only a return to the outermost call's caller is that call's exit: the function is also called
        // normally from inside the allocator (e.g. by realloc), where its return exits nothing.
        HookReturns(target, AFUNPTR(AtOuterReturn), AFUNPTR(TailExit), tailExitArgs);
        RTN_Close(target);
        HookTailJumps(target, depth - 1, instrumented);
    }
}

static VOID InstrumentAllocator(IMG img, const char* name, AFUNPTR enter, IARGLIST enterArgs, AFUNPTR exit,
                                IARGLIST exitArgs)
{
    static std::set< ADDRINT > instrumented; // aliases (e.g. memalign, aligned_alloc) share an address
    RTN rtn = FindRoutine(img, name);
    // Skip PLT stubs: they jump to the real function and never return.
    if (!RTN_Valid(rtn) || SEC_Name(RTN_Sec(rtn)).find(".plt") == 0) return;
    if (!instrumented.insert(RTN_Address(rtn)).second) return;
    if (getenv("MEMPRINT_VERBOSE"))
        cerr << "memprint_trace: hook " << name << " = " << RTN_Name(rtn) << " in " << IMG_Name(img) << endl;
    RTN_Open(rtn);
    // read by analysis routines of other threads: reserved in main, so pushing never moves it
    if (allocatorCode.size() < allocatorCode.capacity()) allocatorCode.push_back({RTN_Address(rtn), RoutineEnd(rtn)});
    if (sitesOn && KnobSiteContext.Value())
    {
        // the size: argument 0 (malloc, new), 1 (memalign, realloc), 2 (posix_memalign), or 0 x 1 (calloc)
        UINT32 sizeArg = 0, second = 0;
        if (!strcmp(name, "memalign") || !strcmp(name, "aligned_alloc") || !strcmp(name, "realloc")) sizeArg = 1;
        if (!strcmp(name, "posix_memalign")) sizeArg = 2;
        if (!strcmp(name, "calloc")) second = 1;
        if (second)
            RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(SiteContext), IARG_THREAD_ID, IARG_CONST_CONTEXT,
                           IARG_REG_VALUE, REG_STACK_PTR, IARG_RETURN_IP,
                           IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_FUNCARG_ENTRYPOINT_VALUE, 1, IARG_END);
        else
            RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(SiteContext), IARG_THREAD_ID, IARG_CONST_CONTEXT,
                           IARG_REG_VALUE, REG_STACK_PTR, IARG_RETURN_IP,
                           IARG_FUNCARG_ENTRYPOINT_VALUE, sizeArg, IARG_ADDRINT, (ADDRINT)0, IARG_END);
    }
    RTN_InsertCall(rtn, IPOINT_BEFORE, enter, IARG_THREAD_ID, IARG_REG_VALUE, REG_STACK_PTR, IARG_RETURN_IP, IARG_BOOL,
                   exit == AFUNPTR(MallocExit), IARG_IARGLIST, enterArgs, IARG_END);
    if (foreignAllocator)
        HookReturns(rtn, NULL, exit, exitArgs);
    else
        RTN_InsertCall(rtn, IPOINT_AFTER, exit, IARG_THREAD_ID, IARG_IARGLIST, exitArgs, IARG_END);
    RTN_Close(rtn);
    // Allocators other than glibc's leave some calls through a jump to another function of theirs
    // (jemalloc's malloc to je_malloc_default on its slow path; tcmalloc two levels deep), whose
    // return is the call's exit: follow such jumps within the allocator's code, three levels deep.
    if (foreignAllocator) HookTailJumps(rtn, 3, instrumented);
}


// Allocators other than glibc's leave some calls through paths whose return Pin's routine exits
// do not see (a jump into code without a symbol, as in tcmalloc). So, with such an allocator, the
// exit is also taken on the caller's side: before the instruction at the return address of every
// call to an allocation function, where RAX holds the result. A second exit for the same call is
// ignored (the call's depth is already 0). Direct calls (and calls of PLT stubs) are found when
// their code is translated; any other call (through the GOT, as code built with -fno-plt does, or
// through a function pointer) when the allocation function is entered, from its return address:
// that code is then translated again, with the exit, before the call returns.
std::map< ADDRINT, int > returnSites; // return address -> 1 malloc-like, 2 realloc, 3 posix_memalign
std::map< ADDRINT, int > callTargets; // call target -> kind (0: not an allocation function)
PIN_LOCK siteLock;                    // returnSites, written by analysis routines too

static VOID NoteReturnSite(ADDRINT ret, int kind)
{
    if (ret >= mallocCodeLo && ret < mallocCodeHi) return; // the allocator calling itself
    PIN_GetLock(&siteLock, 1);
    BOOL added = returnSites.emplace(ret, kind).second;
    PIN_ReleaseLock(&siteLock);
    if (added) PIN_RemoveInstrumentationInRange(ret, ret + 1);
}

static int AllocationKind(ADDRINT target)
{
    auto it = callTargets.find(target);
    if (it != callTargets.end()) return it->second;
    int kind = 0;
    RTN rtn  = RTN_FindByAddress(target);
    if (RTN_Valid(rtn) && RTN_Address(rtn) == target)
    {
        string name = RTN_Name(rtn);
        name        = name.substr(0, name.find('@')); // malloc@plt
        if (name == "malloc" || name == "calloc" || name == "valloc" || name == "pvalloc" || name == "memalign" ||
            name == "aligned_alloc")
            kind = 1;
        else if (name == "realloc")
            kind = 2;
        else if (name == "posix_memalign")
            kind = 3;
    }
    return callTargets[target] = kind;
}

static VOID InsertReturnSite(INS ins, int kind)
{
    if (kind == 1)
        INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(MallocExit), IARG_THREAD_ID, IARG_REG_VALUE, REG_GAX, IARG_END);
    else if (kind == 2)
        INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(ReallocExit), IARG_THREAD_ID, IARG_CONTEXT, IARG_REG_VALUE, REG_GAX, IARG_END);
    else if (kind == 3)
        INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(PosixMemalignExit), IARG_THREAD_ID, IARG_REG_VALUE, REG_GAX, IARG_END);
}

static VOID CallSites(TRACE trace, VOID* v)
{
    if (!foreignAllocator) return;
    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl))
        for (INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins))
        {
            PIN_GetLock(&siteLock, 1);
            auto site = returnSites.find(INS_Address(ins));
            int siteKind = site != returnSites.end() ? site->second : 0;
            PIN_ReleaseLock(&siteLock);
            if (siteKind) InsertReturnSite(ins, siteKind);
            if (!INS_IsCall(ins) || !INS_IsDirectControlFlow(ins)) continue;
            int kind = AllocationKind(INS_DirectControlFlowTargetAddress(ins));
            if (!kind) continue;
            ADDRINT ret = INS_NextAddress(ins);
            PIN_GetLock(&siteLock, 1);
            BOOL added = returnSites.emplace(ret, kind).second;
            PIN_ReleaseLock(&siteLock);
            if (added)
            {
                // later in this trace, the loop hooks it; elsewhere, code translated before is redone
                INS next = INS_Next(ins);
                if (!INS_Valid(next) && BBL_Valid(BBL_Next(bbl))) next = BBL_InsHead(BBL_Next(bbl));
                if (!INS_Valid(next) || INS_Address(next) != ret) PIN_RemoveInstrumentationInRange(ret, ret + 1);
            }
        }
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

// glibc's malloc.c is one compilation unit, so its code is contiguous: the range spanned by
// its routines (public and internal, e.g. _int_free) is the allocator's code. Internal names
// need the library's symbol table; without one only the public functions are covered.
static VOID FindMallocCode(IMG img)
{
    static BOOL chosen = FALSE; // a non-glibc allocator was found: later images (libc) do not count
    if (chosen || IMG_IsMainExecutable(img)) return;
    RTN m = FindRoutine(img, "malloc");
    string imgName = IMG_Name(img);
    imgName = imgName.substr(imgName.find_last_of('/') + 1);
    if (RTN_Valid(m) && SEC_Name(RTN_Sec(m)).find(".plt") != 0 && imgName.compare(0, 5, "libc.") != 0 &&
        imgName.compare(0, 5, "libc-") != 0)
    {
        chosen = foreignAllocator = TRUE;
        mallocCodeLo = IMG_LowAddress(img), mallocCodeHi = IMG_HighAddress(img) + 1;
        RTN u = FindRoutine(img, "malloc_usable_size");
        usableFn = RTN_Valid(u) ? RTN_Address(u) : 0;
        ElfFunctions(IMG_Name(img), [&](const char*, ADDRINT value, ADDRINT size, BOOL) {
            if (size) functionSizes[IMG_LoadOffset(img) + value] = size;
        });
        if (getenv("MEMPRINT_VERBOSE"))
            cerr << "memprint_trace: allocator " << IMG_Name(img) << " [" << std::hex << mallocCodeLo << ", " << mallocCodeHi
                 << ") malloc_usable_size " << usableFn << std::dec << endl;
        return;
    }
    // names without leading underscores and "libc_" (aliases such as __libc_malloc)
    static const std::set< string > names = {
        "malloc", "free", "cfree", "realloc", "calloc", "memalign", "valloc", "pvalloc", "aligned_alloc", "posix_memalign",
        "malloc_trim", "malloc_usable_size", "int_malloc", "int_free", "int_realloc", "int_memalign", "mid_memalign",
        "sysmalloc", "systrim", "malloc_consolidate", "unlink_chunk", "tcache_init", "ptmalloc_init", "malloc_init_state",
        "arena_get2", "arena_get_retry", "get_free_list", "new_heap", "munmap_chunk", "mremap_chunk"};
    static BOOL haveInternals = FALSE;
    ADDRINT lo = ~(ADDRINT)0, hi = 0;
    BOOL internals = FALSE, any = FALSE;
    for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec))
    {
        if (SEC_Name(sec).find(".plt") == 0) continue;
        for (RTN rtn = SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn))
        {
            string name = RTN_Name(rtn);
            name        = name.substr(0, name.find('.')); // e.g. tcache_init.part.7
            name.erase(0, name.find_first_not_of('_'));
            if (name.compare(0, 5, "libc_") == 0) name.erase(0, 5);
            if (!names.count(name)) continue;
            any |= name == "malloc";
            internals |= name == "int_free";
            lo = std::min(lo, RTN_Address(rtn));
            hi = std::max(hi, RTN_Address(rtn) + RTN_Size(rtn));
        }
    }
    // the first image with malloc, or a later one that has the internals when it did not
    if (!any || hi <= lo || haveInternals || (mallocCodeHi != 0 && !internals)) return;
    mallocCodeLo = lo, mallocCodeHi = hi, haveInternals = internals;
}

VOID ImageLoad(IMG img, VOID* v)
{
    FindMallocCode(img);
    if (sitesOn)
    {
        PIN_GetLock(&stateLock, 1);
        images.push_back({IMG_LowAddress(img), IMG_HighAddress(img) + 1, IMG_LoadOffset(img), IMG_Name(img)});
        PIN_ReleaseLock(&stateLock);
    }
    if (windowed)
        for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec))
            if (SEC_Mapped(sec) && SEC_IsWriteable(sec) && SEC_Size(sec) > 0)
            {
                PIN_GetLock(&stateLock, 1);
                dataRegions.push_back({SEC_Address(sec), SEC_Address(sec) + SEC_Size(sec)});
                PIN_ReleaseLock(&stateLock);
            }
    IARGLIST ret = Args({{IARG_FUNCRET_EXITPOINT_VALUE, 0}});
    IARGLIST ctxtRet = Args({{IARG_CONTEXT, 0}, {IARG_FUNCRET_EXITPOINT_VALUE, 0}});

    InstrumentAllocator(img, "malloc", AFUNPTR(AllocEnter), EnterArgs(0, -1, -1), AFUNPTR(MallocExit), ret);
    InstrumentAllocator(img, "valloc", AFUNPTR(AllocEnter), EnterArgs(0, -1, -1), AFUNPTR(MallocExit), ret);
    InstrumentAllocator(img, "memalign", AFUNPTR(AllocEnter), EnterArgs(1, -1, -1), AFUNPTR(MallocExit), ret);
    InstrumentAllocator(img, "aligned_alloc", AFUNPTR(AllocEnter), EnterArgs(1, -1, -1), AFUNPTR(MallocExit), ret);
    InstrumentAllocator(img, "calloc", AFUNPTR(CallocEnter),
                        Args({{IARG_FUNCARG_ENTRYPOINT_VALUE, 0}, {IARG_FUNCARG_ENTRYPOINT_VALUE, 1}}),
                        sitesOn ? AFUNPTR(CallocExit) : AFUNPTR(MallocExit), sitesOn ? ctxtRet : ret);
    if (sitesOn)
        // -sites: operator new is an allocator too, so the outermost call's return address is in the
        // program (or in STL code instantiated there), not in libstdc++
        for (const char* name : {"_Znwm", "_Znam", "_ZnwmRKSt9nothrow_t", "_ZnamRKSt9nothrow_t", "_ZnwmSt11align_val_t",
                                 "_ZnamSt11align_val_t", "_ZnwmSt11align_val_tRKSt9nothrow_t",
                                 "_ZnamSt11align_val_tRKSt9nothrow_t"})
            InstrumentAllocator(img, name, AFUNPTR(NewEnter), EnterArgs(0, -1, -1), AFUNPTR(MallocExit), ret);
    if (sitesOn)
    {
        for (const char* name : {"__mpr_growth", "__mpr_capacity"})
        {
            RTN rtn = FindRoutine(img, name);
            if (!RTN_Valid(rtn)) continue;
            RTN_Open(rtn);
            if (name[6] == 'g')
                RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(GrowthEnter), IARG_THREAD_ID, IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                               IARG_END);
            else
                RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(CapacityEnter), IARG_THREAD_ID, IARG_FUNCARG_ENTRYPOINT_VALUE,
                               2, IARG_FUNCARG_ENTRYPOINT_VALUE, 3, IARG_END);
            RTN_Close(rtn);
        }
        // name, site argument, size argument(s) (calloc: count x size)
        struct MprEntry { const char* name; UINT32 site, size; INT32 size2; };
        for (MprEntry e : {MprEntry{"__mpr_alloc", 1, 0, -1}, MprEntry{"__mpr_calloc", 2, 0, 1},
                           MprEntry{"__mpr_realloc", 2, 1, -1}, MprEntry{"__mpr_posix_memalign", 3, 2, -1}})
        {
            RTN rtn = FindRoutine(img, e.name);
            if (!RTN_Valid(rtn)) continue;
            RTN_Open(rtn);
            if (e.size2 >= 0)
                RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(MprEnter), IARG_THREAD_ID, IARG_CONST_CONTEXT, IARG_RETURN_IP,
                               IARG_FUNCARG_ENTRYPOINT_VALUE, e.site, IARG_FUNCARG_ENTRYPOINT_VALUE, e.size,
                               IARG_FUNCARG_ENTRYPOINT_VALUE, (UINT32)e.size2, IARG_END);
            else
                RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(MprEnter), IARG_THREAD_ID, IARG_CONST_CONTEXT, IARG_RETURN_IP,
                               IARG_FUNCARG_ENTRYPOINT_VALUE, e.site, IARG_FUNCARG_ENTRYPOINT_VALUE, e.size,
                               IARG_ADDRINT, (ADDRINT)0, IARG_END);
            RTN_Close(rtn);
        }
    }
    InstrumentAllocator(img, "posix_memalign", AFUNPTR(AllocEnter), EnterArgs(2, -1, 0), AFUNPTR(PosixMemalignExit), ret);
    InstrumentAllocator(img, "realloc", AFUNPTR(AllocEnter), EnterArgs(1, 0, -1), AFUNPTR(ReallocExit), ctxtRet);

    if (skipZeroing)
    {
        // memset's implementations (glibc picks one with an IFUNC: memset itself is the resolver
        // there, called once by the loader, and is skipped), bzero, and libmpr's memset and __mpr_zero
        string imgName = IMG_Name(img);
        imgName        = imgName.substr(imgName.find_last_of('/') + 1);
        BOOL libc      = imgName.compare(0, 5, "libc.") == 0 || imgName.compare(0, 5, "libc-") == 0;
        for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec))
        {
            if (SEC_Name(sec).find(".plt") == 0) continue;
            for (RTN rtn = SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn))
            {
                const string& name = RTN_Name(rtn);
                BOOL isMemset = name == "memset" || name.compare(0, 9, "__memset_") == 0;
                BOOL isBzero  = name == "bzero" || name == "__bzero" || name == "__mpr_zero" || name.compare(0, 8, "__bzero_") == 0;
                if (!isMemset && !isBzero) continue;
                if (libc && (name == "memset" || name == "__memset_chk" || name == "bzero")) continue; // IFUNC resolvers
                if (zeroCode.size() < zeroCode.capacity()) zeroCode.push_back({RTN_Address(rtn), RTN_Address(rtn) + RTN_Size(rtn)});
                RTN_Open(rtn);
                if (isMemset)
                    RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(ZeroEnter), IARG_THREAD_ID, IARG_CONST_CONTEXT,
                                   IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_FUNCARG_ENTRYPOINT_VALUE, 1,
                                   IARG_FUNCARG_ENTRYPOINT_VALUE, 2, IARG_END);
                else
                    RTN_InsertCall(rtn, IPOINT_BEFORE, AFUNPTR(BzeroEnter), IARG_THREAD_ID, IARG_CONST_CONTEXT,
                                   IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_FUNCARG_ENTRYPOINT_VALUE, 1, IARG_END);
                RTN_Close(rtn);
            }
        }
    }

    RTN free = FindRoutine(img, "free");
    if (RTN_Valid(free) && SEC_Name(RTN_Sec(free)).find(".plt") != 0)
    {
        RTN_Open(free);
        RTN_InsertCall(free, IPOINT_BEFORE, AFUNPTR(FreeEnter), IARG_THREAD_ID, IARG_CONTEXT, IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                       IARG_END);
        RTN_Close(free);
    }
}

// munmap(addr, len) is caught at the system call, whoever makes it; with
// -window so is an anonymous mmap made outside the allocator (a new block).
VOID SyscallEntry(THREADID tid, CONTEXT* ctxt, SYSCALL_STANDARD std, VOID* v)
{
    ThreadData& t = threads[tid];
    t.munmapLo = t.munmapHi = 0;
    t.mmapLen               = 0;
    ADDRINT number          = PIN_GetSyscallNumber(ctxt, std);
    BOOL anonymous = number == SYS_mmap && (PIN_GetSyscallArgument(ctxt, std, 3) & MAP_ANONYMOUS);
    if (windowed && anonymous && t.allocDepth == 0) t.mmapLen = (PIN_GetSyscallArgument(ctxt, std, 1) + 4095) & ~(ADDRINT)4095;
    if (windowed && (anonymous || number == SYS_mremap) && t.allocDepth > 0) t.allocFresh = TRUE;
    // A region the kernel creates or grows (brk, mmap, mremap) reads as soft-dirty in pagemap
    // until the next clear, whatever was written: harvest the bits now and clear them on exit.
    if (clearRefsFd >= 0 && (number == SYS_brk || number == SYS_mmap || number == SYS_mremap))
    {
        PIN_GetLock(&stateLock, tid + 1);
        if (!finished && clearRefsFd >= 0) Harvest();
        PIN_ReleaseLock(&stateLock);
        t.clearDirty = TRUE;
    }
    if (number != SYS_munmap) return;
    ADDRINT lo  = PIN_GetSyscallArgument(ctxt, std, 0);
    ADDRINT len = PIN_GetSyscallArgument(ctxt, std, 1);
    t.munmapLo  = lo;
    t.munmapHi  = (lo + len + 4095) & ~(ADDRINT)4095; // whole pages are unmapped
}

VOID SyscallExit(THREADID tid, CONTEXT* ctxt, SYSCALL_STANDARD std, VOID* v)
{
    ThreadData& t = threads[tid];
    ADDRINT ret   = PIN_GetSyscallReturn(ctxt, std);
    if (t.clearDirty)
    {
        PIN_GetLock(&stateLock, tid + 1);
        if (clearRefsFd >= 0) ClearDirty();
        PIN_ReleaseLock(&stateLock);
        t.clearDirty = FALSE;
    }
    if (t.munmapHi > t.munmapLo && ret == 0)
    {
        QueueRelease(tid, ctxt, t.munmapLo, t.munmapHi);
        if (windowed)
        {
            PIN_GetLock(&stateLock, tid + 1);
            if (!finished) UnmapBlocks(t.munmapLo, t.munmapHi);
            PIN_ReleaseLock(&stateLock);
        }
    }
    if (t.mmapLen && ret != (ADDRINT)MAP_FAILED)
    {
        PIN_GetLock(&stateLock, tid + 1);
        if (!finished) AddBlock(ret, t.mmapLen, TRUE, TRUE);
        PIN_ReleaseLock(&stateLock);
        ApplyWindowChange();
    }
    t.munmapLo = t.munmapHi = t.mmapLen = 0;
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

// spatial: every access to a selected address; its hash also picks the bucket.
static VOID SpatialReference(ADDRINT address, UINT32 size)
{
    if (!countBytes)
    {
        footprint.Record(address, size);
        UINT32 bucket = Mix(address ^ salt) % numBins;
        bins[bucket].Record(address, size);
        ++binObservations[bucket];
        return;
    }
    // bytes: only the part of the access inside each selected word; the word picks the bucket
    for (ADDRINT word = address >> WORD_BITS; word <= (address + size - 1) >> WORD_BITS; word++)
    {
        if (!WordSelected(word)) continue;
        ADDRINT lo = std::max(address, word << WORD_BITS), hi = std::min(address + size, (word + 1) << WORD_BITS);
        footprint.Record(lo, (UINT32)(hi - lo));
        UINT32 bucket = Mix(word ^ salt) % numBins;
        bins[bucket].Record(lo, (UINT32)(hi - lo));
        ++binObservations[bucket];
    }
}

static VOID ApplyRelease(ADDRINT lo, ADDRINT hi)
{
    SiteRelease(lo, hi);
    footprint.Erase(lo, hi);
    for (Footprint& bin : bins)
        bin.Erase(lo, hi);
    for (Footprint& u : unions)
        u.Erase(lo, hi);
}

// A queued release, or a -sites probe
static VOID ApplyPending(THREADID tid, const Release& r)
{
    if (r.zero)
    {
        threads[tid].zeroLo = r.lo, threads[tid].zeroHi = r.hi;
        return;
    }
    if (r.probe)
        SiteProbe(r.lo);
    else
        ApplyRelease(r.lo, r.hi);
}

static UINT64 MainInterval() { return (mode == SPLITTER) ? 1 : KnobSamplingInterval.Value(); }

static UINT64 BinInterval(UINT32 mapIndex)
{
    if (mode == SPLITTER) return intervals[mapIndex / numBins];
    if (mode == SPATIAL) return (UINT64)KnobSamplingInterval.Value() * numBins; // each bucket selects 1-in-(i*s) addresses
    return sampledBinInterval;
}

static VOID WriteWindowed();

// microseconds as seconds with 6 decimals (hand-rolled, see WriteSnapshot)
static string SecondsText(UINT64 us)
{
    string text = std::to_string(us / 1000000) + ".", frac = std::to_string(us % 1000000);
    return text + string(6 - frac.size(), '0') + frac;
}

static VOID WriteSnapshot()
{
    if (timeNow == lastSnapshot) return;
    lastSnapshot   = timeNow;
    snapshotMicros = Micros() - startMicros;
    string seconds = SecondsText(snapshotMicros);
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
        for (int k = 1; k <= 4; k++)
            text += ',', num(f.sampled[k]);
        text += ',', num(f.discovered), text += ',', text += seconds, text += '\n';
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
    if (windowed) WriteWindowed();
}

static const char* WINDOWED_HEADER = "Time,Watching,Windows,AllocatedBytes,Blocks,FreshBytes,FreshResident,ReusedBytes,"
                                     "ReusedResident,FreshEst,ReusedEst,OtherEst,FreshCovered,ReusedCovered,Density,Estimate,"
                                     "Window,Period,ReusedWritten,TriggeredWindows,WatchedRefs,RegionsResident,PartialEdges,Seconds";

// Resident bytes of the program's writable image sections and of its main thread's stack (other
// threads' stacks are anonymous mmaps, tracked as blocks). The stack is the 8 MB (the usual limit)
// below the top of "[stack]" in /proc/self/maps, which is read once: under Pin the file is long
// and Pin's C library slow (reading it at every snapshot made 2mm LARGE take 75 s instead of 20).
static UINT64 RegionsResident()
{
    static ADDRINT stackTop = 0;
    if (!stackTop)
    {
        ifstream maps("/proc/self/maps");
        string line;
        while (std::getline(maps, line))
            if (line.find("[stack]") != string::npos)
                stackTop = strtoull(line.substr(line.find('-') + 1).c_str(), NULL, 16);
        if (!stackTop) stackTop = 1; // no [stack]: do not look again
    }
    vector< std::pair< ADDRINT, ADDRINT > > regions = dataRegions;
    if (stackTop > (8 << 20)) regions.push_back({stackTop - (8 << 20), stackTop});
    std::sort(regions.begin(), regions.end());
    UINT64 total = 0;
    ADDRINT coveredTo = 0;
    ForgetResidency();
    for (const auto& r : regions) // sections can overlap (.tbss inside .bss); count each page once
    {
        ADDRINT lo = std::max(r.first, coveredTo);
        if (r.second > lo) total += Resident(lo, r.second), coveredTo = r.second;
    }
    ForgetResidency();
    return total;
}

// -footprint pages, windowed: bytes of the resident pages that hold a byte of a live block
static UINT64 BlockPagesResident()
{
    UINT64 total = 0;
    ADDRINT lo = 0, hi = 0; // the page-aligned run of blocks being merged
    ForgetResidency();
    for (const auto& kv : blocks) // in address order
    {
        ADDRINT a = kv.first >> PAGE_BITS << PAGE_BITS;
        ADDRINT b = (kv.first + kv.second.Extent() + (1 << PAGE_BITS) - 1) >> PAGE_BITS << PAGE_BITS;
        if (a > hi)
        {
            if (hi > lo) total += Resident(lo, hi);
            lo = a;
        }
        hi = std::max(hi, b);
    }
    if (hi > lo) total += Resident(lo, hi);
    ForgetResidency();
    return total;
}

// windowed: the footprint estimate (see "Windows" at the top)
static VOID WriteWindowed()
{
    if (watching) Attribute(); // the selected footprint changes only inside windows
    const UINT64 R = KnobSamplingInterval.Value();
    double reusedDensity = R * kindCovered[0] >= 65536 ? (double)kindSampled[0] / kindCovered[0] : 1.0;
    UINT64 bytes[2] = {0, 0}, resident[2] = {0, 0}, reusedWritten = 0; // reused [0], fresh [1]
    UINT64 edges = 0; // touched pages that border untouched ones inside blocks (probably partly used)
    double blocksEstimate = 0;
    ForgetResidency();
    for (auto& kv : blocks)
    {
        Block& b = kv.second;
        bytes[b.fresh] += b.size;
        UINT64 w = 0;
        UINT64 r = Resident(kv.first, kv.first + b.Extent(), b.fresh || clearRefsFd < 0 ? NULL : &b.written, &w, &edges);
        resident[b.fresh] += r;
        if (!b.fresh) reusedWritten += w;
        // touched bytes x density: the block's own once its sample covers 64 KB / i, else that of all
        // reused blocks (reused) or 1 (fresh: large arrays, mostly written in whole words)
        UINT64 touched = b.fresh ? r : clearRefsFd >= 0 ? w : r;
        double density = R * b.covered >= 65536 ? (double)b.sampled / b.covered : b.fresh ? 1.0 : reusedDensity;
        blocksEstimate += touched * density;
    }
    // Clear the soft-dirty bits every dirtyEvery references (not at every snapshot: after a
    // clear the kernel write-protects pages, so each page's next write faults once).
    if (clearRefsFd >= 0 && (dirtyBySnapshots ? ++snapshotsSinceClear >= 10 : timeNow >= nextDirtyClear))
    {
        snapshotsSinceClear = 0;
        if (write(clearRefsFd, "4", 1) != 1) clearRefsFd = -1; // the scan above harvested the bits
        while (nextDirtyClear <= timeNow)
            nextDirtyClear += dirtyEvery;
    }
    // Reused blocks may sit on pages that earlier blocks touched, so their residency overstates
    // them; the pages written since they were allocated do not (heap memory is written before it
    // is read). Without soft-dirty tracking, residency is used.
    double density  = reusedDensity;
    // -pages_only: memory outside blocks from the pages of the program's segments and stack;
    // with windows, from the bytes the windows selected outside blocks
    UINT64 regions  = pagesOnly || countPages ? RegionsResident() : 0; // (reads /proc/self/maps: only when used)
    // -pages_only: discard the translated code once, after the first snapshot. Without this, gemm
    // and 2mm LARGE ran 8x slower (gemm: 104 s against 12.6 s); windowed runs, which flush when
    // their first window closes, did not. Not the snapshot work (10x fewer snapshots: 104 s) nor
    // soft-dirty tracking (-dirty_every 0: 103 s); the cause is unknown.
    static BOOL flushedOnce = FALSE;
    if (pagesOnly && !flushedOnce) flushedOnce = TRUE, flushPending = TRUE, epoch++;
    double estimate = blocksEstimate + (pagesOnly ? (double)regions : (double)R * otherSampled);
    // a touched page next to an untouched one inside a block is, on average, half used
    if (KnobEdgeCorrection.Value()) estimate = std::max(0.0, estimate - (double)edges * (1 << (PAGE_BITS - 1)));
    // -footprint pages: the resident pages of live blocks (each page once, whoever touched it) and of
    // the program's segments and stack; no densities, no partial pages
    if (countPages) estimate = (double)(BlockPagesResident() + regions);
    windowedFile << timeNow << "," << (watching ? 1 : 0) << "," << windowsDone << "," << bytes[0] + bytes[1] << ","
                 << blocks.size() << "," << bytes[1] << "," << resident[1] << "," << bytes[0] << "," << resident[0] << ","
                 << R * kindSampled[1] << "," << R * kindSampled[0] << "," << R * otherSampled << "," << R * kindCovered[1]
                 << "," << R * kindCovered[0] << "," << density << "," << (UINT64)estimate << "," << KnobWindow.Value() << ","
                 << KnobPeriod.Value() << "," << reusedWritten << "," << triggeredWindows << ","
                 << watchedRefs + (watching ? timeNow - windowStart : 0) << "," << regions << "," << edges << ","
                 << SecondsText(snapshotMicros) << "\n";
}

// windowed: windows are periodic (one of -window references at every multiple of -period)
// or triggered by a large allocation (-window_alloc). Opening or closing one changes the
// instrumentation, which the caller applies after releasing stateLock (flushPending).
// A window opened or closed. Flushing: discard translated code, and make threads restart in
// fresh code (epoch). -window_switch auto: after an expensive flush, switch to versions (one last
// flush installs them).
static VOID WindowChanged()
{
    if (!versioned && autoSwitch && tracesSinceChange > KnobVersionTraces.Value()) versioned = TRUE;
    else if (versioned) return;
    tracesSinceChange = 0;
    flushPending      = TRUE;
    epoch++;
}

static VOID OpenWindow(UINT64 start, BOOL triggered)
{
    watching        = TRUE;
    triggeredWindow = triggered;
    windowStart     = start;
    windowEnd       = start + KnobWindow.Value();
    WindowChanged();
}

static VOID CloseWindow()
{
    Attribute();
    windowsDone++;
    watchedRefs += windowEnd - windowStart;
    if (triggeredWindow) lastTriggeredEnd = windowEnd;
    watching     = FALSE;
    WindowChanged();
}

// windowed: open or close windows when time reaches them
static VOID AdvanceWindows()
{
    for (;;)
    {
        if (watching && nextPeriodic <= timeNow)
        {
            // a periodic window that starts inside an open (triggered) window extends it
            windowEnd = std::max(windowEnd, nextPeriodic + KnobWindow.Value());
            nextPeriodic += KnobPeriod.Value();
        }
        else if (watching && timeNow >= windowEnd)
            CloseWindow();
        else if (!watching && timeNow >= nextPeriodic)
        {
            OpenWindow(nextPeriodic, FALSE);
            nextPeriodic += KnobPeriod.Value();
        }
        else
            break;
    }
}

// windowed: a block of `size` bytes was allocated. A large allocation is usually followed by
// the program filling the block, so a window opened now measures the block's density. At most
// one such window per period, so at most twice the periodic watched fraction is watched.
static VOID MaybeTriggerWindow(ADDRINT size)
{
    UINT64 threshold = KnobWindowAlloc.Value();
    if (pagesOnly || !threshold || size < threshold || watching) return;
    if (triggeredWindows && timeNow < lastTriggeredEnd + KnobPeriod.Value()) return;
    OpenWindow(timeNow, TRUE);
    triggeredWindows++;
}

static VOID WriteOutputs();

// Advance time to `now` (memory references executed); returns FALSE once -stop is reached.
static BOOL Tick(UINT64 now)
{
    if (now > timeNow) timeNow = now;
    if (windowed) AdvanceWindows();
    if (KnobSnapshot.Value() && !clockSeconds && timeNow >= nextSnapshot)
    {
        WriteSnapshot();
        while (nextSnapshot <= timeNow)
            nextSnapshot += KnobSnapshot.Value();
    }
    else if (KnobSnapshot.Value() && clockSeconds && timeNow >= nextClockCheck)
    {
        nextClockCheck = timeNow + CLOCK_CHECK_REFS;
        UINT64 now     = Micros() - startMicros;
        if (now >= nextSnapshot)
        {
            WriteSnapshot();
            while (nextSnapshot <= now)
                nextSnapshot += KnobSnapshot.Value() * 1000;
        }
    }
    if (KnobStop.Value() && timeNow >= KnobStop.Value())
    {
        WriteOutputs();
        PIN_Detach();
        return FALSE;
    }
    return TRUE;
}

// spatial: every reference counts toward time (memory references executed so far)
static UINT64 TotalRefs()
{
    UINT64 total = 0;
    for (const ThreadData& t : threads)
        total += t.refs;
    return total;
}

// spatial: an access to a selected address, recorded when it happens
static VOID RecordSelected(THREADID tid, ADDRINT address, UINT32 size)
{
    if (address == 0) return;
    PIN_GetLock(&stateLock, tid + 1);
    if (!finished && watching) // (windowed: code from a closed window runs until the thread's next check)
    {
        ++observations;
        SpatialReference(address, size);
        Tick(TotalRefs());
    }
    BOOL flush   = flushPending;
    flushPending = FALSE;
    PIN_ReleaseLock(&stateLock);
    if (flush) PIN_RemoveInstrumentation(); // re-instrument for the window change
}

// windowed: a thread has counted another checkStep references. Removing
// instrumentation does not change code that is running: a loop keeps jumping
// back into its old translation. So each thread, at its next check after a
// window change, restarts at the current instruction, in code instrumented
// for the new state.
// returns whether a window is open (-window_switch version: into the version register)
static ADDRINT CheckTime(THREADID tid, CONTEXT* ctxt)
{
    ThreadData& t = threads[tid];
    t.checkAt     = t.refs + checkStep;
    PIN_GetLock(&stateLock, tid + 1);
    if (!finished) Tick(TotalRefs());
    BOOL flush   = flushPending;
    flushPending = FALSE;
    PIN_ReleaseLock(&stateLock);
    if (flush) PIN_RemoveInstrumentation();
    if (t.epoch != epoch)
    {
        t.epoch = epoch;
        PIN_ExecuteAt(ctxt);
    }
    return watching;
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
            ApplyPending(tid, pending[nextRelease]);

        ADDRINT address = reference->ea;
        UINT32 size     = reference->size;
        if (address == 0) continue;
        if (size != MALLOC_REF && (size & ZERO_REF))
        {
            size &= ~ZERO_REF;
            const ThreadData& t = threads[tid];
            if (address >= t.zeroLo && address < t.zeroHi) // zeros over [zeroLo, zeroHi): time only
            {
                ++observations, zeroSkipped += size;
                if (!Tick(timeNow + MainInterval())) break;
                continue;
            }
        }

        ++observations;
        if (size == MALLOC_REF)
            ; // made by the allocator: time only (see InMallocCode)
        else if (mode == SPLITTER)
            SplitReference(tid, address, size);
        else
            BootstrapReference(tid, address, size);
        if (!Tick(timeNow + MainInterval())) break;
    }
    for (; nextRelease < pending.size() && !finished; nextRelease++)
        ApplyPending(tid, pending[nextRelease]);
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
    prefix << outDir << "/" << (mode == SPLITTER ? "Buffered_" : mode == SAMPLER ? "Sampled_" : "Spatial_") << name << "_" << MainInterval() << "_"
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
        nextSnapshot = KnobSnapshot.Value() * (clockSeconds ? 1000 : 1);
        if (windowed)
        {
            windowedFile.open(outputFileNamePrefix + "_windowed.csv");
            windowedFile << WINDOWED_HEADER << "\n";
        }
    }
    if (sitesOn)
    {
        blocksFile.open(outputFileNamePrefix + "_blocks.csv");
        blocksFile << "site,kind,size,born,died,touched,high_water,pages_touched,pages,alloc_touched,grown_from,moved,context,"
                      "runs\n";
    }
    return TRUE;
}

// Write the summary files (and the last snapshot). Called with stateLock held
// or after all threads have finished.
static VOID WriteOutputs()
{
    if (finished) return;
    finished = TRUE;
    SitesFinish();

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
        // the last snapshot must show the end state (e.g. releases applied in Fini)
        if (timeNow == lastSnapshot) timeNow++;
        WriteSnapshot();
        timelineFile.close();
        windowedFile.close();
    }
}

VOID Fini(INT32 code, VOID* v)
{
    PIN_GetLock(&stateLock, 0);
    if (mode == SPATIAL && TotalRefs() > timeNow) timeNow = TotalRefs();
    for (UINT32 tid = 0; tid < MAX_THREADS; tid++)
    {
        for (const Release& r : threads[tid].pending)
            ApplyPending(tid, r);
        threads[tid].pending.clear();
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
    startMicros = Micros();
    if (KnobClock.Value() != "refs" && KnobClock.Value() != "seconds") return Usage("-clock must be refs or seconds");
    clockSeconds = KnobClock.Value() == "seconds";

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
    else if (KnobMode.Value() == "spatial")
    {
        mode    = SPATIAL;
        numBins = KnobNumSplits.Value();
        if (KnobSamplingInterval.Value() == 0 || numBins == 0) return Usage("-i and -s must be positive");
        spatialThreshold = ~(UINT64)0 / KnobSamplingInterval.Value();
        if (KnobWindow.Value() || KnobPagesOnly.Value())
        {
            pagesOnly = KnobPagesOnly.Value();
            const string& sw = KnobWindowSwitch.Value();
            if (sw != "version" && sw != "flush" && sw != "auto") return Usage("-window_switch must be auto, version or flush");
            versioned  = sw == "version" && !pagesOnly;
            autoSwitch = sw == "auto" && !pagesOnly;
            if (versioned || autoSwitch)
            {
                watchReg = PIN_ClaimToolRegister();
                if (!REG_valid(watchReg)) return Usage("no tool register for -window_switch version");
            }
            if (pagesOnly && KnobWindow.Value()) return Usage("-pages_only and -window exclude each other");
            if (!pagesOnly && KnobPeriod.Value() <= KnobWindow.Value()) return Usage("-period must be larger than -window");
            if (!KnobSnapshot.Value()) return Usage("-window and -pages_only need -snapshot");
            windowed   = TRUE;
            // residency of fresh blocks is their touched pages, at 4 KB granularity only without huge pages
            syscall(SYS_prctl, PR_SET_THP_DISABLE, 1, 0, 0, 0); // Pin's CRT has no prctl()
            pagemapFd = open("/proc/self/pagemap", O_RDONLY);
            if (pagemapFd < 0) return Usage("-window needs /proc/self/pagemap");
            // (-clock seconds: every 10 snapshots, counted in WriteWindowed)
            dirtyBySnapshots = clockSeconds && KnobDirtyEvery.Value().empty();
            dirtyEvery       = dirtyBySnapshots              ? 1
                               : KnobDirtyEvery.Value().empty() ? 10 * KnobSnapshot.Value()
                                                                : strtoull(KnobDirtyEvery.Value().c_str(), NULL, 10);
            if (dirtyEvery)
            {
                clearRefsFd = open("/proc/self/clear_refs", O_WRONLY);
                // start from a clean state: pages written before the program started are not its blocks'
                if (clearRefsFd >= 0 && write(clearRefsFd, "4", 1) != 1) clearRefsFd = -1;
                nextDirtyClear = dirtyEvery;
            }
            // the first periodic window opens at time 0 (-pages_only: none ever does)
            nextPeriodic = pagesOnly ? ~(UINT64)0 : KnobPeriod.Value();
            windowEnd    = KnobWindow.Value();
            if (pagesOnly) watching = FALSE;
            // Threads advance time every checkStep references: well inside a window, the gap
            // between windows and a snapshot, but not so often that the lock costs (a window
            // shorter than 8 x 10^4 references lasts longer than asked).
            // (-clock seconds: snapshots are not in references; Tick reads the clock every CLOCK_CHECK_REFS)
            UINT64 snapshotRefs = clockSeconds ? CLOCK_CHECK_REFS : KnobSnapshot.Value();
            UINT64 shortest     = pagesOnly ? snapshotRefs
                                            : std::min({KnobWindow.Value(), KnobPeriod.Value() - KnobWindow.Value(), snapshotRefs});
            checkStep       = std::max< UINT64 >(10000, shortest / 8);
        }
        UINT64 base      = KnobSeed.Value().empty() ? PIN_GetPid() : strtoull(KnobSeed.Value().c_str(), NULL, 10);
        salt             = Mix(base + 0x9e3779b97f4a7c15ULL);
        bins.resize(numBins);
    }
    else
        return Usage("unknown -mode " + KnobMode.Value());
    binObservations.assign(bins.size(), 0);

    if (KnobFootprint.Value() == "bytes")
        countBytes = TRUE;
    else if (KnobFootprint.Value() == "starts")
        countBytes = FALSE;
    else if (KnobFootprint.Value() == "pages")
    {
        if (mode != SPLITTER && !windowed) return Usage("-footprint pages needs -mode splitter, -window or -pages_only");
        countBytes = countPages = TRUE;
        if (mode == SPLITTER) footprint.SetPages(TRUE); // the exact footprint; sampled bins stay in bytes
    }
    else if (KnobFootprint.Value().empty())
        countBytes = KnobTrackFrees.Value() || windowed;
    else
        return Usage("-footprint must be bytes, starts or pages");
    sitesOn = KnobSites.Value();
    skipZeroing = mode == SPLITTER && (KnobSkipZeroing.Value() < 0 ? sitesOn : KnobSkipZeroing.Value() != 0);
    if (KnobSkipZeroing.Value() > 0 && mode != SPLITTER) return Usage("-skip_zeroing needs -mode splitter");
    zeroCode.reserve(256);
    if (sitesOn && (mode != SPLITTER || !KnobTrackFrees.Value() || !countBytes))
        return Usage("-sites needs -mode splitter -track_frees 1 with the bytes footprint");

    allocatorCode.reserve(64);
    PIN_InitLock(&stateLock);
    PIN_InitLock(&allocLock);
    PIN_InitLock(&siteLock);
    PIN_InitLock(&blockSiteLock);
    if (KnobTrackFrees.Value() || windowed)
    {
        footprint.SetIndexed(TRUE);
        for (Footprint& bin : bins)
            bin.SetIndexed(TRUE);
        for (Footprint& u : unions)
            u.SetIndexed(TRUE);
        IMG_AddInstrumentFunction(ImageLoad, 0);
        TRACE_AddInstrumentFunction(CallSites, 0);
        PIN_AddSyscallEntryFunction(SyscallEntry, 0);
        PIN_AddSyscallExitFunction(SyscallExit, 0);
    }

    // Each thread's accesses wait in its own buffer until it fills, so across
    // threads they are processed late by up to a buffer: a free in one thread
    // can be applied before another thread's earlier accesses. Timelines and
    // free tracking therefore use small buffers; without them the default is
    // kept. (Spatial mode records its rare accesses immediately, unbuffered.)
    UINT32 pages = KnobBufferPages.Value();
    if (pages == 0)
        pages = (KnobSnapshot.Value() || KnobTrackFrees.Value()) ? 16 : NUM_BUF_PAGES;
    bufId = PIN_DefineTraceBuffer(sizeof(struct MEMREF), pages, BufferFull, 0);
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

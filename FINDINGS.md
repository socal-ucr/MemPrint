# Findings: footprint over time

Summary: sampling *addresses* instead of references (`-mode spatial`) estimates the live footprint over time within about 1–10% at 1-in-25 to 1-in-250 addresses on PolyBench, and within 0.2–6% on miniVite with 1, 4 or 16 threads. It needs no training and runs at about the cost of instrumentation alone (miniVite 8192: ~30 s vs ~180 s for a full trace). Everything after "Data" documents the reference-sampling route that led there.

Goal: reconstruct a program's live memory footprint over time from a sparse Pin trace. The data is the timelines written with `-snapshot`/`-track_frees` (see README).

On `main`, the paper's α model applied per snapshot gives 25–55% MAPE. This branch adds per-address sample counts to the Pin tool and tries estimators of the memory that was never sampled.

## Windowed sampling: watching part of a run (branch `windowed`)

Summary: for long runs, the Pin tool can watch memory accesses for a small fraction of the run, as long as it tracks allocations and page residency the whole time. On PolyBench LARGE this reconstructs the live footprint over time within 0.2% at any watched fraction. On miniVite 65536 it is within 1.7–4.5% on average and 0.4–3% at the peak, watching 5–20% of the run, and on miniVite 32768 within 2.1–8.3% on average and 4–10% at the peak. A heap-churn test, where new blocks land on pages that freed blocks touched, stays within about 24% (page granularity) instead of +250%. At 5% watched that costs 2.8–3.2× native on PolyBench, against 21–36× for full spatial sampling, and 13–15× on miniVite, against 25–27×.


### Current definition: the footprint is the bytes touched

From 2026-10-08, with `-track_frees` or `-window`, the tool counts the bytes a program touches (`-footprint bytes`): the union of every access's byte range, live until released. The paper's definition (`-footprint starts`, still the default without those knobs) adds up the largest access at each start address, so overlapping accesses count more than once. An 8-byte read at every byte of a 1 MB buffer gives 8.5 MB under it and 1.15 MB of bytes touched. **The sections below, up to "Ground truth fixed", were measured with the old definition.**

Changes that came with it:
- **The truth** keeps a bitmap of touched bytes per page, and a release clears exactly the released bytes.
- **Freed blocks release their whole glibc chunk** (size field, block and slack, read from the chunk header at `free`'s entry). Under bytes, string and copy functions that read past a block's requested size left those bytes "live" for ever: Lua ended with 5.3 MB live and nothing allocated, and now ends with 0.3 MB.
- **Spatial and windowed sampling select 64-byte chunks** and keep the bytes of each access inside selected chunks, so selected bytes × i is unbiased. Selecting start addresses would let a byte covered by k overlapping accesses in with probability about k/i.
- **Density is 1 by construction,** so the windowed estimate is the fresh blocks' resident bytes, plus the reused blocks' written bytes, plus the windows' selected bytes outside blocks × i.
- Default splitter and sampler outputs are byte-identical to before (address randomisation off); the tests pass.

**Results** (single thread, against new full traces; mean error / peak error, %; 1-in-100 chunks, about 20 windows):

| Run | True peak, old → new | Nothing watched | 1% | 5% | 5%, allocation windows | Watching all the time |
|---|---|---|---|---|---|---|
| 2mm LARGE | 37.1 → 37.1 MB | — | — | 0.1 / +0.1 | 0.1 / +0.1 | 0.1 / +0.1 |
| miniVite 65536 | 35.5 → 28.2 MB | — | — | 2.8 / +1.0 | 2.7 / +1.0 | 3.9 / +1.5 |
| miniVite 32768 | 19.3 → 15.4 MB | 14.5 / −6.8 | 4.3 / +2.0 | 1.8 / +0.8 | 4.0 / +1.9 | 6.9 / +3.2 |
| GAP bfs | 102.7 → 72.8 MB | 5.0 / −0.3 | 3.0 / −0.0 | 2.3 / +0.1 | 3.4 / +0.4 | 2.4 / +0.2 |
| darknet AlexNet | 399 → 270 MB | 3.1 / +4.0 | 3.2 / +4.1 | 3.2 / +4.1 | 3.2 / +4.1 | 3.2 / +4.1 |
| python | 126 → 96 MB | 3.7 / +2.8 | 3.7 / +3.6 | 6.0 / +3.8 | 3.9 / +3.7 | 4.6 / +3.9 |
| lua | 70.9 → 57.2 MB | 7.5 / −8.1 | 6.2 / −7.5 | 5.8 / −7.5 | 6.2 / −7.3 | 5.3 / −5.2 |
| perl | 64.5 → 54.8 MB | 1.9 / −3.4 | 1.8 / −2.9 | 1.8 / −1.6 | 1.9 / −2.8 | 1.9 / −2.3 |
| sqlite | 263 → 74 MB | 6.5 / +0.2 | 9.0 / +1.9 | 8.7 / +1.7 | 8.3 / +2.8 | 8.5 / +0.8 |
| churn, 25% touched | 1.3 → 1.3 MB | 25.1 / +22.2 | 22.7 / +20.2 | 24.3 / +23.0 | 22.4 / +21.7 | 30.9 / +27.6 |

"Old" is the paper's definition with the allocator fix below.

- **Every real workload is within −8% to +4% of its peak at 5% watched,** against −4% to −61% before (5% with allocation windows, old definition, allocator fix applied). The underestimates came from the definition, which windows could not measure, not from heap reuse.
- **Watching hardly matters any more.** With nothing watched, the estimate is pages alone, and it is within −8% to +4% of every peak except miniVite 32768 (−6.8%, 14.5% mean). That run has 2–3 MB outside blocks (stack, MPI), which only windows see; 1% watched fixes it.
- **Cost.** Without windows the runs took 20–54 s for the interpreters and miniVite 32768, against 48–163 s at 5% and 276–982 s for the full traces (18 runs at once, so rough). With almost nothing to watch, the cost is the allocator hooks and the soft-dirty clears.
- **What is left:**
  - Page granularity: the churn test is still about +22%, and darknet's +4% is fresh pages touched only in part.
  - Lua is 5–8% low at every fraction, including full watching. Not yet investigated.

### Why

Spatial sampling still checks every memory access, even when it records only 1 in 100 addresses. On 2mm LARGE that alone takes 203 s against 7.5 s native, and Pin itself costs almost nothing (7.7 s). For long real workloads that overhead is the obstacle, not the sampling rate.

### Method

`-mode spatial -window W -period P` watches accesses for W of every P memory references.

**Between windows,** only two things run:
- a reference counter, one inlined add per basic block, which keeps time;
- the allocator hooks.

**Live allocations are tracked all the time.** These are heap blocks (the malloc family) and anonymous mmaps. Frees remove them.

**At every snapshot (`-snapshot N`), the estimate is:**

    Σ over live blocks (touched bytes × density)
    + windows' selected footprint outside blocks × i

- **Touched bytes:** the bytes of a block on resident pages, read from `/proc/self/pagemap`. A page becomes resident when it is first read or written, so residency records first touches whether or not a window is open.
- **Fresh blocks** got pages mapped for them: an anonymous mmap, or a malloc that made a new mmap. Their pages start untouched, so residency is exact to the page.
- **Reused blocks** sit on heap pages that may have been used before, so their residency can include pages that earlier blocks touched. They are measured instead by the pages *written* since they were allocated, from the kernel's soft-dirty bits (pagemap bit 55, cleared by writing `4` to `/proc/self/clear_refs`). Heap memory is written before it is read, so written pages are its touched pages.
  - Clearing write-protects every page, so the next write to each page faults once. Bits are therefore cleared at three points rather than at every snapshot.
    - At the allocator's entry, for allocations of 16 KB or more, after recording every reused block's written pages. Writes to the new block's pages by earlier blocks then do not count, and calloc's zeroing and realloc's copy, which happen inside the call, do.
    - Around `brk`, `mmap` and `mremap`: pagemap reports a whole region as soft-dirty when the kernel creates or grows it, whatever was written. The bits are recorded at the system call's entry and cleared at its exit.
    - Every `-dirty_every` references (default: every 10 snapshots).
  - Blocks whose pages are all marked written are skipped when recording, and realloc in place keeps a block's marks.
- **Huge pages:** the tool disables transparent huge pages for the traced process (`prctl(PR_SET_THP_DISABLE)`). Without that, a single touch makes a whole 2 MB region resident.
- **Density:** the footprint counts the largest access at every start address, so overlapping accesses (unaligned copies, mixed widths) count more than the bytes they touch. Density is footprint ÷ covered bytes, measured per block once the block's sample covers 64 KB / i. Otherwise a reused block uses the density of all reused blocks and a fresh block uses 1. In miniVite's heap it is about 1.45 for most of the run and about 1.1 at the end.
  - Windows select 1-in-i 64-byte chunks instead of single addresses. They therefore see every access in a selected chunk, and can measure footprint and covered bytes on the same memory.
  - The ratio doesn't depend on how much of the run the windows covered.
- **Memory outside blocks** (stack, globals, MPI shared memory) is seen only in windows.

**Output:** each snapshot is a row of `<prefix>_windowed.csv` with every component. `analysis -m memprint timeline windowed` scores the runs against the splitter's truth.

### Accuracy

Ground truth comes from full traces (splitter with `-track_frees`) of the same input and thread count. Each windowed run is scored on its whole timeline, with time as a fraction of the run. Settings: 1-in-100 chunks, about 20 windows per run, nothing tuned per workload.

**PolyBench LARGE** (2mm, gemm, jacobi-2d; live peaks of 27–37 MB, 28–49 billion references):
- Mean error is 0.0–0.2% and peak error within 0.2%, at every watched fraction including none.
- Every array is fresh, so residency alone gives the exact curve.
- Baselines:
  - live allocated bytes: 0.4–4.5% mean error;
  - the windows' selected footprint alone at 5% watched: 0.6–8.3%;
  - the same at 1% watched: 11–49%.

**miniVite** (Louvain on random geometric graphs; its footprint grows in reused heap blocks). Each cell is mean error / peak error, %:

| Input, threads | Nothing watched | 1% | 5% | 20% | 5%, 1-in-25 |
|---|---|---|---|---|---|
| 65536, 1 thread | 25.9 / −24.6 | 9.1 / −14.5 | 3.8 / −1.9 | 4.5 / −0.4 | 3.5 / −3.1 |
| 65536, 4 threads | 26.4 / −24.8 | 8.4 / −14.4 | 3.1 / −3.0 | 1.7 / −2.8 | 3.9 / −2.6 |
| 32768, 1 thread | 32.1 / −27.9 | 14.4 / −14.1 | 8.3 / −9.8 | 4.3 / −4.3 | 5.6 / −8.8 |
| 32768, 4 threads | 32.3 / −28.0 | 17.1 / −15.2 | 5.7 / −9.0 | 2.1 / −4.8 | 6.4 / −9.5 |

Baselines at 5% watched (65536, 1 thread):
- residency without the density correction: 12.6 / −19.7;
- the windows' selected footprint alone: 44.5 / −45.5;
- live allocated bytes: 67.4 / +44.2. miniVite reserves much more than it touches.

Before soft-dirty tracking and per-block density, the same runs gave 2.1–4.0 / −1.7 to −3.9 on 65536, about the same, and 5.3–6.2 / −11.9 to −17.2 on 32768 with 4 threads.

- **Residency is necessary.** Windows alone see only accesses that happen inside a window. They miss memory that is touched once and then left alone.
- **Density is necessary.** Without it, miniVite's peak is about 20% low at every watched fraction.
- **Watched fraction.** 5% is enough on miniVite 65536.
  - At 1% the density measured mid-run is 1.05–1.12, against about 1.45 at 5% and 20%.
  - With nothing watched, density falls back to 1 and memory outside blocks isn't seen at all.
- **miniVite 32768's peak** is a spike that lasts about 0.3% of the run, at 90.5%: 4.3 MB of new heap blocks appear and the true footprint jumps from 13.3 to 19.9 MB.
  - It was not threads or windows: 1 and 4 threads estimated 17.6 and 17.5 MB at the spike, three repeated runs agreed within 1.4 points, 40 or 80 shorter windows were slightly worse than 20, and watching all the time at 1-in-16 chunks still missed by 12%.
  - At the spike a large fresh block is written with overlapping accesses, at about 1.3 footprint bytes per touched byte, while fresh blocks were counted at their touched bytes. Density per block fixed most of it: −9% at 5% watched (from −17%) and −5% at 20% (from −12%). At 5% the spike is not always inside a window.
- **Threads.** 1 and 4 threads agree within a few points on both inputs.

**Heap churn** (`tests/pintool/churn.c`): 64 live heap blocks of 16–112 KB, each freed and replaced in turn, of which only the first quarter (or half) is written; the live footprint peaks at 1.3 MB (2.5 MB). Each cell is mean error / peak error, %, 5% watched:

| Fraction of each block touched | Live allocated bytes | Residency | Windowed (written pages) |
|---|---|---|---|
| 25% | 259 / +268 | 248 / +269 | 23.9 / +22.2 |
| 50% | 92 / +93 | 91 / +90 | 12.9 / +12.1 |
| 100% | 2.4 / −1.7 | 0.9 / +0.5 | 0.8 / +0.5 |

- Residency fails because freed blocks' pages stay resident and are handed to later blocks.
- The windowed estimate's remaining error is page granularity: each block's touched prefix ends inside a 4 KB page, which adds up to 4 KB per block (64 blocks, about 0.25 MB).
- It is the same at every watched fraction (21.9–25.8% for 25% touched), because written pages, not windows, carry the estimate.

### Windows opened by large allocations

`-window_alloc N` (default 1 MB; 0 turns it off): an allocation of at least N bytes opens a window at once, at most one per period on top of the periodic windows, so at most twice the periodic fraction is watched. A large allocation is usually followed by the program filling the block, and a window then measures the block's density. A periodic window that starts inside such a window extends it.

Results with 1-in-100 chunks, about 20 periodic windows, mean error / peak error, %:

| Run | Periodic only | With allocation windows | Watched |
|---|---|---|---|
| miniVite 65536, 1 thread, 5% | 3.8 / −1.9 | 0.8 / −5.5 | 6.7% |
| miniVite 65536, 4 threads, 5% | 3.1 / −3.0 | 1.5 / −5.1 | 6.7% |
| miniVite 32768, 1 thread, 5% | 8.3 / −9.8 | 3.0 / −6.1 | 6.3% |
| miniVite 32768, 4 threads, 5% | 5.7 / −9.0 | 3.2 / −6.0 | 6.3% |
| miniVite 65536, 1 thread, 1% | 9.1 / −14.5 | 2.4 / −9.5 | 1.4% |
| miniVite 65536, 4 threads, 1% | 8.4 / −14.4 | 2.2 / −10.1 | 1.4% |
| miniVite 32768, 1 thread, 1% | 14.4 / −14.1 | 17.3 / −16.7 | 1.3% |
| miniVite 32768, 4 threads, 1% | 17.1 / −15.2 | 15.5 / −16.2 | 1.3% |

- Allocations of 1 MB or more opened 5–7 windows on miniVite (at most one per period).
- The mean error drops by 2–7 points, and the 32768 peak improves from −9% to −6%, but the 65536 peak gets worse (−2% to −5.5%).
- PolyBench (arrays allocated inside the first window) and the churn test (blocks under 1 MB) open no extra windows, and their results are unchanged.

**Why the 65536 peak gets worse: periodic windows were right by luck.** Single-thread reruns with the current tool (`-window_alloc 0` against the default) give peak errors of −2.6% and −4.4% at 5% watched. A third run watches the whole run (1-in-100 chunks) and gives −1.5%, the floor of the method; its per-block densities serve as the reference. A debug build wrote every block of 256 KB or more at each snapshot. At the true peak (94.6% of the run), per block:

| 5% watched | Fresh blocks: estimate − full watch | Reused blocks: estimate − full watch | Peak error |
|---|---|---|---|
| Periodic only | −3.75 MB | +3.09 MB | −2.6% |
| With allocation windows | +0.03 MB | −0.95 MB | −4.4% |

- **Periodic only:** two large blocks allocated late in the run carry errors that cancel.
  - A 25 MB fresh block allocated at 88% of the run is not inside any window. It counts at density 1.00 against its true 1.28 (−3.75 MB).
  - A 9 MB reused block allocated at 94% is not inside a window either. It takes the pooled density of reused blocks, 1.44, against its true 1.00 (+4 MB).
- **Allocation windows** open at both allocations and measure both blocks exactly (1.28 and 1.00).
  - The rest of the error is in a few 0.3–0.5 MB reused blocks allocated at the start, below the 1 MB trigger. Their true density is 1.9–2.4, from overlapping accesses throughout the run, and windows measure 1.1–1.3. Full watching removes it.
  - The guess above that windows opened at allocation bias density toward aligned fills is wrong: every triggered block gets its full-watch density.
- **32768** has the same structure without the luck. Its 12.6 MB fresh block falls inside a periodic window as well, so allocation windows only fix the reused blocks (−1.02 → −0.54 MB) and the peak goes from −10.1% to −5.7% (full watch: −1.9%).
- So allocation windows are the more reliable setting. What they still miss is small reused blocks whose overlapping accesses happen outside windows, which a lower trigger threshold would not reach either, since those blocks are allocated at the start.

### GAP and darknet as heap-reuse workloads

MemGaze used GAP (graph kernels) and darknet (neural networks). Copies are in `~/memory_estimator/workloads`. Single-thread windowed runs at 5% watched measured how much of their memory sits in reused heap blocks and how far those blocks' residency exceeds their written pages (the error that soft-dirty tracking removes):

| Run | Fresh blocks (resident) | Reused blocks (resident / written) | Residency − written, max |
|---|---|---|---|
| GAP bfs `-g 18 -n 8` | 71 MB (71) | 1.1 MB (1.1 / 1.1) | 1.0 MB |
| GAP pr `-g 18 -n 4` | 71 MB (71) | 1.1 MB (1.1 / 1.1) | 1.0 MB |
| GAP cc `-g 18 -n 8` | 71 MB (71) | 1.1 MB (1.1 / 1.1) | 1.0 MB |
| GAP sssp `-g 18 -n 4` | 130 MB (111) | 5.3 MB (5.3 / 5.3) | 1.0 MB |
| darknet AlexNet, 4 images, random weights | 513 MB (263) | 17 MB (17 / 17) | 1.1 MB |

- Neither is a heavy heap-reuse workload in the sense that breaks residency. Their large buffers are above glibc's mmap threshold (128 KB, raised dynamically up to 32 MB after frees), so they get new pages, and the reused blocks they do have are fully rewritten.
- Forcing large blocks onto the heap with `MALLOC_MMAP_THRESHOLD_=33554432` (glibc ignores values above 32 MB) moves more memory into reused blocks (GAP 5–7 MB, darknet 86 MB) but residency still exceeds written pages by at most 1 MB.
- They are still useful real workloads: darknet allocates 513 MB but touches 263 MB, so allocation-based estimates are 2× off, and GAP has large, irregularly accessed graphs.
- Better candidates for heap reuse are programs that repeatedly allocate buffers larger than they fill: interpreters, compilers, hash-table-heavy servers. They are tested below.

**Accuracy on GAP and darknet** (single thread, against full traces with `-track_frees`; mean error / peak error, %):

| Run | True peak | Allocated bytes | Nothing watched | 5% | 5%, allocation windows | 20%, allocation windows | Watching all the time |
|---|---|---|---|---|---|---|---|
| GAP bfs | 102.7 MB | 200.6 / −29.4 | 8.2 / −29.4 | 3.9 / −27.0 | 4.2 / −26.2 | 2.9 / −13.7 | 1.6 / −3.8 |
| GAP pr | 102.6 MB | 157.1 / −29.4 | 20.1 / −29.4 | 17.7 / −28.9 | 14.4 / −25.2 | 15.9 / −8.4 | 3.2 / −3.6 |
| GAP cc | 102.7 MB | 197.7 / −29.4 | 7.6 / −29.4 | 4.7 / −29.0 | 5.0 / −27.5 | 3.9 / −19.5 | 1.9 / −3.7 |
| GAP sssp | 168.2 MB | 183.7 / −19.4 | 7.5 / −20.6 | 3.4 / −8.8 | 4.0 / −14.7 | 4.0 / −10.0 | 2.4 / −0.0 |
| darknet AlexNet | 400.3 MB | 115.6 / +32.8 | 17.1 / −29.8 | 16.8 / −26.9 | 15.7 / −24.0 | 15.4 / −16.9 | 3.9 / +4.9 |

- **Watching all the time** is within 5% of the peak, so the error at 5% is not a limit of the method.
- **Density is the problem.** GAP's arrays have a density of 1.36 over the whole run and darknet's 1.47, and windows measure much less. For GAP bfs it is 1.02, 1.08 and 1.19 at 1%, 5% and 20% watched.
  - A block's density accumulates over the run: each new access pattern (another width, another offset) adds start addresses to memory already covered. A window sees only the patterns used while it is open, so it measures a lower density, and the shortfall shrinks only slowly with the watched fraction.
  - miniVite shows the same thing: its density is 1.05–1.12 at 1% watched and about 1.45 at 5% and 20%.
- With 4 threads GAP bfs and pr give the same picture (4.4 / −25.5 and 20.1 / −23.5 at 5%).

### Interpreters, a database and a compiler

Six programs that churn their heap, run under Pin with the system's binaries; native time and memory references in brackets:
- `python3` 3.6 serialising, parsing and sorting records in rounds of 10k–90k (3.5 s, 10.5 G);
- `lua` 5.3 growing tables and concatenating strings in rounds (2.5 s, 4.7 G);
- `perl` 5.26 doing the same with hashes and strings (1.4 s, 2.4 G);
- `sqlite3` 3.26, an in-memory database of 300k rows with indexes, deletes, a join and `VACUUM` (0.9 s, 2.5 G);
- GCC 8's `cc1` compiling the SQLite amalgamation at `-O0` (1.6 s, 2.8 G) and `-O1` (4.6 s, 7.3 G).

The scripts are in the session's scratch space, not in the repository.

**They are heap-reuse workloads.** Lua, Perl and SQLite keep 84–100% of their live memory in reused heap blocks (44, 46 and 59 MB at the peak, in 13k–650k blocks). Python and cc1 keep most of theirs in fresh blocks: pymalloc's 256 KB arenas and GCC's garbage-collected pages are mapped with `mmap`.

**Residency still does not break.** The reused blocks' residency exceeds their written pages by at most 0.5 MB at any point of any run, except Python (2.8 MB). The blocks are reused, but each new block is written all over.

**The windowed estimate does break**, for other reasons (mean error / peak error, %, allocation windows on):

| Program | True peak | Allocated bytes | 1% | 5% | 20% | Watching all the time |
|---|---|---|---|---|---|---|
| python | 123.7 MB | 26.2 / −20.3 | 20.8 / −17.7 | 17.3 / −17.6 | 12.2 / −10.6 | — |
| lua | 83.0 MB | 40.6 / −36.8 | 34.6 / −32.8 | 30.3 / −29.5 | 16.4 / −20.4 | 5.4 / −6.9 |
| perl | 69.0 MB | 20.6 / −22.3 | 16.5 / −17.2 | 14.6 / −10.3 | 9.7 / −13.1 | 2.0 / +2.6 |
| sqlite | 312.6 MB | 56.0 / −76.3 | 42.9 / −69.7 | 41.5 / −66.3 | 35.7 / −58.8 | 17.4 / −36.2 |
| cc1 `-O0` | 132.3 MB | 9.5 / −13.7 | 9.5 / −10.5 | 5.7 / −5.3 | — | — |

—: not run. cc1's 20% run and its `-O1` windowed runs were stopped (see Cost). Python's full-watch run was not made.

Three things go wrong, in order of size:

1. **Density accumulates, as in GAP.** Lua's reused blocks have a density of 1.36 over the run, and windows at 5% measure 1.11. Perl: 1.21 against 1.07.
2. **The ground truth counted allocator metadata as live memory** (now fixed, see below). The splitter removes `[ptr, ptr + size)` when a block is freed. glibc then writes into the freed chunk (free-list links), and every allocation writes a chunk header just below its block. Those writes are counted again and never released.
   - A test program that does 3 rounds of allocating, writing and freeing 200,000 blocks of 32–64 bytes leave 5.7 MB "live" at the end, and the peak reads 23.4 MB against about 14.4 MB of program data.
   - Lua ends its run with 16.7 MB "live" and 0.1 MB allocated. SQLite ends with 104 MB against 16.5 MB, though part of that is the overlap described in 3.
   - The windowed estimate sees this memory only as memory outside blocks, and only inside windows: 1.0 MB for Lua at 5% watched against 7.0 MB when watching all the time.
   - miniVite (8 million malloc/free calls) is affected less: at most the 2.2 MB of its 36 MB peak that lies outside blocks, which also holds its stack and MPI buffers.
3. **SQLite's footprint is not its memory.** Its process peaks at 80 MB resident. Its footprint, the largest access at each start address, peaks at 313 MB, because it reads records and keys at almost every byte offset with overlapping word and vector accesses. Near the end of the run, the live blocks hold 260 MB of footprint over 83 MB of covered bytes (3.2×). Even watching all the time, written pages × density misses the peak by 36%.

**Cost.** Lua, Perl and SQLite with 5% watched took 61–81 s (16 runs at once), and the full trace took 176–319 s. Watching all the time took 53–75 s (3 runs at once), so the windows save almost nothing: the cost is in the allocator hooks (hundreds of thousands of live blocks under a lock) and in clearing soft-dirty bits. cc1 is pathological: 13–18 minutes windowed against 4.7 minutes for its full trace. GCC maps and unmaps its garbage-collected pages constantly, and every `mmap` harvests and clears the soft-dirty bits of all reused blocks. Python took 151–199 s windowed against 700 s for the full trace.

**What this means:**
- Heap reuse as such is handled: written pages track reused blocks in every program tried.
- The weak point is density. Windows underestimate it whenever a block's access patterns change over the run, which is the normal case in real programs (GAP, darknet, Lua, Perl), and by 20–30% at the peak at 5% watched.
- The ground truth should not count allocator metadata. This is now fixed (below).

**Ground truth fixed: the allocator's own accesses no longer count.** With `-track_frees` or windows, the tool finds glibc's `malloc.c` code by symbol name, both the public functions and internal ones such as `_int_free`. Memory accesses made by that code count as time but go into no footprint. memset and memcpy called by `calloc` and `realloc` still count. Default splitter and sampler outputs are byte-identical to before (checked with address randomisation off), and the test suite passes.
- The test program above now peaks at 14.45 MB, against about 14.4 MB of program data (was 23.4), and ends at 1.7 MB, its 1.6 MB static array (was 7.3).
- Re-scored with the fixed tool: new truth for every run, single thread, mean error / peak error, %:

  | Run | True peak, old → new | Live at the end, old → new | 5% | 5%, allocation windows | Watching all the time |
  |---|---|---|---|---|---|
  | miniVite 65536 | 36.1 → 35.5 MB | 3.5 → 2.8 MB | 4.6 / −4.2 | 2.6 / −4.7 | 5.2 / −1.0 |
  | miniVite 32768 | 19.9 → 19.3 MB | 3.5 → 2.8 MB | 6.3 / −8.0 | 4.3 / −4.1 | 5.8 / +0.0 |
  | GAP bfs | 102.7 → 102.7 MB | 0.5 → 0.5 MB | 3.8 / −26.5 | 3.7 / −26.7 | 2.6 / −3.6 |
  | darknet AlexNet | 400.3 → 399.4 MB | 390 → 390 MB | 16.8 / −25.5 | 15.7 / −23.8 | 3.7 / +4.8 |
  | python | 123.7 → 125.9 MB | 6.2 → 6.2 MB | 16.5 / −16.6 | 16.6 / −18.8 | 3.7 / −2.8 |
  | lua | 83.0 → 70.9 MB | 16.7 → 0.4 MB | 9.4 / −15.2 | 12.2 / −18.9 | 6.4 / −2.9 |
  | perl | 69.0 → 64.5 MB | 50.0 → 42.2 MB | 7.5 / −11.9 | 7.6 / −5.4 | 2.8 / +3.2 |
  | sqlite | 312.6 → 263.2 MB | 103.9 → 103.4 MB | 37.9 / −58.7 | 41.3 / −60.6 | 17.1 / −24.9 |

- Metadata made up 15% of Lua's peak, 7% of Perl's and 16% of SQLite's. For miniVite, GAP and darknet the truth moved by at most 3%, so the earlier results on them stand within a few points.
- Python's truth went up by 2 MB, which the fix cannot cause. Python randomises string hashing per run, so its runs differ slightly.
- With the fixed truth, watching all the time is within 5% of the peak everywhere except SQLite. At 5% watched the peak is still 4–27% low, apart from SQLite: density remains the main error.
- SQLite still ends with 103 MB "live" against 16.5 MB allocated. That remainder is overlap (point 3), not metadata.
- Whether footprint should count overlapping accesses at all is worth deciding: for byte-oriented programs like SQLite it is several times the memory touched.

**darknet exposed a bug.** The tool recorded 53 KB of darknet's 513 MB. glibc's `calloc`, the first time the allocator is used, calls `malloc` through an initialization function and leaves with a jump to `memset`, so its exit is never seen. The tool then treated every later allocation made from deeper in the stack as nested inside it and recorded none. A call now counts as nested only if it comes from inside an instrumented allocation function and runs deeper in the stack, and a block returned by a call nested in a malloc-like call is recorded at once. `tests/pintool/first_calloc.c` checks this; default outputs are unchanged. miniVite, GAP and PolyBench were not affected, because their first allocation is a `malloc`.

### Cost

Seconds, quiet machine, one run each, 1-in-100, about 20 windows:

| Workload | Native | Pin, no tool | Full spatial | Nothing watched | 1% | 5% | 20% |
|---|---|---|---|---|---|---|---|
| 2mm LARGE | 7.5 | 8.1 | 211 | 15.8 | 16.9 | 23.1 | 43.6 |
| gemm LARGE | 6.0 | 6.1 | 215 | 11.7 | 13.1 | 19.0 | 37.0 |
| jacobi-2d LARGE | 11.3 | 11.6 | 236 | 21.5 | 23.3 | 31.2 | 59.4 |
| miniVite 65536, 1 thread | 6.5 | 11.5 | 163 | 59.6 | 79.4 | 84.3 | 104.1 |
| miniVite 65536, 4 threads | 6.1 | 11.0 | 161 | 72.3 | 85.8 | 93.0 | 111.2 |

- **Full traces** for the truth took 1237–1717 s. Those ran concurrently, so they are not comparable to this table.
- **Outliers:** one timing of jacobi-2d at 5% watched took 173 s and a rerun took 31.2 s, which the table shows. With the previous tool, one miniVite run at 20% took 199 s against 100 s on a rerun.
- **Soft-dirty tracking** costs about 14% on miniVite at 5% watched (73.7 → 84.3 s with 1 thread, 81.5 → 93.0 s with 4) and 4–14% on PolyBench, against the tool without it. With 1 thread miniVite clears the bits 8509 times; recording only blocks not yet fully written brought that run from 138 s down to 86 s.
- **PolyBench:** with nothing watched, the cost is the per-block reference counter, about 2× native. Each further 1% watched adds about 1% of the full spatial cost.
- **miniVite** has a higher floor. It makes 8 million malloc/free calls at 65536, each hooked and recorded under a lock. On miniVite 16384, turning the allocator hooks off saved about 7 s of 22.

### Implementation notes

- **Switching windows.** `PIN_RemoveInstrumentation` doesn't change code that is running: a loop keeps jumping back into its old translation. In 2mm, a window that had "closed" kept sampling for the rest of the loop nest.
  - Each thread therefore checks the window state every few thousand references. If it changed, the thread restarts at its current instruction (`PIN_ExecuteAt`) in freshly instrumented code.
  - Accesses recorded by stale code after a window closes are dropped.
- **Reference counting.** Counting per basic block matches the splitter's per-access count to within 0.0001% on miniVite (1.78 billion references). A REP string instruction counts once, not once per iteration.
- **Footprint per block** comes from the chunk sample, sorted, as the union of the selected accesses' byte ranges.

### Limitations

- **Written pages miss reads.** A reused block is measured by the pages written since it was allocated, so memory it only reads (rare for heap memory) is missed. Blocks under 16 KB do not clear the bits when allocated, so they can inherit marks from earlier writes since the last clear; they share pages with other blocks anyway. With several threads, writes made by other threads between a `brk`/`mmap` system call's entry and exit are lost.
- **Page granularity:** up to 4 KB per block, which dominates for programs with many small, partly touched blocks (about +22% on the churn test).
- **Density** needs a block (or all reused blocks) to have a few windows' worth of chunks (64 KB covered) before it moves off 1, and fresh blocks never seen in a window count at 1.
- **Not tracked:** `mremap` outside realloc and stack frames. Memory outside blocks is estimated from windows only.
- **Selection is by absolute address,** so spatial and windowed outputs change between tool builds (the app's mmap addresses move). Splitter and sampler outputs stay byte-identical.

### Reproduce

    scripts/run.sh polybench --bench 2mm --configs LARGE --mode spatial -i 100 -s 20 --runs 1 \
        --snapshot 7000000 --window 85000000 --period 1700000000      # 5% watched, ~20 windows
    scripts/run.sh polybench --bench 2mm --configs LARGE --mode splitter --runs 1 \
        --intervals 1000 --bins 1 --snapshot 7000000 --track-frees     # truth
    python -m memprint timeline windowed 2mm                           # from analysis/

## Headline: spatial (address) sampling

`-mode spatial -i R -s 20` selects 1-in-R *addresses* with a salted hash and records every access to them. An address is then in the sample with probability 1/R whatever its access pattern, so the footprint estimate is simply the selected footprint × R, with no model and no training. The 20 hash buckets give each snapshot a standard error.

This removes the core difficulty of all the reference-sampling estimators below. With reference sampling, an address accessed r times is seen with probability 1 − (1 − p)^r, and r is unknown and grows with input size.

Selected accesses are rare, so they are recorded immediately, under a lock, as they happen; frees are applied as they happen too. The order across threads is therefore exact, and time is the true count of references executed.

### PolyBench

Held-out runs (largest and middle size of 2mm, gemm, jacobi-2d, atax; truth from the splitter); mean absolute error, extrapolation / interpolation, %:

| Method | MAPE | Error of peak | Error at peak | Runtime, MEDIUM (2mm / gemm / jacobi-2d) |
|---|---|---|---|---|
| Spatial 1/25 addresses | 0.9 / 3.4 | 0.8 / 3.4 | 0.8 / 3.4 | 6.0 / 3.9 / 4.2 s |
| Spatial 1/100 | 2.3 / 4.2 | 2.3 / 3.9 | 2.3 / 4.0 | 3.7 / 3.3 / 3.4 s |
| Spatial 1/250 | 2.3 / 9.9 | 2.4 / 9.4 | 2.3 / 9.6 | 3.5 / 3.4 / 3.3 s |
| Spatial 1/1000 | 6.7 / 17.5 | 6.6 / 14.9 | 6.6 / 27.2 | 3.3 / 3.2 / 3.1 s |
| Best reference-sampling hybrid, `-i 25` | 24.3 / 8.7 | 21.0 / 12.2 | 38.3 / 24.0 | 7.6 / 6.5 / 8.2 s |
| Best reference-sampling hybrid, `-i 3` | 4.8 / 4.2 | 6.2 / 6.4 | 9.1 / 9.4 | ~25 / 20 / 32 s |
| Splitter (full trace) | — | — | — | 41.3 / 31.8 / 47.9 s |

- **Accuracy follows the binomial prediction,** relative error ≈ 1/√(addresses / R).
  - The largest sizes (1–1.6 MB, about 130–200k addresses) stay within 1–7% down to 1/1000.
  - The middle sizes (about 0.22 MB, about 28k addresses) need 1/100 or denser. At 1/1000 only about 28 addresses are selected.
  - The error is largest early in a run, when the footprint is still tiny.
- **The bucket error bar is honest at dense rates.** The truth lies within ±2 standard errors in 100% of snapshots at 1/25 and 1/100, 88% at 1/250 and 77% at 1/1000 (nominal 95%; a 20-bucket standard error is itself noisy).

### miniVite and multithreading

miniVite was rebuilt against spack's OpenMPI 5.0.5, the MPI behind the paper's runs, and run with `OMP_WAIT_POLICY=passive`.
- **Footprint:** a full trace of 1024 vertices gives 6.8–7.0 MB cumulative footprint, matching the paper's 6.84 MB. With the system OpenMPI it was ~175 MB, mostly MPI start-up memory.
- **References:** 44 million instead of the paper-era 1.76 billion. About 97% of those were OpenMP threads spinning while they waited.

Every spatial run is compared with the full trace of the same size and the same thread count, with no training. The live peak with `-track_frees` is 4.9, 6.2 and 7.9 MB at 1024, 4096 and 8192 vertices, the same at every thread count.

| Threads | Spatial 1/25 | 1/100 | 1/250 | 1/1000 |
|---|---|---|---|---|
| 1 | 0.2–0.9% | 0.4–1.0% | 2.0–3.3% | 1.9–4.1% |
| 4 | 0.2–1.3% | 1.0–1.6% | 1.2–4.9% | 1.3–6.0% |
| 16 | 0.4–0.8% | 0.6–2.0% | 1.3–2.3% | 1.5–6.1% |

(MAPE range over the three sizes.)

- **Peaks and error bars hold under threads.** Error of peak is ≤ 2.8% in every case. The truth lies within ±2 standard errors in 94–100% of snapshots, and run lengths match the full trace exactly.
- **Cost, 8192 vertices:**

  | Threads | Native | Splitter | Spatial (1/25 … 1/1000) |
  |---|---|---|---|
  | 1 | 0.24 s | 168 s | 28–32 s |
  | 4 | 0.20 s | 198 s | 28–41 s |
  | 16 | 0.28 s | 176 s | 26–38 s |

  The per-access lock is not a bottleneck at these rates.
- **Multithreading bugs found and fixed.** `tests/pintool/parallel.c` has 8 threads with 12 MB live at a barrier.
  - Buffered modes kept each thread's accesses in its own trace buffer, so a free in one thread could be applied before another thread's earlier accesses: the full-trace peak came out at 9.8–10.8 MB.
  - With `-snapshot`/`-track_frees` the buffer is now 16 pages, giving 12.08 MB on every run; default outputs are unchanged.
  - A first spatial fix held frees back with vector clocks. It was correct, but on miniVite, which frees constantly while MPI helper threads rarely flush, it took 7474 s. Recording spatial accesses immediately replaced it: 12 runs give 12.03–12.27 MB, with 28–40 s on miniVite.

### Caveats

- Small structures are sampled at R too, so a single small buffer is either missed or over-weighted, and there is no per-object breakdown at sparse rates.
- Hashing start addresses mirrors the paper's footprint definition (largest access size per start address).

## Data

- **PolyBench:** 2mm, gemm, jacobi-2d and atax at MINI…MEDIUM (7 sizes).
  - One splitter run per size, using the 13 standard intervals with 20 bins each.
  - Sampler runs with `-s 20 -r 20` (λ = 1, nearly independent bins) at `-i 25` and `-i 50`.
  - `-i 3` runs for the held-out configs (MEDIUM, SMALL).
  - Snapshots every ~1/200 of each run; `-track_frees` on.
- **Evaluation:**
  - The largest (extrapolation) and the middle (interpolation) size are held out, as in the paper.
  - Error is MAPE over the snapshots of the held-out run, plus the relative error of the peak.
  - Sampler curves are compared on the fraction of their own run. For these single-threaded programs the sampler's estimated reference count is within 0.5% of the true count.

## What the tool now records

Each footprint keeps, per address, how often it was sampled, and reports f1..f4: the number of addresses sampled exactly 1, 2, 3 and 4 times. The splitter also keeps the union of each interval's 20 bins (timeline rows with `Bin = -2`), which is a 1-in-interval/20 sample. The sampler's main row is the union of its bins. Summary files are unchanged.

## Results

Mean absolute error over the four kernels, in %. Metrics:
- **MAPE** is over the snapshots of the held-out run.
- **Error of peak** compares the estimated maximum with the true maximum, wherever each occurs.
- **Error at peak** reads the estimate at the moment the true footprint peaks.

| Estimator | Sampler run | Extrap. MAPE / error of peak / error at peak | Interp. MAPE / error of peak / error at peak |
|---|---|---|---|
| α model per snapshot (paper features, NZ) | `-i 25`/`-i 50` | 45 / 30 / 58 | 31 / 59 / 49 |
| Chao1 | `-i 25` | 28 / 9 / 33 | 57 / 39 / 60 |
| iChao1 (adds f3, f4) | `-i 25` | 27 / 11 / 33 | 55 / 36 / 58 |
| Known-rate (`richness.py`) | `-i 3` | 12 / 26 / 17 | 7 / 23 / 10 |
| **Hybrid** (known-rate × learned correction) | `-i 3` | **6 / 11 / 12** | **5 / 9 / 8** |
| Hybrid | `-i 25` | 23 / 21 / 35 | 12 / 24 / 20 |
| Hybrid | `-i 50` | 61 / 160 / 119 | 24 / 33 / 39 |

From the splitter's own dense union (1 in 5), known-rate alone gives 12% (extrapolation) and 7% (interpolation) MAPE; the α model gives 42% and 31%, and Chao1/iChao1 38–40% and 29–30%.

The **hybrid** regresses log(true / known-rate estimate) on the training sizes' splitter unions at the sampling rate closest to the run being predicted (Ridge, α = 1). Features:
- how far the known-rate estimate extrapolates beyond the observed footprint;
- log((f2+1)/(f1+1)), the fraction of samples that were new addresses, f1/seen, and the sampling rate;
- the bins' spread σ, mean observed footprint and interval (the α model's inputs).

Per kernel, hybrid, MAPE / error of peak:

| Kernel | `-i 3` extrap. | `-i 3` interp. | `-i 25` extrap. | `-i 25` interp. |
|---|---|---|---|---|
| 2mm | 6 / +9 | 6 / +15 | 20 / +20 | 18 / +39 |
| gemm | 3 / +9 | 5 / +9 | 16 / +3 | 8 / +16 |
| jacobi-2d | 5 / +7 | 2 / +4 | 31 / −20 | 10 / +21 |
| atax | 11 / +19 | 5 / +8 | 24 / +40 | 11 / +19 |

jacobi-2d's error at peak stays at −15% to −25% at every density, because its true peak is in the very first snapshot.

**Correction to earlier numbers:** a sampler's union holds only the sampled references that land in at least one bin, a fraction 1 − e^−λ of them, so its rate is (1 − e^−λ)/i, not 1/i. The first known-rate sampler numbers on this branch used 1/i, which overstated the sampled fraction 1.6×. They were 17% / 20% MAPE and are now 12% / 7%. `timeline.union_rate` recovers λ from the bin rows. Splitter unions were not affected.

## Improving the hybrid at `-i 25`

The steps were tested in the order below. All numbers are mean absolute error over the four kernels at `-i 25`, extrapolation / interpolation, in %.

| Step | MAPE | Error of peak | Error at peak |
|---|---|---|---|
| Before: correction trained on the splitter union with the nearest rate (1/37 for a 1/40 sampler) | 22.7 / 11.8 | 20.8 / 23.7 | 34.9 / 20.1 |
| 2. Correction trained at exactly the sampler's rate (splitter `-intervals 95,253,791,1582,3164,7910`) | 24.7 / 11.9 | 25.2 / 23.6 | 36.7 / 18.2 |
| 2 + 1. Estimate made non-decreasing between frees (isotonic per segment) | 23.8 / 10.1 | 19.8 / 15.7 | 37.9 / 19.0 |
| 2 + 3. One reuse shape per run, fitted on the run's data-rich snapshots | 24.2 / 11.7 | 30.5 / 20.6 | 46.3 / 18.7 |
| 2 + 1 on fresh runs (baseline for step 4) | 23.8 / 10.0 | 22.9 / 13.8 | 43.4 / 24.6 |
| 2 + 1 + 4. Feature: share of new addresses among the last 5 snapshots' samples (new `Discovered` counter) | 24.3 / 8.7 | 21.0 / 12.2 | 38.3 / 24.0 |

- **Step 2** matters when the old rate mismatch was large. At `-i 50` (1.6× off before), extrapolation MAPE drops from 61% to 28–36%. At `-i 25`, where the mismatch was 8%, it changes nothing.
- **Step 1** reliably lowers the error of the peak, and is best at dense rates. At `-i 3` it gives 3.8–5.3% / 3.9–4.2% MAPE and 2.5–7% error of peak.
- **Step 3** does not help: the per-snapshot shape is no worse once the correction is applied.
- **Step 4** helps interpolation (MAPE 10.0 → 8.7) and the error at peak, but not extrapolation MAPE.
- **What remains is a size bias.** At `-i 25` the largest size is underestimated by 13–47% (2mm −11% to −36%, gemm −10% to −15%, jacobi-2d −28% to −47%, atax −27% to +11%), mostly in the first half of the run. Interpolation is almost unbiased.
  - The correction is learned on the smaller sizes, and at the same rate and phase the largest size's sample statistics fall outside that range.
  - None of the four steps addresses that.
- **Size-invariant features do not fix it either.** The bins' σ and mean (bytes) were replaced by the relative spread σ/mean and the bin/union footprint ratio, with smoothing and the discovery feature on, at `-i 25`:

  | Feature set | MAPE | Error of peak | Error at peak |
  |---|---|---|---|
  | Original (σ and bin mean in bytes) | 24.3 / 8.7 | 21.0 / 12.2 | 38.3 / 24.0 |
  | Size-invariant (ratios only) | 24.0 / 15.4 | 21.9 / 16.5 | 31.1 / 24.1 |

  - Per kernel, extrapolation MAPE improves for jacobi-2d (37.5 → 27.7) and atax (24.7 → 21.0) but worsens for 2mm (18.5 → 24.3) and gemm (16.3 → 23.1).
  - Interpolation gets worse, and at `-i 50`..`-i 250` it is unstable (43–72% interpolation MAPE, against 13–17%).
  - So the byte features carry useful information inside the training range and are not what causes the bias. The original set stays the default; the invariant one is `HYBRID_FEATURE_SETS["invariant"]`.
  - The bias more likely sits in the sample statistics themselves. In 2mm and gemm each element is reused about N times, so a larger input means more reuse per address. At the same sampling rate the largest size then shows more repeats than any training size did, whatever units the features use.
- **Remaining options:** train the correction on the larger sizes only (the tiniest ones may mislead it), constrain it to extrapolate monotonically in the size-related features, or add a feature for reuse per address relative to the known rate (e.g. the expected samples per address implied by the known-rate fit).

## The hybrid between `-i 3` and `-i 25`

Where the hybrid falls off, and what each density costs. Sampler runs `-s 20 -r 20` at `-i` 3, 5, 8, 12, 25, 50, 100 and 250 on the held-out sizes (MEDIUM, SMALL) of 2mm, gemm, jacobi-2d and atax. The correction is trained on splitter unions at exactly each sampler's rate (splitter `-intervals 95,158,253,380,791,1582,3164,7910`, 20 bins). Variant: known-rate per snapshot, smoothed between frees, with the discovery feature. Mean absolute error over the four kernels, extrapolation / interpolation, %; time is MEDIUM, averaged over 2mm, gemm and jacobi-2d (3 runs each, one at a time on a quiet machine):

| `-i` | MAPE | Error of peak | Error at peak | Time, MEDIUM (2mm / gemm / jacobi-2d) | Of a full trace |
|---|---|---|---|---|---|
| 3 | 3.6 / 4.9 | 2.0 / 6.1 | 9.1 / 8.6 | 24.5 / 18.1 / 28.8 s | 44% |
| 5 | 5.9 / 5.8 | 2.7 / 7.6 | 14.8 / 14.2 | 16.8 / 12.8 / 19.7 s | 30% |
| 8 | 14.2 / 9.5 | 19.9 / 16.4 | 30.1 / 18.7 | 11.1 / 8.9 / 13.0 s | 20% |
| 12 | 21.8 / 12.4 | 65.1 / 20.4 | 34.9 / 13.7 | 9.3 / 7.5 / 10.8 s | 17% |
| 25 | 24.5 / 10.4 | 28.1 / 17.1 | 45.5 / 20.4 | 6.4 / 5.4 / 6.9 s | 12% |
| 50 | 30.7 / 13.2 | 47.4 / 11.3 | 64.0 / 27.1 | 4.9 / 4.3 / 5.1 s | 9% |
| 100 | 36.3 / 18.2 | 32.4 / 22.2 | 64.5 / 30.1 | — | — |
| 250 | 40.1 / 16.9 | 24.5 / 21.5 | 55.6 / 44.1 | — | — |

The full trace (splitter, 8 intervals) took 54.4 / 41.2 / 68.4 s; native runs take 0.05–0.09 s.

- **The hybrid holds to `-i 5` and falls off between 5 and 8.** At `-i 5` it is within 4–10% MAPE and −6% to +11% of the peak on every kernel and split, for 69% of the cost of `-i 3` and 30% of a full trace. At `-i 8` 2mm's extrapolation goes to 27% MAPE and +40% on the peak, and atax's peak to +29%.
- **Past the fall-off it overshoots the peak.** At `-i 12` 2mm extrapolation is +153% on the peak, worse than at `-i 25` (+18%). The correction then extrapolates beyond its training range.
- **Cost scales with the sampled fraction, on top of a fixed floor.** Going from `-i 3` to `-i 5` saves 31%, and to `-i 8` 53%. At `-i 25` and `-i 50` a run still costs 9–12% of a full trace.
- Per kernel, MAPE / error of peak:

  | Kernel | `-i 5` extrap. | `-i 5` interp. | `-i 8` extrap. | `-i 8` interp. |
  |---|---|---|---|---|
  | 2mm | 5 / +4 | 7 / +11 | 27 / +40 | 11 / +17 |
  | gemm | 4 / +3 | 4 / +6 | 9 / +10 | 9 / +14 |
  | jacobi-2d | 4 / +0 | 7 / −6 | 12 / −1 | 7 / −6 |
  | atax | 10 / +3 | 5 / +8 | 9 / +29 | 12 / +29 |

- The `-i 3` and `-i 25` rows are close to the earlier results (3.8–5.3 / 3.9–4.2 and 24.3 / 8.7 MAPE). The sampler runs at 3, 8, 25, 50, 100 and 250 are the earlier ones; the splitter runs and the `-i 5` and `-i 12` sampler runs are new.

## What we learned

1. **A single bin cannot see reuse.** Each bin holds 0.1–10% of the footprint and almost never samples an address twice. So a model working from one bin, like the α model, can't tell new memory from memory being touched again: on a plateau its estimate keeps rising.
2. **Chao1/iChao1 fix plateaus but are biased elsewhere.**
   - jacobi-2d extrapolation falls from 42–52% to 4% (splitter union) and 9% (sampler).
   - Kernels with many addresses touched only once or a few times (initialisation writes) are underestimated by 30–60% at moderate density, because Chao is a lower bound when reuse is uneven.
   - At the densest union (1 in 5) Chao overshoots by +50–245%, because f2 is much smaller than f1.
   - The sampler shows exactly the same bias as splitter unions of the same density.
3. **Using the known rate is the key.**
   - The sampling rate is known exactly: sampled references / references executed. Together with the reference count T and the seen count S, it pins down the access-count distribution well once the sample has repeats.
   - Dense unions (1 in 5): bias +1% to −12%, against +53% to +245% for iChao1.
   - At 1 in 12: within about ±10–20%.
   - At 1 in 50 and sparser: −90% to +215%, because f2..f4 are too small to identify the reuse shape.
4. **The reuse shape does not transfer between sizes.** Fixing the negative-binomial shape from the training sizes and solving only from S, T and p gives 40–150% error. The fitted shapes are tiny (0.02–0.08): real reuse is a mix of touched-once memory and heavily reused data, not one smooth distribution.
5. **Cost.** A sampler run dense enough for the known-rate estimator (`-i 3`) takes about 40–55% of a splitter run, but about 5× a sparse `-i 50` run. MEDIUM timings: 2mm 24.7 s vs 58.3 s (splitter) vs 5.4 s (`-i 50`); gemm 19.8 vs 43.5 vs 4.8; jacobi-2d 31.9 vs 67.4 vs 5.3. These were measured with other runs in parallel, so they are rough. Unlike the α model, the estimator needs no training.
6. **A learned correction makes known-rate usable at sparser rates.**
   - The known-rate estimator's error is systematic: it underestimates while addresses have not yet been reused, which is exactly when f2/f1 is low. A regression on the training sizes learns this.
   - With the dense sampler, the corrected estimate is within 2–11% MAPE on all eight held-out runs.
   - With `-i 25` (about 1/5 of the cost) it gets 8–31%, against 35–102% raw.
   - At `-i 50` it is unstable.
7. **The error of the peak and the error at the peak are different questions.** The estimators get the maximum memory requirement right well before they get it right at the moment it happens. Example: raw known-rate on 2mm MEDIUM is +1% on the peak value but −19% at the peak time, because the estimate only catches up when the program starts re-reading its arrays halfway through.

## miniVite

Re-measured with `OMP_NUM_THREADS=4 OMP_WAIT_POLICY=passive`, using the f1/f2-only build, so Chao1 but not iChao1 or known-rate.

- **Stable run lengths:** with pinned, passive threads the sampler and splitter runs now agree within 1% on references executed. Unpinned, OpenMP threads spin and runs differed by up to ±50%.
- **Footprint not representative:** it is ~175 MB at every size from 1024 to 8192 vertices, against 7–14 MB in the paper's runs. The likely cause is that the system OpenMPI (`mpi/openmpi-x86_64`) touches large buffers at start-up; the paper used spack's openmpi-5.0.5. Until miniVite is built against that MPI, these numbers mostly measure MPI start-up memory.
- **Results on this data:**
  - α model: 22–25% (extrapolation) and 14–21% (interpolation) from splitter bins; 25–33% from the sampler.
  - Chao1: 34–35% from splitter unions; +219–241% from the sampler (the sparse-union overshoot).

## Open questions / next steps

- **Windowed density** is moot with bytes touched (density 1). It still applies to `-footprint starts`.
- **Ground truth:** allocator accesses are now excluded for glibc. Other allocators (jemalloc, tcmalloc, pymalloc's arenas inside Python) are neither tracked as blocks nor excluded.
- **Footprint definition:** decided: bytes touched with `-track_frees` or `-window`. The reference-sampling estimators (α model, Chao, known-rate, hybrid) have not been re-run against bytes-touched truths. Their PolyBench numbers should barely move (arrays of aligned doubles), but miniVite's would.
- **Windowless mode:** with bytes, pages alone are within 8% of every peak tested except where memory lies outside blocks. Watching 1% covers that; a mode that drops windows entirely would be cheaper still.
- **Lua is 5–8% low** at every watched fraction, including full watching.
- **Windowed cost on allocation-heavy programs:** the allocator hooks and soft-dirty clears around `mmap` dominate (cc1 runs 3–4× slower windowed than fully traced).
- **Hybrid below `-i 5`:** the correction overshoots once it extrapolates past its training range (`-i 8` and sparser); constraining it might move the fall-off.
- **Generalisation:** the correction is trained per workload on its smaller sizes, like the paper's α model. It has not been tested across workloads, or on miniVite.
- **Reuse model:** a two-class mixture (touched-once plus reused) instead of one negative binomial might transfer between sizes and allow sparser sampling.
- **miniVite:** build against spack openmpi-5.0.5 and re-measure with all three estimators.
- **Forecasting** is unchanged from `main`: the peak is within about 10–20% for interpolation and unreliable for extrapolation.

# Sampling short runs, huge pages and allocators the shim cannot query

Three questions, fixed before any run below.

1. Short runs. Does sampling faster, or sampling on memory growth, recover the peaks that the 5 ms sampler misses in
   runs of a few milliseconds, and what does each sampler cost?
2. Huge pages. How much memory do transparent huge pages add to the footprint that a node actually holds, and can any
   kernel interface tell which 4 KiB pages of a huge page were touched?
3. Allocators the shim cannot query. Does routing CPython's object allocator through glibc (`PYTHONMALLOC=malloc`)
   make the data estimate exact, and how much of `cc1`'s memory is in GCC's own garbage-collected heap?

## Samplers (tools/mprshim.c)

- `pss` at 5 ms (the model's sampler, as published) and at 1 ms (the blind test's).
- `fast`: `RssAnon` and `VmRSS` from `/proc/self/status` every 100 us; glibc's free bytes (mallinfo, which walks the
  arenas) every 1 ms, the last value held in between. Columns as the `info` mode.
- `faults`: a software page-fault event per CPU (`perf_event_open`, `PERF_COUNT_SW_PAGE_FAULTS`, inherited by the
  program's threads) with a sample period of 256 faults; the sampler thread waits on the events and takes a sample
  (status and mallinfo) after every 256 faults, so at most every 1 MiB of new 4 KiB pages, and at least every 5 ms.
  Under THP one fault maps 2 MiB, so the period is in faults, not bytes.
- Every mode also writes `VmHWM` (the kernel's peak RSS) at exit to `<MPR_OUT>.hwm`.

## Short runs: measurements

- (a) The transfer study's smaller inputs (`run_transfer.sh`, 44 runs): 3 runs under `pss` 5 ms, `pss` 1 ms, `fast` and
  `faults`. Per run: the sampled peak RSS over `VmHWM` (the share of the true RSS peak that the sampler sees), and the
  peak of the data estimate.
- (b) The transfer to the largest input (`compute.transfer`, rule unchanged) rebuilt from each sampler's first run,
  scored against the pages-in-use truth: median and per-program signed and absolute whole-run error, NPB-MG, 2mm,
  gemm and jacobi-2d named in advance.
- (c) The blind configurations (`blind/run.sh`) under `fast` and `faults`, scored by `blind/score.py`'s rule against
  the same truths, against the published 1 ms `pss` result.
- (d) Cost: 3 runs of each program at its largest input, plain and under each sampler, interleaved, one at a time;
  the median wall time over the plain run's. The model's error at the largest input under each sampler.

## Huge pages: measurements

- From the existing runs (`data/thp.csv`, `alloc*.csv`, `llm*.csv`): per program and allocator, the peak anonymous
  RSS with THP enabled over that with THP disabled for the process (the memory huge pages add). No new runs.
- A test program maps 64 MiB, touches one 4 KiB page in each 2 MiB, and reports `AnonHugePages`, `RssAnon` and the
  page flags readable without root, with THP enabled, after `MADV_NOHUGEPAGE` on the range, and with THP disabled for
  the process. Expected: 64 MiB, 128 KiB, 128 KiB.

## Allocators: measurements

- CPython: the corpus program `python` at its three inputs under `pss` with `PYTHONMALLOC=malloc`, 3 runs; the pages-in-use
  truth of the largest input traced again under `PYTHONMALLOC=malloc` (scripts/footprint_corpus.sh truthpages, the
  corpus's Pin tool). Reported: the data estimate against both truths; the change in the truth, peak RSS and wall time
  that the environment variable causes.
- `cc1`: a shim mode that interposes `mmap` and `munmap` (glibc's malloc maps through internal calls that the
  interposition does not see, so what it counts is the program's own mappings, GCC's GC pages) and records their
  mapped bytes with the samples, 3 runs at the largest input; and one run with `-fmem-report`, whose GC statistics at
  exit give the GC heap's allocated and used bytes. Reported: the GC heap's share of anonymous RSS at the peak, and
  its unused share at exit, which no interposition can see.

## What would change the paper

The short-run result goes into the transfer and blind sections if (b) or (c) changes a named program's error by more
than 5 points; otherwise it is reported as a check. The cost goes into the approaches table either way.

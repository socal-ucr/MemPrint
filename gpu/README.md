# Ground truth for kernel dependences (needs a GPU)

MemPrint's static interpreter predicts the memory dependences between a CUDA program's kernel launches,
copies and memsets without running it on a GPU (`analysis/memprint/static/interp_cuda.py`, `kdeps.py`). This
directory holds the other side: an NVBit tool, `nvbit_rw/rw_trace.so`. It traces which bytes each of those
operations actually reads and writes. `kdeps_score.py` then compares the two. In the footprint studies, the
Pin tool plays this role.

**Status: built and run** with NVBit 1.8, CUDA 13.4 (driver 610) and an A30 (sm_80). On the test programs of
`tests/static/cuda` (`scripts/kdeps_truth_tests.sh`) every program aligns, and must precision and recall are 1
with no units error. `maybe` has must recall 0.6, as intended: its kernel branches on values the interpreter
cannot know.

## Requirements

- An NVIDIA GPU supported by NVBit (see the release notes of your NVBit release), a driver, and a CUDA toolkit (nvcc) that
  your NVBit release supports.
- NVBit, the release tarball from <https://github.com/NVlabs/NVBit/releases>. `NVBIT_PATH` is its `core/`
  directory.
- A Python with numpy and the libclang bindings, the same environment the static interpreter uses (on the
  development machine, `~/memprint-venv`).

## Build

```sh
cd gpu/nvbit_rw
make NVBIT_PATH=/path/to/nvbit_release/core ARCH=sm_80   # ARCH: your GPU, or all
```

`rw_trace.cu` follows NVBit's `tools/mem_trace` example (1.8). As there, a `flush_channel` kernel is built to a
fatbin, embedded with `bin2c` and loaded as a tool module; it pushes the records left in the channel to the
host after each launch and at the end. If the build fails with another NVBit release, check these calls
against the `mem_trace.cu` shipped with it, because they changed between versions:

- `nvbit_set_at_launch`: here `(ctx, f, value)`. It must come before `nvbit_enable_instrumented`, otherwise
  the records carry a wrong event number. Pass a stream only for graph nodes.
- `nvbit_set_tool_pthread` (renamed `nvbit_set_tool_thread` in 1.8.1, which also builds with CMake)
- `nvbit_load_tool_module`, `nvbit_launch_kernel`

`KDEPS_VERBOSE=1` names every driver call the program makes on stderr, and every record received. Use it to
find launches, copies or memsets the tool does not handle.

## Run

```sh
# one program: build, run plainly and under the tool, run the interpreter, score
PYTHON=~/memprint-venv/bin/python scripts/kdeps_truth.sh OUT NAME prog.cu [args...]

# the test programs of tests/static/cuda
PYTHON=~/memprint-venv/bin/python scripts/kdeps_truth_tests.sh OUT
```

The tool can also be used on its own:

```sh
LD_PRELOAD=gpu/nvbit_rw/rw_trace.so KDEPS_OUT=dir KDEPS_UNIT=4 ./prog args
```

It writes three files to `dir`:

| file | rows |
|---|---|
| `truth_allocs.csv` | every `cuMemAlloc` / `cuMemAllocManaged`, in order: index, base, bytes |
| `truth_events.csv` | every launch, copy (`cuMemcpy*`) and memset (`cuMemsetD8/D32*`), in order: index, kind, name, grid, block |
| `truth_access.csv` | event, allocation, `[unit_lo, unit_hi)`, bits: 1 read, 2 write, 3 both (atomics) |

Units are `KDEPS_UNIT` bytes, counted from the allocation's base.

## What is compared

`kdeps_score.py` matches the two sides by position:

- **Events** by their order among device operations. The static side runs with `--no-host`. If the
  sequences differ, the score row says `aligned=0`. That happens when a program issues operations the
  interpreter does not model, or the runtime adds its own.
- **Objects** by the order of their device allocations.

Both sides' per-event bytes go through the same `kdeps.Tracker`, so only the bytes each event touched can
differ. The scores are computed over dependences keyed by (src, dst, RAW/WAR/WAW, allocation):

| score | meaning |
|---|---|
| `must_precision` | share of the static must dependences that the trace also has (should be 1) |
| `recall` | share of the traced dependences that the static side has, must or may (should be 1) |
| `must_recall` | the same, counting only must (below 1 where the interpreter could only say may) |
| `units_error` | how far the number of units through each shared dependence is off |

The same four scores are also computed over the transitive reductions (`needed_*`).

## Known gaps

- **Uninstrumented memory.** `__device__` / `__constant__` variables, `cudaMemcpyToSymbol` targets and pinned
  host memory are not `cuMemAlloc` allocations. The tool counts accesses to them as "outside" and the score
  leaves them out.
- **Compiler optimisation.** The traced binary is optimised (`-O2`), so it may skip loads the source makes
  (a value kept in a register), or make extra ones. Dependences are about bytes, so this rarely changes an
  edge, but it can change `units_error`.
- **Thread interleaving.** A kernel that reads what other threads of the same launch write (label
  propagation with `atomicMin` in place, `cc.cu`) can touch different bytes, and the host loop around it can
  run a different number of rounds, from one run to the next: five traced runs of `cc` took 22 to 24 rounds.
  The scripts run the interpreter under four legal schedules (`--schedules`: all threads in lockstep, and
  one block after another in order, in reverse and at random). Events after the start and end shared by
  every schedule, and dependences not must in every schedule, are may and marked `schedule_dependent`.
  The scorer then matches the trace's events to the interpreter's the same way (common start, common end).
  Four schedules are a sample, not every interleaving: a traced run can have a dependence none of them has
  (`cc`: recall 0.98 to 0.995 over five runs, must precision 1). A worklist filled through `atomicAdd`
  positions (`wl_bfs.cu`) does not depend on the schedule at this level: the set of positions and the set
  of vertices are the same in every run, only which vertex goes where is not.
- **Copy direction.** Copies through `cuMemcpy` / `cuMemcpyAsync` (unified addressing) are classified by
  whether each pointer falls in a known allocation.

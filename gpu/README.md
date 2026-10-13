# Ground truth for kernel dependences (needs a GPU)

MemPrint's static interpreter predicts the memory dependences between a CUDA program's kernel launches,
copies and memsets without running it on a GPU (`analysis/memprint/static/interp_cuda.py`, `kdeps.py`). This
directory holds the other side: an NVBit tool, `nvbit_rw/rw_trace.so`. It traces which bytes each of those
operations actually reads and writes. `kdeps_score.py` then compares the two. In the footprint studies, the
Pin tool plays this role.

**Status: none of the code here has been compiled or run.** It was written on a machine with no GPU and no
nvcc. The interpreter side and the scoring pipeline were tested on that machine without a GPU
(`tests/static/test_cuda.py`). On a GPU machine, build the tool first; expect small fixes for your NVBit
version (see Build).

## Requirements

- An NVIDIA GPU supported by NVBit (sm_35 to sm_90 for NVBit 1.7), a driver, and a CUDA toolkit (nvcc) that
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

`rw_trace.cu` follows NVBit's `tools/mem_trace` example (1.7.x). If the build fails, check these calls
against the `mem_trace.cu` shipped with your release, because they changed between versions:

- `nvbit_tool_init`
- `nvbit_set_at_launch` (here `(ctx, f, &value, sizeof value)`)
- `nvbit_add_call_arg_launch_val64`

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
- **Thread interleaving.** The interpreter runs threads in lockstep and atomics in thread order. A kernel
  whose touched addresses depend on the interleaving can differ from any one traced run. An example is a
  worklist filled through `atomicAdd` positions: the set of positions is the same, the contents are not.
- **Copy direction.** Copies through `cuMemcpy` / `cuMemcpyAsync` (unified addressing) are classified by
  whether each pointer falls in a known allocation.

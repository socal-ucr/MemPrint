/* rw_trace: the bytes each kernel launch, device copy and memset of a CUDA program reads and writes, per
 * device allocation. It is the ground truth MemPrint's static dependences (analysis/memprint/static/kdeps.py)
 * are scored against (kdeps_score.py), as the Pin tool is for the footprints.
 *
 *   LD_PRELOAD=rw_trace.so KDEPS_OUT=dir [KDEPS_UNIT=4] ./program args
 *
 * Writes dir/truth_allocs.csv (index, base address, bytes: cuMemAlloc / cuMemAllocManaged in order),
 * dir/truth_events.csv (index, kind, name, grid, block) and dir/truth_access.csv (event, alloc, unit_lo,
 * unit_hi, bits: units [lo, hi) of the allocation the event read (1) and / or wrote (2); atomics both).
 * Accesses outside every allocation (shared and local memory through generic pointers, __device__
 * variables, the runtime's own memory) are counted, not listed.
 *
 * KDEPS_VERBOSE=1 names every driver call on stderr (to find calls the tool does not handle).
 *
 * KDEPS_WORK=1 instead counts the warp-level instructions each block of each launch executes (as NVBit's
 * instr_count_bb, into a per-block slot of a device buffer) and writes dir/truth_block_work.csv (event, block,
 * warp_instrs): how the work of a kernel divides among its blocks. (Timing blocks with clock64 under NVBit
 * measures mostly the instrumentation: an empty block takes 6,000 to 37,000 cycles.)
 *
 * KDEPS_BLOCKS=1 also writes dir/truth_block_access.csv (event, alloc, block, unit_lo, unit_hi, bits): per
 * kernel launch, the units each block (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) read
 * and wrote.
 *
 * Built and run with NVBit 1.8 and CUDA 13 (tools/mem_trace has the same structure). See gpu/README.md. */
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/channel.hpp"

#include "common.h"

#define CHANNEL_SIZE (1l << 20)

enum class RecvState { WORKING, STOP, FINISHED };

struct CTXstate {
    ChannelDev *channel_dev;
    ChannelHost channel_host;
    volatile RecvState recv_state;
    CUmodule tool_module;
    CUfunction flush_channel_func;  /* pushes the device side's partial buffer to the host */
};

#include "tool_func/flush_channel.c"

#define WORK_CAPACITY (1ull << 22)  /* blocks per launch whose instructions are counted */
static unsigned long long *work_slots = nullptr;  /* device: one counter per block (count_block) */
static std::vector<unsigned long long> work_host;

struct Event {
    std::string kind, name, grid, block;
    std::unordered_map<uint64_t, uint8_t> units;  /* global unit number (address / unit) -> bits */
    std::unordered_map<uint64_t, std::unordered_map<uint32_t, uint8_t>> blocks;  /* unit -> block -> bits */
    std::map<uint32_t, unsigned long long> work;  /* KDEPS_WORK: block -> warp-level instructions */
};

static pthread_mutex_t mutex;          /* tool state below, shared with the receiver threads */
static pthread_mutex_t cuda_event_mutex;
static std::unordered_map<CUcontext, CTXstate *> ctx_state_map;
static std::unordered_set<CUfunction> already_instrumented;
static std::map<uint64_t, std::pair<uint64_t, int>> allocs;  /* base -> (bytes, index) */
static std::vector<std::pair<uint64_t, uint64_t>> alloc_list;
static std::vector<Event> events;
static uint64_t launch_event = 0;      /* the event number passed to the next launch's records */
static uint64_t outside = 0;           /* lane accesses outside every allocation */
static bool skip_callback_flag = false;
static uint64_t unit = 4;
static std::string out_dir = ".";
static bool verbose = false;           /* KDEPS_VERBOSE: name every driver call on stderr */
static bool per_block = false;         /* KDEPS_BLOCKS: also keep which block made each access */
static bool timing = false;            /* KDEPS_WORK: only count each block's instructions */

static Event &new_event(const std::string &kind, const std::string &name) {
    events.push_back(Event());
    events.back().kind = kind;
    events.back().name = name;
    return events.back();
}

/* bytes [addr, addr + n) of an event: mutex held */
static void mark(Event &ev, uint64_t addr, uint64_t n, uint8_t bits) {
    if (n == 0) return;
    for (uint64_t u = addr / unit; u <= (addr + n - 1) / unit; u++) ev.units[u] |= bits;
}

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    if (getenv("KDEPS_OUT")) out_dir = getenv("KDEPS_OUT");
    if (getenv("KDEPS_UNIT")) unit = strtoull(getenv("KDEPS_UNIT"), 0, 10);
    verbose = getenv("KDEPS_VERBOSE") && atoi(getenv("KDEPS_VERBOSE"));
    per_block = getenv("KDEPS_BLOCKS") && atoi(getenv("KDEPS_BLOCKS"));
    timing = getenv("KDEPS_WORK") && atoi(getenv("KDEPS_WORK"));
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);
    pthread_mutex_init(&cuda_event_mutex, &attr);
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    CTXstate *ctx_state = ctx_state_map[ctx];
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);
    if (timing) {  /* a count at the head of every basic block, of the kernel and the functions it calls */
        for (auto f : related) {
            if (!already_instrumented.insert(f).second) continue;
            const CFG_t &cfg = nvbit_get_CFG(ctx, f);
            std::vector<std::pair<Instr *, int>> heads;
            if (cfg.is_degenerate) {
                for (auto instr : nvbit_get_instrs(ctx, f)) heads.push_back(std::make_pair(instr, 1));
            } else {
                for (auto bb : cfg.bbs) heads.push_back(std::make_pair(bb->instrs[0], (int)bb->instrs.size()));
            }
            for (auto &h : heads) {
                nvbit_insert_call(h.first, "count_block", IPOINT_BEFORE);
                nvbit_add_call_arg_const_val32(h.first, h.second);
                nvbit_add_call_arg_const_val64(h.first, (uint64_t)work_slots);
                nvbit_add_call_arg_const_val64(h.first, WORK_CAPACITY);
            }
        }
        return;
    }
    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;
        for (auto instr : nvbit_get_instrs(ctx, f)) {
            auto space = instr->getMemorySpace();
            if (space != InstrType::MemorySpace::GLOBAL && space != InstrType::MemorySpace::GENERIC) continue;
            std::string op = instr->getOpcodeShort();
            int flags = 0;
            if (op.rfind("ATOM", 0) == 0 || op.rfind("RED", 0) == 0) flags = RW_READ | RW_WRITE;
            else if (instr->isLoad()) flags = RW_READ;
            else if (instr->isStore()) flags = RW_WRITE;
            if (!flags) continue;
            int mref_idx = 0;
            for (int i = 0; i < instr->getNumOperands(); i++) {
                const InstrType::operand_t *operand = instr->getOperand(i);
                if (operand->type != InstrType::OperandType::MREF) continue;
                nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr, flags);
                nvbit_add_call_arg_const_val32(instr, instr->getSize());
                nvbit_add_call_arg_mref_addr64(instr, mref_idx);
                nvbit_add_call_arg_launch_val64(instr, 0);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
                mref_idx++;
            }
        }
    }
}

static void *recv_thread_fun(void *args) {
    CUcontext ctx = (CUcontext)args;
    pthread_mutex_lock(&mutex);
    CTXstate *ctx_state = ctx_state_map[ctx];
    ChannelHost *ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);
    char *recv_buffer = (char *)malloc(CHANNEL_SIZE);
    for (;;) {
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        if (num_recv_bytes == 0) {
            if (ctx_state->recv_state != RecvState::WORKING) break;  /* stopped and drained */
            continue;
        }
        pthread_mutex_lock(&mutex);
        for (uint32_t off = 0; off + sizeof(mem_access_t) <= num_recv_bytes; off += sizeof(mem_access_t)) {
            mem_access_t *ma = (mem_access_t *)&recv_buffer[off];
            if (verbose)
                fprintf(stderr, "rw_trace: record event %llu mask %08x flags %u size %u addr0 0x%llx\n",
                        (unsigned long long)ma->event, ma->active_mask, ma->flags, ma->size,
                        (unsigned long long)ma->addrs[__builtin_ctz(ma->active_mask | 0x80000000u)]);
            if (ma->event >= events.size()) continue;
            Event &ev = events[ma->event];
            for (int lane = 0; lane < 32; lane++) {
                if (!(ma->active_mask & (1u << lane))) continue;
                uint64_t a = ma->addrs[lane];
                auto it = allocs.upper_bound(a);
                if (it == allocs.begin()) { outside++; continue; }
                --it;
                if (a >= it->first + it->second.first) { outside++; continue; }
                mark(ev, a, ma->size, (uint8_t)ma->flags);
                if (per_block && ma->size)
                    for (uint64_t u = a / unit; u <= (a + ma->size - 1) / unit; u++)
                        ev.blocks[u][ma->cta] |= (uint8_t)ma->flags;
            }
        }
        pthread_mutex_unlock(&mutex);
    }
    free(recv_buffer);
    ctx_state->recv_state = RecvState::FINISHED;
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    CTXstate *ctx_state = new CTXstate;
    ctx_state->recv_state = RecvState::FINISHED;  /* no receiver until nvbit_tool_init (the first launch) */
    ctx_state->channel_dev = nullptr;
    ctx_state_map[ctx] = ctx_state;
    nvbit_load_tool_module(ctx, (const void *)flush_channel_bin, &ctx_state->tool_module);
    nvbit_find_function_by_name(ctx, ctx_state->tool_module, "flush_channel", &ctx_state->flush_channel_func);
    pthread_mutex_unlock(&mutex);
}

/* Wait for the device, then push what the instrumented code left in the channel to the receiver. */
static void flush_channel(CUcontext ctx) {
    CTXstate *ctx_state = ctx_state_map[ctx];
    cudaDeviceSynchronize();
    void *args[] = {&ctx_state->channel_dev};
    nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
    cudaDeviceSynchronize();
}

void nvbit_tool_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    CTXstate *ctx_state = ctx_state_map[ctx];
    ctx_state->recv_state = RecvState::WORKING;
    cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
    if (timing && !work_slots) {
        cudaMalloc(&work_slots, WORK_CAPACITY * sizeof(unsigned long long));
        cudaMemset(work_slots, 0, WORK_CAPACITY * sizeof(unsigned long long));
    }
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE, ctx_state->channel_dev,
                                 recv_thread_fun, ctx);
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
    pthread_mutex_unlock(&mutex);
}

static void alloc_done(uint64_t base, uint64_t bytes) {
    pthread_mutex_lock(&mutex);
    int index = (int)alloc_list.size();
    allocs[base] = std::make_pair(bytes, index);
    alloc_list.push_back(std::make_pair(base, bytes));
    pthread_mutex_unlock(&mutex);
}

static void copy_event(const char *kind, uint64_t dst, uint64_t src, uint64_t n, bool dst_dev, bool src_dev) {
    pthread_mutex_lock(&mutex);
    Event &ev = new_event("copy", std::string("cudaMemcpy ") + kind);
    if (src_dev) mark(ev, src, n, RW_READ);
    if (dst_dev) mark(ev, dst, n, RW_WRITE);
    pthread_mutex_unlock(&mutex);
}

static bool on_device(uint64_t a) {
    auto it = allocs.upper_bound(a);
    if (it == allocs.begin()) return false;
    --it;
    return a < it->first + it->second.first;
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char *name, void *params,
                         CUresult *pStatus) {
    pthread_mutex_lock(&cuda_event_mutex);
    if (skip_callback_flag) {
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }
    skip_callback_flag = true;
    if (verbose && !is_exit) fprintf(stderr, "rw_trace: %s\n", name);
    switch (cbid) {
    case API_CUDA_cuLaunchKernel:
    case API_CUDA_cuLaunchKernel_ptsz:
    case API_CUDA_cuLaunchCooperativeKernel:
    case API_CUDA_cuLaunchCooperativeKernel_ptsz:
    case API_CUDA_cuLaunchKernelEx:
    case API_CUDA_cuLaunchKernelEx_ptsz: {
        /* the runtime launches through cuLaunchKernelEx since CUDA 12; its grid and block are in config */
        CUfunction f;
        unsigned g[3], b[3];
        if (cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
            cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
            const CUlaunchConfig *c = p->config;
            f = p->f;
            g[0] = c->gridDimX, g[1] = c->gridDimY, g[2] = c->gridDimZ;
            b[0] = c->blockDimX, b[1] = c->blockDimY, b[2] = c->blockDimZ;
        } else {  /* the cooperative launch has the same parameters */
            cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
            f = p->f;
            g[0] = p->gridDimX, g[1] = p->gridDimY, g[2] = p->gridDimZ;
            b[0] = p->blockDimX, b[1] = p->blockDimY, b[2] = p->blockDimZ;
        }
        if (!is_exit) {
            instrument_function_if_needed(ctx, f);
            pthread_mutex_lock(&mutex);
            Event &ev = new_event("kernel", nvbit_get_func_name(ctx, f));
            char buf[96];
            snprintf(buf, sizeof buf, "%ux%ux%u", g[0], g[1], g[2]);
            ev.grid = buf;
            snprintf(buf, sizeof buf, "%ux%ux%u", b[0], b[1], b[2]);
            ev.block = buf;
            launch_event = events.size() - 1;
            pthread_mutex_unlock(&mutex);
            nvbit_set_at_launch(ctx, f, launch_event);  /* a stream only for graph nodes */
            nvbit_enable_instrumented(ctx, f, true);  /* after nvbit_set_at_launch, as in mem_trace */
        } else {
            flush_channel(ctx);  /* its records reach the receiver, which files them by event */
            if (timing && work_slots) {  /* the launch's counts, then clear the slots for the next */
                uint64_t nb = std::min((uint64_t)g[0] * g[1] * g[2], (uint64_t)WORK_CAPACITY);
                work_host.resize(nb);
                cudaMemcpy(work_host.data(), work_slots, nb * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                cudaMemset(work_slots, 0, nb * sizeof(unsigned long long));
                pthread_mutex_lock(&mutex);
                for (uint64_t b = 0; b < nb; b++)
                    if (work_host[b]) events[launch_event].work[(uint32_t)b] = work_host[b];
                pthread_mutex_unlock(&mutex);
            }
        }
        break;
    }
    case API_CUDA_cuMemAlloc_v2:
        if (is_exit) {
            cuMemAlloc_v2_params *p = (cuMemAlloc_v2_params *)params;
            alloc_done((uint64_t)*p->dptr, p->bytesize);
        }
        break;
    case API_CUDA_cuMemAllocManaged:
        if (is_exit) {
            cuMemAllocManaged_params *p = (cuMemAllocManaged_params *)params;
            alloc_done((uint64_t)*p->dptr, p->bytesize);
        }
        break;
    case API_CUDA_cuMemcpyHtoD_v2:
    case API_CUDA_cuMemcpyHtoDAsync_v2:
        if (!is_exit) {
            cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *)params;
            copy_event("H2D", (uint64_t)p->dstDevice, 0, p->ByteCount, true, false);
        }
        break;
    case API_CUDA_cuMemcpyDtoH_v2:
    case API_CUDA_cuMemcpyDtoHAsync_v2:
        if (!is_exit) {
            cuMemcpyDtoH_v2_params *p = (cuMemcpyDtoH_v2_params *)params;
            copy_event("D2H", 0, (uint64_t)p->srcDevice, p->ByteCount, false, true);
        }
        break;
    case API_CUDA_cuMemcpyDtoD_v2:
    case API_CUDA_cuMemcpyDtoDAsync_v2:
        if (!is_exit) {
            cuMemcpyDtoD_v2_params *p = (cuMemcpyDtoD_v2_params *)params;
            copy_event("D2D", (uint64_t)p->dstDevice, (uint64_t)p->srcDevice, p->ByteCount, true, true);
        }
        break;
    case API_CUDA_cuMemcpy:
    case API_CUDA_cuMemcpyAsync:
        if (!is_exit) {  /* unified addressing: the direction from where the pointers are */
            cuMemcpy_params *p = (cuMemcpy_params *)params;
            bool d = on_device((uint64_t)p->dst), s = on_device((uint64_t)p->src);
            copy_event(d ? (s ? "D2D" : "H2D") : (s ? "D2H" : "H2H"), (uint64_t)p->dst, (uint64_t)p->src,
                       p->ByteCount, d, s);
        }
        break;
    case API_CUDA_cuMemsetD8_v2:
    case API_CUDA_cuMemsetD8Async:
        if (!is_exit) {
            cuMemsetD8_v2_params *p = (cuMemsetD8_v2_params *)params;
            pthread_mutex_lock(&mutex);
            mark(new_event("set", "cudaMemset"), (uint64_t)p->dstDevice, p->N, RW_WRITE);
            pthread_mutex_unlock(&mutex);
        }
        break;
    case API_CUDA_cuMemsetD32_v2:
    case API_CUDA_cuMemsetD32Async:
        if (!is_exit) {
            cuMemsetD32_v2_params *p = (cuMemsetD32_v2_params *)params;
            pthread_mutex_lock(&mutex);
            mark(new_event("set", "cudaMemset"), (uint64_t)p->dstDevice, 4 * p->N, RW_WRITE);
            pthread_mutex_unlock(&mutex);
        }
        break;
    default:
        break;
    }
    skip_callback_flag = false;
    pthread_mutex_unlock(&cuda_event_mutex);
}

static void write_outputs() {
    std::string path = out_dir + "/truth_allocs.csv";
    FILE *f = fopen(path.c_str(), "w");
    fprintf(f, "index,base,bytes\n");
    for (size_t i = 0; i < alloc_list.size(); i++)
        fprintf(f, "%zu,0x%llx,%llu\n", i, (unsigned long long)alloc_list[i].first,
                (unsigned long long)alloc_list[i].second);
    fclose(f);
    path = out_dir + "/truth_events.csv";
    f = fopen(path.c_str(), "w");
    fprintf(f, "index,kind,name,grid,block\n");
    for (size_t i = 0; i < events.size(); i++)
        fprintf(f, "%zu,%s,\"%s\",%s,%s\n", i, events[i].kind.c_str(), events[i].name.c_str(),
                events[i].grid.c_str(), events[i].block.c_str());
    fclose(f);
    path = out_dir + "/truth_access.csv";
    f = fopen(path.c_str(), "w");
    fprintf(f, "event,alloc,unit_lo,unit_hi,bits\n");
    for (size_t i = 0; i < events.size(); i++) {
        /* per allocation, runs of units with the same bits, in allocation-relative units */
        std::map<std::pair<int, uint64_t>, uint8_t> sorted;
        for (auto &kv : events[i].units) {
            uint64_t a = kv.first * unit;
            auto it = allocs.upper_bound(a);
            if (it == allocs.begin()) continue;
            --it;
            if (a >= it->first + it->second.first) continue;
            sorted[std::make_pair(it->second.second, (a - it->first) / unit)] |= kv.second;
        }
        int alloc = -1;
        uint64_t lo = 0, hi = 0;
        uint8_t bits = 0;
        for (auto &kv : sorted) {
            if (kv.first.first == alloc && kv.first.second == hi && kv.second == bits) {
                hi++;
                continue;
            }
            if (alloc >= 0) fprintf(f, "%zu,%d,%llu,%llu,%u\n", i, alloc, (unsigned long long)lo,
                                    (unsigned long long)hi, bits);
            alloc = kv.first.first;
            lo = kv.first.second;
            hi = lo + 1;
            bits = kv.second;
        }
        if (alloc >= 0) fprintf(f, "%zu,%d,%llu,%llu,%u\n", i, alloc, (unsigned long long)lo,
                                (unsigned long long)hi, bits);
    }
    fclose(f);
    if (timing) {
        path = out_dir + "/truth_block_work.csv";
        f = fopen(path.c_str(), "w");
        fprintf(f, "event,block,warp_instrs\n");
        for (size_t i = 0; i < events.size(); i++)
            for (auto &kv : events[i].work) fprintf(f, "%zu,%u,%llu\n", i, kv.first, kv.second);
        fclose(f);
    }
    if (per_block) {
        path = out_dir + "/truth_block_access.csv";
        f = fopen(path.c_str(), "w");
        fprintf(f, "event,alloc,block,unit_lo,unit_hi,bits\n");
        for (size_t i = 0; i < events.size(); i++) {
            /* per allocation and block, runs of units with the same bits */
            std::map<std::tuple<int, uint32_t, uint64_t>, uint8_t> sorted;
            for (auto &kv : events[i].blocks) {
                uint64_t a = kv.first * unit;
                auto it = allocs.upper_bound(a);
                if (it == allocs.begin()) continue;
                --it;
                if (a >= it->first + it->second.first) continue;
                for (auto &bb : kv.second)
                    sorted[std::make_tuple(it->second.second, bb.first, (a - it->first) / unit)] |= bb.second;
            }
            int alloc = -1;
            uint32_t block = 0;
            uint64_t lo = 0, hi = 0;
            uint8_t bits = 0;
            for (auto &kv : sorted) {
                int a = std::get<0>(kv.first);
                uint32_t b = std::get<1>(kv.first);
                uint64_t u = std::get<2>(kv.first);
                if (a == alloc && b == block && u == hi && kv.second == bits) {
                    hi++;
                    continue;
                }
                if (alloc >= 0) fprintf(f, "%zu,%d,%u,%llu,%llu,%u\n", i, alloc, block, (unsigned long long)lo,
                                        (unsigned long long)hi, bits);
                alloc = a, block = b, lo = u, hi = u + 1, bits = kv.second;
            }
            if (alloc >= 0) fprintf(f, "%zu,%d,%u,%llu,%llu,%u\n", i, alloc, block, (unsigned long long)lo,
                                    (unsigned long long)hi, bits);
        }
        fclose(f);
    }
    fprintf(stderr, "rw_trace: %zu events, %zu allocations, %llu lane accesses outside them -> %s\n",
            events.size(), alloc_list.size(), (unsigned long long)outside, out_dir.c_str());
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    skip_callback_flag = true;
    CTXstate *ctx_state = ctx_state_map[ctx];
    pthread_mutex_unlock(&mutex);
    if (ctx_state->channel_dev) {  /* a program that exits before its first launch never started a receiver */
        flush_channel(ctx);
        ctx_state->recv_state = RecvState::STOP;
        while (ctx_state->recv_state != RecvState::FINISHED) usleep(1000);
        ctx_state->channel_host.destroy(false);
        cudaFree(ctx_state->channel_dev);
    }
    pthread_mutex_lock(&mutex);
    write_outputs();
    skip_callback_flag = false;
    pthread_mutex_unlock(&mutex);
}

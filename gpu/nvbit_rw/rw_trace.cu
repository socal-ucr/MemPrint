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
 * Written against NVBit 1.7 (tools/mem_trace has the same structure); not compiled on the machine it was
 * written on (no GPU there). See gpu/README.md. */
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <map>
#include <string>
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
};

struct Event {
    std::string kind, name, grid, block;
    std::unordered_map<uint64_t, uint8_t> units;  /* global unit number (address / unit) -> bits */
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
    ctx_state_map[ctx] = new CTXstate;
    pthread_mutex_unlock(&mutex);
}

void nvbit_tool_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    CTXstate *ctx_state = ctx_state_map[ctx];
    ctx_state->recv_state = RecvState::WORKING;
    cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
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
    switch (cbid) {
    case API_CUDA_cuLaunchKernel:
    case API_CUDA_cuLaunchKernel_ptsz: {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
        if (!is_exit) {
            instrument_function_if_needed(ctx, p->f);
            nvbit_enable_instrumented(ctx, p->f, true);
            pthread_mutex_lock(&mutex);
            Event &ev = new_event("kernel", nvbit_get_func_name(ctx, p->f));
            char buf[96];
            snprintf(buf, sizeof buf, "%ux%ux%u", p->gridDimX, p->gridDimY, p->gridDimZ);
            ev.grid = buf;
            snprintf(buf, sizeof buf, "%ux%ux%u", p->blockDimX, p->blockDimY, p->blockDimZ);
            ev.block = buf;
            launch_event = events.size() - 1;
            pthread_mutex_unlock(&mutex);
            nvbit_set_at_launch(ctx, p->f, &launch_event, sizeof(uint64_t));
        } else {
            cudaDeviceSynchronize();  /* its records are in the channel; the receiver files them by event */
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
    fprintf(stderr, "rw_trace: %zu events, %zu allocations, %llu lane accesses outside them -> %s\n",
            events.size(), alloc_list.size(), (unsigned long long)outside, out_dir.c_str());
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    skip_callback_flag = true;
    CTXstate *ctx_state = ctx_state_map[ctx];
    pthread_mutex_unlock(&mutex);
    cudaDeviceSynchronize();
    ctx_state->recv_state = RecvState::STOP;
    while (ctx_state->recv_state != RecvState::FINISHED) usleep(1000);
    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    pthread_mutex_lock(&mutex);
    write_outputs();
    skip_callback_flag = false;
    pthread_mutex_unlock(&mutex);
}

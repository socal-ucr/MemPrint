// Runs an unmodified CUDA program (built with -cudart shared) on Unified Memory, with a prefetch plan.
//
//   LD_PRELOAD=uvm_shim.so UVM_DEVICE_BYTES=N UVM_PLAN=plan.txt UVM_OUT=out.txt ./program ...
//
// cudaMalloc becomes cudaMallocManaged; cudaMemcpy between host and device becomes a copy by the host (managed
// pages move as the host touches them). On the first allocation, an ordinary cudaMalloc takes all of the device's
// free memory but UVM_DEVICE_BYTES, so the managed pages share that much. Before the K-th kernel launch, the lines
// "K alloc offset bytes" of UVM_PLAN are prefetched to the device (alloc: the index of the allocation in the order
// the program made them). Each launch is followed by cudaDeviceSynchronize, and UVM_OUT receives the time spent in
// prefetches, kernels and copies (ns), the launches, the allocations and the bytes prefetched.
//
//   g++ -O2 -shared -fPIC -I/usr/local/cuda/include -o uvm_shim.so uvm_shim.cpp -L/usr/local/cuda/lib64 -lcudart -ldl

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace {

struct Alloc { char *base; size_t bytes; };
struct Range { int alloc; size_t offset, bytes; };

std::vector<Alloc> allocs;
std::map<long, std::vector<Range>> plan;                  // launch -> ranges to prefetch before it
long launches = 0;
double gpu_ns = 0, prefetch_ns = 0, copy_ns = 0;
size_t prefetched = 0;
bool ready = false;
void *balloon = nullptr;

double now() {
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename F> F real(const char *name) {
    void *f = dlsym(RTLD_NEXT, name);
    if (!f) { fprintf(stderr, "uvm_shim: no %s\n", name); exit(1); }
    return reinterpret_cast<F>(f);
}

void report() {
    const char *out = getenv("UVM_OUT");
    if (!out) return;
    cudaDeviceSynchronize();
    FILE *f = fopen(out, "w");
    if (!f) return;
    fprintf(f, "gpu_ns %.0f\nprefetch_ns %.0f\ncopy_ns %.0f\nlaunches %ld\nallocs %zu\nprefetched %zu\n",
            gpu_ns, prefetch_ns, copy_ns, launches, allocs.size(), prefetched);
    fclose(f);
}

void init() {
    if (ready) return;
    ready = true;
    if (const char *p = getenv("UVM_PLAN")) {
        FILE *f = fopen(p, "r");
        long k; int a; size_t off, n;
        while (f && fscanf(f, "%ld %d %zu %zu", &k, &a, &off, &n) == 4)
            plan[k].push_back({a, off, n});
        if (f) fclose(f);
    }
    if (const char *keep = getenv("UVM_DEVICE_BYTES")) {   // leave this much of the device to managed pages
        size_t free_b = 0, total = 0;
        cudaMemGetInfo(&free_b, &total);
        size_t want = strtoull(keep, nullptr, 10);
        if (free_b > want) {
            auto malloc_ = real<cudaError_t (*)(void **, size_t)>("cudaMalloc");
            if (malloc_(&balloon, free_b - want) != cudaSuccess)
                fprintf(stderr, "uvm_shim: could not reserve %zu bytes\n", free_b - want);
        }
    }
    atexit(report);
}

}  // namespace

extern "C" {

cudaError_t cudaMalloc(void **p, size_t bytes) {
    init();
    cudaError_t e = cudaMallocManaged(p, bytes, cudaMemAttachGlobal);
    if (e == cudaSuccess) allocs.push_back({static_cast<char *>(*p), bytes});
    return e;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, cudaMemcpyKind kind) {
    init();
    double t = now();
    cudaDeviceSynchronize();
    memcpy(dst, src, n);                                   // the host touches the managed pages
    copy_ns += now() - t;
    return cudaSuccess;
}

}  // extern "C"

namespace {

// Prefetch the plan's ranges for this launch, run it, wait for it, and count the time.
template <typename L> cudaError_t around(cudaStream_t s, L launch) {
    init();
    int dev = 0;
    cudaGetDevice(&dev);
    double t0 = now();
    auto it = plan.find(launches);
    if (it != plan.end()) {
        cudaMemLocation gpu = {cudaMemLocationTypeDevice, dev};
        for (const Range &r : it->second) {
            if (r.alloc < 0 || r.alloc >= (int)allocs.size()) continue;
            const Alloc &a = allocs[r.alloc];
            size_t n = r.offset >= a.bytes ? 0 : std::min(r.bytes, a.bytes - r.offset);
            if (n && cudaMemPrefetchAsync(a.base + r.offset, n, gpu, 0, s) == cudaSuccess) prefetched += n;
        }
        cudaStreamSynchronize(s);
    }
    double t1 = now();
    cudaError_t e = launch();
    cudaDeviceSynchronize();
    double t2 = now();
    prefetch_ns += t1 - t0;
    gpu_ns += t2 - t0;
    launches++;
    return e;
}

}  // namespace

extern "C" {

cudaError_t cudaLaunchKernel(const void *func, dim3 grid, dim3 block, void **args, size_t smem, cudaStream_t s) {
    static auto f = real<cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t)>("cudaLaunchKernel");
    return around(s, [&] { return f(func, grid, block, args, smem, s); });
}

// What <<<>>> calls since CUDA 13 (crt/device_functions.h).
cudaError_t __cudaLaunchKernel(cudaKernel_t k, dim3 grid, dim3 block, void **args, size_t smem, cudaStream_t s) {
    static auto f = real<cudaError_t (*)(cudaKernel_t, dim3, dim3, void **, size_t, cudaStream_t)>(
        "__cudaLaunchKernel");
    return around(s, [&] { return f(k, grid, block, args, smem, s); });
}

cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t k, dim3 grid, dim3 block, void **args, size_t smem,
                                    cudaStream_t s) {
    static auto f = real<cudaError_t (*)(cudaKernel_t, dim3, dim3, void **, size_t, cudaStream_t)>(
        "__cudaLaunchKernel_ptsz");
    return around(s, [&] { return f(k, grid, block, args, smem, s); });
}

}  // extern "C"

/* A stand-in CUDA runtime for the hybrid interpreter (analysis/memprint/static/hybrid.py).
 *
 * A CUDA program is compiled by nvcc as a shared library without the real runtime (-cudart none) and linked
 * against this library instead. Its host code then runs natively on the CPU: device memory is host memory
 * (cudaMalloc is malloc), copies and fills are memcpy and memset, and a kernel launch calls back into Python,
 * where MemPrint's interpreter runs that one kernel on the real memory contents. Every allocation, copy, fill
 * and launch is reported to Python through callbacks set by kdeps_set_callbacks().
 *
 * The CUDA headers are included only for the types and for the names of the API functions (some are macros
 * for _v2 or _ptds variants); nothing of the real runtime is linked. */
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <string>

typedef void (*launch_cb_t)(const char *device_name, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                            unsigned by, unsigned bz, size_t shmem, void **args);
typedef void (*alloc_cb_t)(void *ptr, size_t bytes, int kind);   /* kind: 0 device, 1 managed */
typedef void (*free_cb_t)(void *ptr);
typedef void (*copy_cb_t)(void *dst, const void *src, size_t bytes, int kind);   /* 0 H2H 1 H2D 2 D2H 3 D2D */
typedef void (*set_cb_t)(void *ptr, int value, size_t bytes);
typedef void (*finish_cb_t)(void);
typedef void (*var_cb_t)(void *host, size_t bytes, const char *device_name, int constant);

static launch_cb_t launch_cb;
static alloc_cb_t alloc_cb;
static free_cb_t free_cb;
static copy_cb_t copy_cb;
static set_cb_t set_cb;
static finish_cb_t finish_cb;
static var_cb_t var_cb;

static std::map<const void *, std::string> kernels;      /* host stub -> device (mangled) name */
static std::map<const char *, size_t> allocs;            /* base -> bytes, device and managed */
static std::map<const void *, size_t> symbols;            /* __device__ / __constant__ variable -> bytes */
static dim3 cfg_grid, cfg_block;
static size_t cfg_shmem;

static void at_exit(void) {
    if (finish_cb) finish_cb();
}

extern "C" void kdeps_set_callbacks(launch_cb_t l, alloc_cb_t a, free_cb_t f, copy_cb_t c, set_cb_t s,
                                    finish_cb_t fin) {
    launch_cb = l, alloc_cb = a, free_cb = f, copy_cb = c, set_cb = s, finish_cb = fin;
    atexit(at_exit);   /* a program that calls exit() still has its results written */
}

/* The call site of the last copy or fill, as an offset into the module that called it (for addr2line), so the
   interpreter can tell whether the host issued it under data control. Calls from this library itself (the Async
   variants) keep the site of the program's call. */
static uintptr_t last_site;
static char last_module[1024];

static void note_site(void *ra) {
    Dl_info info;
    if (!dladdr(ra, &info) || !info.dli_fname || strstr(info.dli_fname, "libkdeps_cudart")) return;
    last_site = (uintptr_t)ra - (uintptr_t)info.dli_fbase - 1;   /* inside the call instruction */
    snprintf(last_module, sizeof last_module, "%s", info.dli_fname);
}

extern "C" uintptr_t kdeps_last_site(char *module, size_t n) {
    snprintf(module, n, "%s", last_module);
    return last_site;
}

/* Set before the program is loaded: its variables register while it loads. */
extern "C" void kdeps_set_var_callback(var_cb_t v) { var_cb = v; }

static bool on_device(const void *p) {
    const char *q = (const char *)p;
    auto it = allocs.upper_bound(q);
    if (it == allocs.begin()) return false;
    --it;
    return q < it->first + it->second;
}

static int classify(const void *dst, const void *src, cudaMemcpyKind kind) {
    switch (kind) {
    case cudaMemcpyHostToHost: return 0;
    case cudaMemcpyHostToDevice: return 1;
    case cudaMemcpyDeviceToHost: return 2;
    case cudaMemcpyDeviceToDevice: return 3;
    default: return (on_device(dst) ? 1 : 0) + (on_device(src) ? 2 : 0) == 3 ? 3
                    : on_device(dst) ? 1 : on_device(src) ? 2 : 0;
    }
}

/* ------------------------------------------------------------------ registration and launch */

extern "C" {

void **__cudaRegisterFatBinary(void *fatCubin) {
    static void *handle;
    return &handle;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    kernels[(const void *)hostFun] = deviceName;
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName,
                       int ext, size_t size, int constant, int global) {
    /* __device__ / __constant__ variables live in their host shadow, which the interpreter then uses */
    symbols[hostVar] = size;
    if (var_cb) var_cb(hostVar, size, deviceName, constant);
}

void __cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress, char *deviceAddress,
                              const char *deviceName, int ext, size_t size, int constant, int global) {}

char __cudaInitModule(void **fatCubinHandle) { return 1; }

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream) {
    cfg_grid = gridDim, cfg_block = blockDim, cfg_shmem = sharedMem;
    return 0;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream) {
    *gridDim = cfg_grid, *blockDim = cfg_block, *sharedMem = cfg_shmem;
    if (stream) *(void **)stream = 0;
    return cudaSuccess;
}

}  /* extern "C" */

cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                                       size_t sharedMem, cudaStream_t stream) {
    auto it = kernels.find(func);
    if (it == kernels.end()) {
        fprintf(stderr, "kdeps_cudart: launch of an unregistered kernel %p\n", func);
        return cudaErrorInvalidDeviceFunction;
    }
    if (launch_cb)
        launch_cb(it->second.c_str(), gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
                  sharedMem, args);
    return cudaSuccess;
}

/* CUDA 13's generated launch code: a handle per kernel (here the host stub itself), then a launch by handle */
extern "C" cudaError_t CUDARTAPI __cudaGetKernel(cudaKernel_t *kernel, const void *func) {
    *kernel = (cudaKernel_t)func;
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim, void **args,
                                                    size_t sharedMem, cudaStream_t stream) {
    return cudaLaunchKernel((const void *)kernel, gridDim, blockDim, args, sharedMem, stream);
}

extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim,
                                                         void **args, size_t sharedMem, cudaStream_t stream) {
    return cudaLaunchKernel((const void *)kernel, gridDim, blockDim, args, sharedMem, stream);
}

/* ------------------------------------------------------------------ memory */

static cudaError_t device_alloc(void **ptr, size_t size, int kind) {
    void *p = calloc(size ? size : 1, 1);   /* zeroed, so that a run does not depend on malloc's garbage */
    if (!p) return cudaErrorMemoryAllocation;
    allocs[(const char *)p] = size;
    *ptr = p;
    if (alloc_cb) alloc_cb(p, size, kind);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) { return device_alloc(devPtr, size, 0); }

cudaError_t CUDARTAPI cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    return device_alloc(devPtr, size, 1);
}

cudaError_t CUDARTAPI cudaFree(void *devPtr) {
    if (!devPtr) return cudaSuccess;
    if (free_cb) free_cb(devPtr);
    allocs.erase((const char *)devPtr);
    free(devPtr);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
    *ptr = malloc(size ? size : 1);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags) { return cudaMallocHost(pHost, size); }

cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
    free(ptr);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    note_site(__builtin_return_address(0));
    int k = classify(dst, src, kind);
    memmove(dst, src, count);
    if (copy_cb && k) copy_cb(dst, src, count, k);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
                                      cudaStream_t stream) {
    note_site(__builtin_return_address(0));
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count) {
    note_site(__builtin_return_address(0));
    memset(devPtr, value, count);
    if (set_cb) set_cb(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    note_site(__builtin_return_address(0));
    return cudaMemset(devPtr, value, count);
}

cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset,
                                         enum cudaMemcpyKind kind) {
    note_site(__builtin_return_address(0));
    memcpy((char *)symbol + offset, src, count);
    if (copy_cb) copy_cb((char *)symbol + offset, src, count, kind == cudaMemcpyDeviceToDevice ? 3 : 1);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset,
                                           enum cudaMemcpyKind kind) {
    note_site(__builtin_return_address(0));
    memcpy(dst, (const char *)symbol + offset, count);
    if (copy_cb) copy_cb(dst, (const char *)symbol + offset, count, kind == cudaMemcpyDeviceToDevice ? 3 : 2);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset,
                                              enum cudaMemcpyKind kind, cudaStream_t stream) {
    note_site(__builtin_return_address(0));
    return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset,
                                                enum cudaMemcpyKind kind, cudaStream_t stream) {
    note_site(__builtin_return_address(0));
    return cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
}

cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol) {
    *devPtr = (void *)symbol;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol) {
    auto it = symbols.find(symbol);
    *size = it == symbols.end() ? 0 : it->second;
    return it == symbols.end() ? cudaErrorInvalidSymbol : cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free_, size_t *total) {
    *free_ = (size_t)20 << 30, *total = (size_t)24 << 30;
    return cudaSuccess;
}

/* ------------------------------------------------------------------ device, errors, streams, events */

cudaError_t CUDARTAPI cudaDeviceSynchronize(void) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaDeviceReset(void) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaSetDevice(int device) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaGetDevice(int *device) { *device = 0; return cudaSuccess; }
cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) { *count = 1; return cudaSuccess; }
cudaError_t CUDARTAPI cudaGetLastError(void) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaPeekAtLastError(void) { return cudaSuccess; }
const char *CUDARTAPI cudaGetErrorString(cudaError_t error) { return error ? "error (kdeps_cudart)" : "no error"; }
const char *CUDARTAPI cudaGetErrorName(cudaError_t error) { return error ? "cudaError" : "cudaSuccess"; }
cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaFree(void *devPtr);

cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    memset(prop, 0, sizeof *prop);   /* an A30 */
    strcpy(prop->name, "kdeps_cudart (A30)");
    prop->totalGlobalMem = (size_t)24 << 30;
    prop->sharedMemPerBlock = 48 << 10;
    prop->regsPerBlock = 65536;
    prop->warpSize = 32;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024, prop->maxThreadsDim[1] = 1024, prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647, prop->maxGridSize[1] = 65535, prop->maxGridSize[2] = 65535;
    prop->totalConstMem = 65536;
    prop->major = 8, prop->minor = 0;
    prop->multiProcessorCount = 56;
    prop->maxThreadsPerMultiProcessor = 2048;
    prop->sharedMemPerMultiprocessor = 164 << 10;
    prop->asyncEngineCount = 2;
    prop->unifiedAddressing = 1;
    prop->concurrentKernels = 1;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
    switch (attr) {
    case cudaDevAttrMaxThreadsPerBlock: *value = 1024; break;
    case cudaDevAttrWarpSize: *value = 32; break;
    case cudaDevAttrMultiProcessorCount: *value = 56; break;
    case cudaDevAttrComputeCapabilityMajor: *value = 8; break;
    case cudaDevAttrComputeCapabilityMinor: *value = 0; break;
    default: *value = 0;
    }
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream) { *pStream = 0; return cudaSuccess; }
cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) { *pStream = 0; return cudaSuccess; }
cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) { *event = (cudaEvent_t)malloc(8); return cudaSuccess; }
cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) { return cudaEventCreate(event); }
cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) { *ms = 0; return cudaSuccess; }
cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) { free(event); return cudaSuccess; }
cudaError_t CUDARTAPI cudaThreadSynchronize(void) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaThreadExit(void) { return cudaSuccess; }

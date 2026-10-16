/* The CUDA runtime as MemPrint's static interpreter sees it (interp_cuda.py).
 *
 * Sources are parsed by libclang in CUDA mode, host side only, with no CUDA
 * installation (-nocudainc). This header declares what CUDA's own headers would:
 * the execution-space attributes, dim3 and the built-in index variables, and the
 * runtime and device functions. Functions have no bodies; the interpreter models
 * them by name. It is force-included (-include) before the program's own code.
 */
#ifndef MEMPRINT_CUDA_STUB_H
#define MEMPRINT_CUDA_STUB_H

#include <stddef.h>
/* nvcc includes CUDA's runtime header in every .cu file, and it includes the C math and stdlib headers, so
   programs call ceil or malloc without including them. */
#include <math.h>
#include <stdlib.h>

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#define __forceinline__ inline
#define __noinline__
#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))
#define __CUDACC__ 1
#define CUDART_VERSION 9000

struct uint3 { unsigned int x, y, z; };
struct dim3 {
    unsigned int x, y, z;
    __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};

/* The built-in variables: read by the interpreter from the launch, never from memory. */
extern const __device__ uint3 threadIdx;
extern const __device__ uint3 blockIdx;
extern const __device__ dim3 blockDim;
extern const __device__ dim3 gridDim;
extern const __device__ int warpSize;

#define VEC2(T, N) struct N##2 { T x, y; }; \
    static __host__ __device__ inline N##2 make_##N##2(T x, T y) { N##2 r; r.x = x; r.y = y; return r; }
#define VEC3(T, N) struct N##3 { T x, y, z; }; \
    static __host__ __device__ inline N##3 make_##N##3(T x, T y, T z) { N##3 r; r.x = x; r.y = y; r.z = z; return r; }
#define VEC4(T, N) struct N##4 { T x, y, z, w; }; \
    static __host__ __device__ inline N##4 make_##N##4(T x, T y, T z, T w) { N##4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }
VEC2(int, int) VEC3(int, int) VEC4(int, int)
VEC2(float, float) VEC3(float, float) VEC4(float, float)
VEC2(double, double) VEC3(double, double) VEC4(double, double)
VEC2(unsigned int, uint) VEC4(unsigned int, uint)
#undef VEC2
#undef VEC3
#undef VEC4

typedef enum cudaError { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 } cudaError_t;
typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4
} cudaMemcpyKind;
typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;
enum cudaFuncCache { cudaFuncCachePreferNone = 0, cudaFuncCachePreferShared = 1, cudaFuncCachePreferL1 = 2 };
#define cudaHostAllocDefault 0
#define cudaMemAttachGlobal 1
struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem, sharedMemPerBlock;
    int warpSize, maxThreadsPerBlock, multiProcessorCount, major, minor;
    int maxThreadsDim[3], maxGridSize[3];
};

/* Kernel launches: clang lowers k<<<g, b, s, st>>>(...) to a call configured by one of these: the first
   when it finds no CUDA installation (or one before 9.2), the second when it finds a newer one. */
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0,
                                                cudaStream_t stream = 0);

/* Runtime API (host). */
extern "C" {
cudaError_t cudaMalloc(void **ptr, size_t size);
cudaError_t cudaMallocManaged(void **ptr, size_t size, unsigned int flags = cudaMemAttachGlobal);
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags);
cudaError_t cudaFree(void *ptr);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
cudaError_t cudaMemset(void *ptr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *ptr, int value, size_t count, cudaStream_t stream = 0);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaThreadSynchronize(void);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char *cudaGetErrorString(cudaError_t error);
}
template <class T> cudaError_t cudaMalloc(T **ptr, size_t size);
template <class T> cudaError_t cudaMallocManaged(T **ptr, size_t size, unsigned int flags = cudaMemAttachGlobal);
template <class T> cudaError_t cudaMallocHost(T **ptr, size_t size);
template <class T> cudaError_t cudaFuncSetCacheConfig(T *func, enum cudaFuncCache config);
template <class T> cudaError_t cudaMemcpyToSymbol(const T &symbol, const void *src, size_t count, size_t offset = 0,
                                                  cudaMemcpyKind kind = cudaMemcpyHostToDevice);
template <class T> cudaError_t cudaMemcpyFromSymbol(void *dst, const T &symbol, size_t count, size_t offset = 0,
                                                    cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
template <class T> cudaError_t cudaGetSymbolAddress(void **devPtr, const T &symbol);
template <class T> cudaError_t cudaGetSymbolSize(size_t *size, const T &symbol);

/* Device functions. */
/* Fast single-precision intrinsics (evaluated as their exact counterparts, interp_cuda.CUDA_MATH). */
__device__ float __powf(float, float);
__device__ float __expf(float);
__device__ float __exp10f(float);
__device__ float __logf(float);
__device__ float __log2f(float);
__device__ float __log10f(float);
__device__ float __sinf(float);
__device__ float __cosf(float);
__device__ float __tanf(float);
__device__ float __fdividef(float, float);
__device__ float __fsqrt_rn(float);
__device__ float __saturatef(float);
__device__ float rsqrtf(float);
__device__ double rsqrt(double);
__device__ float exp10f(float);
__device__ double exp10(double);
extern "C" __device__ void __syncthreads(void);
extern "C" __device__ int __syncthreads_or(int);
extern "C" __device__ int __syncthreads_and(int);
extern "C" __device__ int __syncthreads_count(int);
extern "C" __device__ void __syncwarp(unsigned int mask = 0xffffffff);
extern "C" __device__ void __threadfence(void);
extern "C" __device__ void __threadfence_block(void);
__device__ int atomicAdd(int *address, int val);
__device__ unsigned int atomicAdd(unsigned int *address, unsigned int val);
__device__ unsigned long long atomicAdd(unsigned long long *address, unsigned long long val);
__device__ float atomicAdd(float *address, float val);
__device__ double atomicAdd(double *address, double val);
__device__ int atomicSub(int *address, int val);
__device__ unsigned int atomicSub(unsigned int *address, unsigned int val);
__device__ int atomicExch(int *address, int val);
__device__ unsigned int atomicExch(unsigned int *address, unsigned int val);
__device__ unsigned long long atomicExch(unsigned long long *address, unsigned long long val);
__device__ float atomicExch(float *address, float val);
__device__ int atomicMin(int *address, int val);
__device__ unsigned int atomicMin(unsigned int *address, unsigned int val);
__device__ long long atomicMin(long long *address, long long val);
__device__ int atomicMax(int *address, int val);
__device__ unsigned int atomicMax(unsigned int *address, unsigned int val);
__device__ long long atomicMax(long long *address, long long val);
__device__ int atomicCAS(int *address, int compare, int val);
__device__ unsigned int atomicCAS(unsigned int *address, unsigned int compare, unsigned int val);
__device__ unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val);
__device__ int atomicAnd(int *address, int val);
__device__ unsigned int atomicAnd(unsigned int *address, unsigned int val);
__device__ int atomicOr(int *address, int val);
__device__ unsigned int atomicOr(unsigned int *address, unsigned int val);
__device__ int atomicXor(int *address, int val);
__device__ unsigned int atomicXor(unsigned int *address, unsigned int val);
__device__ unsigned int atomicInc(unsigned int *address, unsigned int val);
__device__ unsigned int atomicDec(unsigned int *address, unsigned int val);
template <class T> __device__ T __ldg(const T *address);
__device__ int __shfl_sync(unsigned int mask, int var, int srcLane, int width = 32);
__device__ float __shfl_sync(unsigned int mask, float var, int srcLane, int width = 32);
__device__ int __shfl_down_sync(unsigned int mask, int var, unsigned int delta, int width = 32);
__device__ float __shfl_down_sync(unsigned int mask, float var, unsigned int delta, int width = 32);
__device__ int __shfl_xor_sync(unsigned int mask, int var, int laneMask, int width = 32);
__device__ float __shfl_xor_sync(unsigned int mask, float var, int laneMask, int width = 32);
__device__ unsigned int __ballot_sync(unsigned int mask, int predicate);
__device__ int __any_sync(unsigned int mask, int predicate);
__device__ int __all_sync(unsigned int mask, int predicate);
__device__ int __popc(unsigned int x);
__device__ int __clz(int x);
__device__ int __ffs(int x);

/* min and max as CUDA's math headers declare them for host and device code. */
#define MINMAX(T) \
    static __host__ __device__ inline T min(T a, T b) { return a < b ? a : b; } \
    static __host__ __device__ inline T max(T a, T b) { return a > b ? a : b; }
MINMAX(int) MINMAX(unsigned int) MINMAX(long long) MINMAX(unsigned long long) MINMAX(float) MINMAX(double)
#undef MINMAX

#endif

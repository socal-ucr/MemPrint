// What a Unified Memory page fault costs on this GPU: a kernel touches one word in every STRIDE bytes of a managed
// buffer whose pages are on the host (each block of STRIDE bytes faults), against the same kernel after the buffer
// was prefetched to the device (no fault), and against an explicit prefetch of the buffer alone.
//
//   nvcc -O2 -arch=native -o faultbench faultbench.cu && ./faultbench [STRIDE_KB=64] [REPS=5]
//
// Prints one line per buffer size: bytes, blocks touched, and the median times in microseconds of the kernel with
// faults, the kernel without, and the prefetch.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void touch(int *p, size_t words_per_block, size_t blocks) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < blocks)
        p[i * words_per_block] += 1;
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv) {
    size_t stride = (argc > 1 ? atol(argv[1]) : 64) * 1024;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    int dev = 0;
    CHECK(cudaGetDevice(&dev));
    cudaMemLocation gpu = {cudaMemLocationTypeDevice, dev}, host = {cudaMemLocationTypeHost, 0};
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    printf("bytes,blocks,fault_us,resident_us,prefetch_us\n");
    for (size_t bytes = stride; bytes <= ((size_t)512 << 20); bytes *= 2) {
        size_t blocks = bytes / stride;
        int *p;
        CHECK(cudaMallocManaged(&p, bytes));
        std::vector<float> fault, resident, prefetch;
        for (int r = 0; r < reps + 1; r++) {                     // the first round warms up
            memset(p, 0, bytes);                                  // pages on the host
            CHECK(cudaMemPrefetchAsync(p, bytes, host, 0, 0));
            CHECK(cudaDeviceSynchronize());
            int threads = 256, grid = (int)((blocks + threads - 1) / threads);
            float t;
            CHECK(cudaEventRecord(a));
            touch<<<grid, threads>>>(p, stride / sizeof(int), blocks);
            CHECK(cudaEventRecord(b));
            CHECK(cudaEventSynchronize(b));
            CHECK(cudaEventElapsedTime(&t, a, b));
            if (r) fault.push_back(t * 1000);
            CHECK(cudaEventRecord(a));
            touch<<<grid, threads>>>(p, stride / sizeof(int), blocks);   // now on the device
            CHECK(cudaEventRecord(b));
            CHECK(cudaEventSynchronize(b));
            CHECK(cudaEventElapsedTime(&t, a, b));
            if (r) resident.push_back(t * 1000);
            CHECK(cudaMemPrefetchAsync(p, bytes, host, 0, 0));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(a));
            CHECK(cudaMemPrefetchAsync(p, bytes, gpu, 0, 0));
            CHECK(cudaEventRecord(b));
            CHECK(cudaEventSynchronize(b));
            CHECK(cudaEventElapsedTime(&t, a, b));
            if (r) prefetch.push_back(t * 1000);
        }
        printf("%zu,%zu,%.1f,%.1f,%.1f\n", bytes, blocks, median(fault), median(resident), median(prefetch));
        CHECK(cudaFree(p));
    }
    return 0;
}

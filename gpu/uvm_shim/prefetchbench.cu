// What a cudaMemPrefetchAsync plan costs on this GPU: K separate calls of one 64 KB block each (at a stride of two
// blocks, so no two are contiguous) against one call for the same number of contiguous blocks, from the host to the
// device, timed from the first call to the end of the stream.
//
//   nvcc -O2 -arch=native -o prefetchbench prefetchbench.cu && ./prefetchbench [REPS=5]
//
// Prints one line per K: the median times in microseconds of the K calls and of the one call.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 5;
    const size_t block = 64 << 10, most = 1024;
    int dev = 0;
    CHECK(cudaGetDevice(&dev));
    cudaMemLocation gpu = {cudaMemLocationTypeDevice, dev}, host = {cudaMemLocationTypeHost, 0};
    char *p;
    CHECK(cudaMallocManaged(&p, 2 * most * block));
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    printf("calls,separate_us,contiguous_us\n");
    for (size_t k = 1; k <= most; k *= 2) {
        std::vector<float> sep, con;
        for (int r = 0; r < reps + 1; r++) {                     // the first round warms up
            for (int contiguous = 0; contiguous < 2; contiguous++) {
                memset(p, 1, 2 * most * block);                   // pages on the host
                CHECK(cudaMemPrefetchAsync(p, 2 * most * block, host, 0, 0));
                CHECK(cudaDeviceSynchronize());
                float t;
                CHECK(cudaEventRecord(a));
                if (contiguous)
                    CHECK(cudaMemPrefetchAsync(p, k * block, gpu, 0, 0));
                else
                    for (size_t i = 0; i < k; i++)
                        CHECK(cudaMemPrefetchAsync(p + 2 * i * block, block, gpu, 0, 0));
                CHECK(cudaEventRecord(b));
                CHECK(cudaEventSynchronize(b));
                CHECK(cudaEventElapsedTime(&t, a, b));
                if (r) (contiguous ? con : sep).push_back(t * 1000);
            }
        }
        printf("%zu,%.1f,%.1f\n", k, median(sep), median(con));
    }
    return 0;
}

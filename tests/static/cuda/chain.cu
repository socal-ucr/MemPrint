// A chain of elementwise kernels: init(a), init(b), add(c = a + b), scale(c *= 2), copy back.
// Expected dependences: add <- init a, init b (RAW); scale <- add (RAW, WAW); D2H <- scale (RAW).
#include <cuda_runtime.h>
#include <stdlib.h>

void probe(double v);  // test hook: the interpreter records the value

__global__ void init(float *x, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v * i;
}

__global__ void add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] + b[i];
}

__global__ void scale(float *c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        c[i] *= 2.0f;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1000;
    float *a, *b, *c;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&c, n * sizeof(float));
    int threads = 128, blocks = (n + threads - 1) / threads;
    init<<<blocks, threads>>>(a, 1.0f, n);
    init<<<blocks, threads>>>(b, 2.0f, n);
    add<<<blocks, threads>>>(a, b, c, n);
    scale<<<4, 64>>>(c, n);
    float *h = (float *)malloc(n * sizeof(float));
    cudaMemcpy(h, c, n * sizeof(float), cudaMemcpyDeviceToHost);
    probe(h[3]);
    return 0;
}

// A shared-memory tree reduction per block (__syncthreads between steps), then a second kernel that sums
// the per-block partials. The partials array is written by every block and read by the second kernel.
#include <cuda_runtime.h>
#include <stdlib.h>

void probe(long v);

__global__ void block_sum(const int *in, int *partial, int n) {
    __shared__ int s[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    s[t] = i < n ? in[i] : 0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.x] = s[0];
}

__global__ void final_sum(const int *partial, int *out, int m) {
    if (threadIdx.x != 0) return;
    int acc = 0;
    for (int b = 0; b < m; b++) acc += partial[b];
    out[0] = acc;
}

int main() {
    const int n = 5000, threads = 256, blocks = (n + threads - 1) / threads;
    int *h = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) h[i] = i % 97;
    int *d_in, *d_partial, *d_out;
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_partial, blocks * sizeof(int));
    cudaMalloc(&d_out, sizeof(int));
    cudaMemcpy(d_in, h, n * sizeof(int), cudaMemcpyHostToDevice);
    block_sum<<<blocks, threads>>>(d_in, d_partial, n);
    final_sum<<<1, 64>>>(d_partial, d_out, blocks);
    int out = 0, p0 = 0;
    cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&p0, d_partial, sizeof(int), cudaMemcpyDeviceToHost);
    probe(out);
    probe(p0);
    return 0;
}

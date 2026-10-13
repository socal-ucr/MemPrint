// Indirect accesses: out[idx[i]] with idx built on the host (the even slots 0, 2, ..., 2n-2, in a scrambled
// order) and copied to the device. gather_even reads exactly the slots the scatter wrote (a RAW dependence
// through n units); gather_odd reads only the others (no dependence on the scatter, only on the memset).
#include <cuda_runtime.h>
#include <stdlib.h>

void probe(long v);

__global__ void scatter(int *out, const int *idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[idx[i]] = i + 1;
}

__global__ void gather_even(const int *out, int *res, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] = out[2 * i];
}

__global__ void gather_odd(const int *out, int *res, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] = out[2 * i + 1];
}

int main() {
    const int n = 1000;
    int *h_idx = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) h_idx[i] = 2 * ((i * 389) % n);   // a permutation of the even slots
    int *d_idx, *d_out, *d_even, *d_odd;
    cudaMalloc(&d_idx, n * sizeof(int));
    cudaMalloc(&d_out, 2 * n * sizeof(int));
    cudaMalloc(&d_even, n * sizeof(int));
    cudaMalloc(&d_odd, n * sizeof(int));
    cudaMemcpy(d_idx, h_idx, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 2 * n * sizeof(int));
    scatter<<<(n + 255) / 256, 256>>>(d_out, d_idx, n);
    gather_even<<<(n + 255) / 256, 256>>>(d_out, d_even, n);
    gather_odd<<<(n + 255) / 256, 256>>>(d_out, d_odd, n);
    int *h = (int *)malloc(n * sizeof(int));
    cudaMemcpy(h, d_even, n * sizeof(int), cudaMemcpyDeviceToHost);
    probe(h[0]);    // out[0] = 1 + the i with (i * 389) % n == 0: i = 0
    probe(h[389]);  // out[778] = 1 + i with (i * 389) % 1000 == 389: i = 1
    return 0;
}

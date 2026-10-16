// __device__ and __constant__ variables: coefficients copied to a __constant__ array, a __device__ table
// filled at indices chosen by the data, a __device__ counter, a pointer to the table obtained with
// cudaGetSymbolAddress, and copies back with cudaMemcpyFromSymbol. The dependences through the variables are
// not in the GPU trace (the NVBit tool sees only cudaMalloc buffers); the ones through buffers are.
#include <cuda_runtime.h>
#include <stdlib.h>

void probe(long v);

__constant__ int coef[4];
__device__ int table[256];
__device__ int counter;

__global__ void scale(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i] * coef[i % 4];
    atomicAdd(&counter, 1);
}

__global__ void fill(const int *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&table[in[i] % 256], 1);
}

__global__ void gather(const int *t, const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += t[in[i] % 256];
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1000;
    int *h = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) h[i] = (i * 37) % 1000;
    int c[4] = {1, 2, 3, 4};
    int *d_in, *d_out, *d_table;
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMemcpy(d_in, h, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(coef, c, sizeof c);
    int threads = 128, blocks = (n + threads - 1) / threads;
    scale<<<blocks, threads>>>(d_in, d_out, n);
    fill<<<blocks, threads>>>(d_in, n);
    cudaGetSymbolAddress((void **)&d_table, table);
    gather<<<blocks, threads>>>(d_table, d_in, d_out, n);
    int count, t[256];
    cudaMemcpyFromSymbol(&count, counter, sizeof count);
    cudaMemcpyFromSymbol(t, table, sizeof t);
    cudaMemcpy(h, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    long s = 0;
    for (int i = 0; i < n; i++) s += h[i];
    probe(count);
    probe(t[37]);
    probe(s);
    return 0;
}

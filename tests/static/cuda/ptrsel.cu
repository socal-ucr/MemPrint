// Buffers chosen by the data: a table of device pointers lives in device memory, and the producer writes
// each element into the bucket its key selects (through the table, at an atomicAdd slot). One consumer per
// bucket then reads its bucket. Which buckets the producer writes, and so which consumers depend on it,
// depends on the keys; with these keys bucket 2 stays empty, so its consumer reads nothing it wrote.
#include <cuda_runtime.h>
#include <stdlib.h>

void probe(long v);

#define K 4

__global__ void distribute(const int *keys, int *const *bucket, int *count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int k = keys[i] % K;
    bucket[k][atomicAdd(&count[k], 1)] = i;
}

__global__ void consume(const int *b, const int *count, int k, int *sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count[k]) atomicAdd(&sum[k], b[i]);
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1000;
    int *keys = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) keys[i] = (i * 7) % 3 == 0 ? 0 : (i % 2 ? 1 : 3);   // never 2 mod 4
    int *d_keys, *d_count, *d_sum, **d_table, *d_bucket[K];
    cudaMalloc(&d_keys, n * sizeof(int));
    for (int k = 0; k < K; k++) cudaMalloc(&d_bucket[k], n * sizeof(int));
    cudaMalloc(&d_table, K * sizeof(int *));
    cudaMalloc(&d_count, K * sizeof(int));
    cudaMalloc(&d_sum, K * sizeof(int));
    cudaMemcpy(d_keys, keys, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table, d_bucket, K * sizeof(int *), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, K * sizeof(int));
    cudaMemset(d_sum, 0, K * sizeof(int));
    distribute<<<(n + 127) / 128, 128>>>(d_keys, d_table, d_count, n);
    for (int k = 0; k < K; k++) consume<<<(n + 127) / 128, 128>>>(d_bucket[k], d_count, k, d_sum);
    int sum[K];
    cudaMemcpy(sum, d_sum, K * sizeof(int), cudaMemcpyDeviceToHost);
    for (int k = 0; k < K; k++) probe(sum[k]);
    return 0;
}

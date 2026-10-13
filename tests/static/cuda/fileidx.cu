// The index array comes from a file named on the command line (fscanf: a count, then the indices), and
// a float weight array from a binary file (fread). update writes w only at the listed indices; sum reads
// the first half of w, so the dependence covers exactly the listed indices below n / 2.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void probe(double v);

__global__ void update(float *w, const int *idx, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) w[idx[i]] += 1.0f;
}

__global__ void sum_half(const float *w, float *out, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0;
    for (int i = 0; i < n / 2; i++) s += w[i];
    *out = s;
}

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "r");
    int k;
    fscanf(f, "%d", &k);
    int *idx = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) fscanf(f, "%d", &idx[i]);
    fclose(f);
    int n = 4096;
    float *w = (float *)malloc(n * sizeof(float));
    FILE *g = fopen(argv[2], "rb");
    fread(w, sizeof(float), n, g);
    fclose(g);
    float *d_w, *d_out;
    int *d_idx;
    cudaMalloc(&d_w, n * sizeof(float));
    cudaMalloc(&d_idx, k * sizeof(int));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_w, w, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, idx, k * sizeof(int), cudaMemcpyHostToDevice);
    update<<<(k + 63) / 64, 64>>>(d_w, d_idx, k);
    sum_half<<<1, 32>>>(d_w, d_out, n);
    float out;
    cudaMemcpy(&out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    probe(k);
    probe(out);
    return 0;
}

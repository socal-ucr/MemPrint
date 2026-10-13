// atomicAdd into bins chosen by the data: the bins a launch writes depend on the input.
// atomicMax and atomicMin over all threads at one address; a second kernel reads the histogram.
#include <cuda_runtime.h>
#include <stdlib.h>

void probe(long v);

__global__ void hist(const int *data, int *bins, int *hi, int *lo, int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int old = atomicAdd(&bins[data[i] % nbins], 1);
        atomicMax(hi, data[i]);
        atomicMin(lo, data[i] + old - old);
    }
}

__global__ void scan_bins(const int *bins, int *total, int nbins) {
    int s = 0;
    for (int b = 0; b < nbins; b++) s += bins[b];
    if (threadIdx.x == 0) *total = s;
}

int main() {
    const int n = 3000, nbins = 64;
    int *h = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) h[i] = (i * i + 7 * i) % 1000;   // bins 0..63 hit unevenly
    int *d, *bins, *hi, *lo, *total;
    cudaMalloc(&d, n * sizeof(int));
    cudaMalloc(&bins, 128 * sizeof(int));                         // only the first nbins are used
    cudaMalloc(&hi, sizeof(int));
    cudaMalloc(&lo, sizeof(int));
    cudaMalloc(&total, sizeof(int));
    cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(bins, 0, 128 * sizeof(int));
    int big = 1 << 30, zero = 0;
    cudaMemcpy(lo, &big, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(hi, &zero, sizeof(int), cudaMemcpyHostToDevice);
    hist<<<(n + 255) / 256, 256>>>(d, bins, hi, lo, n, nbins);
    scan_bins<<<1, 32>>>(bins, total, nbins);
    int hb[64], t, mx, mn;
    cudaMemcpy(hb, bins, nbins * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&t, total, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&mx, hi, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&mn, lo, sizeof(int), cudaMemcpyDeviceToHost);
    probe(hb[0]);
    probe(hb[7]);
    probe(t);
    probe(mx);
    probe(mn);
    return 0;
}

// Connected components by label propagation (Pannotia / Lonestar style): every round, each edge pulls the
// smaller label of its endpoints into both with atomicMin, in place, until a round changes nothing. A label
// read in a round can be one another thread lowered earlier in the same round, so how many rounds it takes,
// and so how many launches there are, can depend on how threads interleave.
#include <cuda_runtime.h>
#include <stdlib.h>

void probe(long v);

__global__ void init(int *label, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) label[v] = v;
}

__global__ void hook(const int *src, const int *dst, int *label, int *changed, int m) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= m) return;
    int a = label[src[e]], b = label[dst[e]];
    if (a < b && atomicMin(&label[dst[e]], a) > a) *changed = 1;
    if (b < a && atomicMin(&label[src[e]], b) > b) *changed = 1;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 4096;
    int m = n;                                       // sparse: many components
    int *src = (int *)malloc(m * sizeof(int));
    int *dst = (int *)malloc(m * sizeof(int));
    unsigned s = 777;
    for (int e = 0; e < m; e++) {
        s = s * 1103515245u + 12345u;
        src[e] = (s >> 8) % n;
        s = s * 1103515245u + 12345u;
        dst[e] = (s >> 8) % n;
    }
    int *d_src, *d_dst, *d_label, *d_changed;
    cudaMalloc(&d_src, m * sizeof(int));
    cudaMalloc(&d_dst, m * sizeof(int));
    cudaMalloc(&d_label, n * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMemcpy(d_src, src, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst, m * sizeof(int), cudaMemcpyHostToDevice);
    init<<<(n + 255) / 256, 256>>>(d_label, n);
    int changed = 1, rounds = 0;
    while (changed) {
        cudaMemset(d_changed, 0, sizeof(int));
        hook<<<(m + 255) / 256, 256>>>(d_src, d_dst, d_label, d_changed, m);
        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        rounds++;
    }
    int *label = (int *)malloc(n * sizeof(int));
    cudaMemcpy(label, d_label, n * sizeof(int), cudaMemcpyDeviceToHost);
    int comps = 0;
    for (int v = 0; v < n; v++) comps += label[v] == v;
    probe(rounds);
    probe(comps);
    return 0;
}

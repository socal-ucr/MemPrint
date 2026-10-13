// Level-synchronous BFS on a CSR graph built on the host: each level is a launch of expand, and the host
// loops while the device sets a flag (cudaMemset, launch, copy back). The number of launches, and which
// vertices each one claims, depend on the graph: input-dependent dependences under data control.
#include <cuda_runtime.h>
#include <stdlib.h>

void probe(long v);

__global__ void expand(const int *row, const int *col, int *dist, int *changed, int level, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n || dist[v] != level) return;
    for (int e = row[v]; e < row[v + 1]; e++) {
        int u = col[e];
        if (atomicCAS(&dist[u], -1, level + 1) == -1) *changed = 1;
    }
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2000;
    int deg = 2;
    int *row = (int *)malloc((n + 1) * sizeof(int));
    int *col = (int *)malloc(n * deg * sizeof(int));
    for (int v = 0; v <= n; v++) row[v] = v * deg;
    for (int v = 0; v < n; v++) {                    // a sparse pseudo-random graph
        col[v * deg] = (v * 7 + 1) % n;
        col[v * deg + 1] = (v * 13 + 5) % n;
    }
    int *dist = (int *)malloc(n * sizeof(int));
    for (int v = 0; v < n; v++) dist[v] = -1;
    dist[0] = 0;
    int *d_row, *d_col, *d_dist, *d_changed;
    cudaMalloc(&d_row, (n + 1) * sizeof(int));
    cudaMalloc(&d_col, n * deg * sizeof(int));
    cudaMalloc(&d_dist, n * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMemcpy(d_row, row, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col, n * deg * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, n * sizeof(int), cudaMemcpyHostToDevice);
    int level = 0, changed = 1;
    while (changed) {
        cudaMemset(d_changed, 0, sizeof(int));
        expand<<<(n + 127) / 128, 128>>>(d_row, d_col, d_dist, d_changed, level, n);
        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        level++;
    }
    cudaMemcpy(dist, d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
    int reached = 0, far = 0;
    for (int v = 0; v < n; v++) {
        if (dist[v] >= 0) reached++;
        if (dist[v] > far) far = dist[v];
    }
    probe(level);
    probe(reached);
    probe(far);
    return 0;
}

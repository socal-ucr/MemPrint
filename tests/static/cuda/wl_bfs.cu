// Worklist BFS (the frontier-queue style of Gunrock and LonestarGPU): each level is a launch over the
// input queue only, which pushes newly reached vertices to the output queue at atomicAdd positions. The
// host copies the output length back, sizes the next launch by it and swaps the queues. Which slots hold
// which vertex depends on how threads interleave; the set of vertices, and so the units of dist and row the
// next level reads through the queue, does not.
#include <cuda_runtime.h>
#include <stdlib.h>

void probe(long v);

__global__ void expand(const int *row, const int *col, int *dist, const int *in, int in_len, int *out,
                       int *out_len, int level) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in_len) return;
    int v = in[i];
    for (int e = row[v]; e < row[v + 1]; e++) {
        int u = col[e];
        if (atomicCAS(&dist[u], -1, level + 1) == -1) out[atomicAdd(out_len, 1)] = u;
    }
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 4096;
    int *row = (int *)malloc((n + 1) * sizeof(int));
    row[0] = 0;
    unsigned s = 12345;
    for (int v = 0; v < n; v++) {                    // 1 to 4 edges per vertex, pseudo-random targets
        s = s * 1103515245u + 12345u;
        row[v + 1] = row[v] + 1 + (s >> 16) % 4;
    }
    int m = row[n];
    int *col = (int *)malloc(m * sizeof(int));
    for (int e = 0; e < m; e++) {
        s = s * 1103515245u + 12345u;
        col[e] = (s >> 8) % n;
    }
    int *dist = (int *)malloc(n * sizeof(int));
    for (int v = 0; v < n; v++) dist[v] = -1;
    dist[0] = 0;
    int *d_row, *d_col, *d_dist, *d_in, *d_out, *d_len;
    cudaMalloc(&d_row, (n + 1) * sizeof(int));
    cudaMalloc(&d_col, m * sizeof(int));
    cudaMalloc(&d_dist, n * sizeof(int));
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMalloc(&d_len, sizeof(int));
    cudaMemcpy(d_row, row, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, n * sizeof(int), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_in, &zero, sizeof(int), cudaMemcpyHostToDevice);
    int len = 1, level = 0;
    while (len > 0) {
        cudaMemset(d_len, 0, sizeof(int));
        expand<<<(len + 127) / 128, 128>>>(d_row, d_col, d_dist, d_in, len, d_out, d_len, level);
        cudaMemcpy(&len, d_len, sizeof(int), cudaMemcpyDeviceToHost);
        int *t = d_in;
        d_in = d_out;
        d_out = t;
        level++;
    }
    cudaMemcpy(dist, d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
    int reached = 0;
    for (int v = 0; v < n; v++) reached += dist[v] >= 0;
    probe(level);
    probe(reached);
    return 0;
}

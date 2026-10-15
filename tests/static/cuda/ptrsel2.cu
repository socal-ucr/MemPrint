// More buffers chosen by the data, in the shapes graph codes use. Each vertex is a struct whose pointer
// field names its neighbour list in one of two arrays (low and high degree vertices are stored apart); a
// kernel walks the list through a pointer it keeps in a local, inside a loop. A second kernel picks an
// output buffer through a table by the vertex's class, the pointer kept in a local across statements. A
// third does an atomic before the load in the same statement (a comma expression), where the interpreter
// cannot split the threads: its references through the pointer stay may references, to any buffer.
#include <cuda_runtime.h>
#include <stdlib.h>

void probe(long v);

struct Vertex {
    int deg;
    const int *nbrs;  // into lo_adj or hi_adj
};

__global__ void degree_sum(const Vertex *vs, const int *weight, int *out, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const int *p = vs[v].nbrs;
    int s = 0;
    for (int j = 0; j < vs[v].deg; j++) s += weight[p[j]];
    out[v] = s;
}

__global__ void split_out(const int *cls, const int *val, int *const *outs, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int *o = outs[cls[v]];
    o[v] = val[v] * 2;
}

__global__ void late_load(const int *cls, int *const *outs, int *cnt, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int *o = (atomicAdd(cnt, 1), outs[cls[v]]);     // the atomic runs first: no split, may references
    o[v] += 1;
}

__global__ void sum(const int *a, int *s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(s, a[i]);
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2048;
    int *deg = (int *)malloc(n * sizeof(int)), *cls = (int *)malloc(n * sizeof(int));
    int lo_m = 0, hi_m = 0;
    for (int v = 0; v < n; v++) {
        deg[v] = (v % 17 == 0) ? 8 : 1 + v % 2;
        cls[v] = deg[v] > 2;
        if (cls[v]) hi_m += deg[v]; else lo_m += deg[v];
    }
    int *lo_adj = (int *)malloc(lo_m * sizeof(int)), *hi_adj = (int *)malloc(hi_m * sizeof(int));
    for (int e = 0; e < lo_m; e++) lo_adj[e] = (e * 37 + 11) % n;
    for (int e = 0; e < hi_m; e++) hi_adj[e] = (e * 53 + 7) % n;
    int *weight = (int *)malloc(n * sizeof(int));
    for (int v = 0; v < n; v++) weight[v] = v % 7;
    int *d_lo, *d_hi, *d_weight, *d_out, *d_cls, *d_a, *d_b, *d_cnt, *d_s, **d_outs;
    Vertex *d_vs;
    cudaMalloc(&d_lo, lo_m * sizeof(int));
    cudaMalloc(&d_hi, hi_m * sizeof(int));
    cudaMalloc(&d_weight, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMalloc(&d_cls, n * sizeof(int));
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_cnt, sizeof(int));
    cudaMalloc(&d_s, 2 * sizeof(int));
    cudaMalloc(&d_outs, 2 * sizeof(int *));
    cudaMalloc(&d_vs, n * sizeof(Vertex));
    Vertex *vs = (Vertex *)malloc(n * sizeof(Vertex));
    int lo_at = 0, hi_at = 0;
    for (int v = 0; v < n; v++) {                    // device addresses into the two lists, set on the host
        vs[v].deg = deg[v];
        if (cls[v]) { vs[v].nbrs = d_hi + hi_at; hi_at += deg[v]; }
        else { vs[v].nbrs = d_lo + lo_at; lo_at += deg[v]; }
    }
    int *outs[2] = {d_a, d_b};
    cudaMemcpy(d_lo, lo_adj, lo_m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hi, hi_adj, hi_m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, vs, n * sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cls, cls, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outs, outs, sizeof outs, cudaMemcpyHostToDevice);
    cudaMemset(d_a, 0, n * sizeof(int));
    cudaMemset(d_b, 0, n * sizeof(int));
    cudaMemset(d_cnt, 0, sizeof(int));
    cudaMemset(d_s, 0, 2 * sizeof(int));
    int blocks = (n + 127) / 128;
    degree_sum<<<blocks, 128>>>(d_vs, d_weight, d_out, n);
    split_out<<<blocks, 128>>>(d_cls, d_out, d_outs, n);
    sum<<<blocks, 128>>>(d_a, d_s, n);                // reads only what split_out wrote for class 0
    sum<<<blocks, 128>>>(d_b, d_s + 1, n);
    late_load<<<blocks, 128>>>(d_cls, d_outs, d_cnt, n);
    int s[2];
    cudaMemcpy(s, d_s, sizeof s, cudaMemcpyDeviceToHost);
    probe(s[0]);
    probe(s[1]);
    return 0;
}

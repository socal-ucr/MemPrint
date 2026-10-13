// BFS in the style of Rodinia's bfs: the graph is read from a file in Rodinia's format (node count, then
// "start degree" per node, the source, the edge count, then "destination cost" per edge), nodes are an
// array of structs, the frontier and visited sets are bool arrays, and each level is two kernels (expand the
// frontier, then commit the next one) that the host repeats while the device sets a flag.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void probe(long v);

struct Node {
    int starting;
    int no_of_edges;
};

__global__ void expand(const Node *nodes, const int *edges, bool *mask, bool *updating, const bool *visited,
                       int *cost, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && mask[tid]) {
        mask[tid] = false;
        for (int i = nodes[tid].starting; i < nodes[tid].starting + nodes[tid].no_of_edges; i++) {
            int id = edges[i];
            if (!visited[id]) {
                cost[id] = cost[tid] + 1;
                updating[id] = true;
            }
        }
    }
}

__global__ void commit(bool *mask, bool *updating, bool *visited, bool *over, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && updating[tid]) {
        mask[tid] = true;
        visited[tid] = true;
        *over = true;
        updating[tid] = false;
    }
}

int main(int argc, char **argv) {
    FILE *fp = fopen(argv[1], "r");
    int n, source = 0, edge_list_size;
    fscanf(fp, "%d", &n);
    Node *nodes = (Node *)malloc(sizeof(Node) * n);
    bool *mask = (bool *)malloc(sizeof(bool) * n);
    bool *updating = (bool *)malloc(sizeof(bool) * n);
    bool *visited = (bool *)malloc(sizeof(bool) * n);
    for (int i = 0; i < n; i++) {
        int start, degree;
        fscanf(fp, "%d %d", &start, &degree);
        nodes[i].starting = start;
        nodes[i].no_of_edges = degree;
        mask[i] = false;
        updating[i] = false;
        visited[i] = false;
    }
    fscanf(fp, "%d", &source);
    mask[source] = true;
    visited[source] = true;
    fscanf(fp, "%d", &edge_list_size);
    int *edges = (int *)malloc(sizeof(int) * edge_list_size);
    for (int i = 0; i < edge_list_size; i++) {
        int id, cost;
        fscanf(fp, "%d %d", &id, &cost);
        edges[i] = id;
    }
    fclose(fp);
    int *cost = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) cost[i] = -1;
    cost[source] = 0;

    Node *d_nodes;
    int *d_edges, *d_cost;
    bool *d_mask, *d_updating, *d_visited, *d_over;
    cudaMalloc((void **)&d_nodes, sizeof(Node) * n);
    cudaMalloc((void **)&d_edges, sizeof(int) * edge_list_size);
    cudaMalloc((void **)&d_mask, sizeof(bool) * n);
    cudaMalloc((void **)&d_updating, sizeof(bool) * n);
    cudaMalloc((void **)&d_visited, sizeof(bool) * n);
    cudaMalloc((void **)&d_cost, sizeof(int) * n);
    cudaMalloc((void **)&d_over, sizeof(bool));
    cudaMemcpy(d_nodes, nodes, sizeof(Node) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges, sizeof(int) * edge_list_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, sizeof(bool) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_updating, updating, sizeof(bool) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, visited, sizeof(bool) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost, cost, sizeof(int) * n, cudaMemcpyHostToDevice);

    int threads = n < 512 ? n : 512, blocks = (n + threads - 1) / threads;
    int k = 0;
    bool stop;
    do {
        stop = false;
        cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice);
        expand<<<blocks, threads>>>(d_nodes, d_edges, d_mask, d_updating, d_visited, d_cost, n);
        commit<<<blocks, threads>>>(d_mask, d_updating, d_visited, d_over, n);
        cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
        k++;
    } while (stop);

    cudaMemcpy(cost, d_cost, sizeof(int) * n, cudaMemcpyDeviceToHost);
    long total = 0;
    int reached = 0;
    for (int i = 0; i < n; i++) {
        if (cost[i] >= 0) {
            total += cost[i];
            reached++;
        }
    }
    probe(k);
    probe(reached);
    probe(total);
    return 0;
}

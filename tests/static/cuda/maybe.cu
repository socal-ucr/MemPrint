// Values the interpreter cannot know (drand48 is not modelled): the kernel's branch on them is uncertain,
// so its writes are may references, and so are the dependences through them. The unconditional write is
// still certain.
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void threshold(const double *x, int *flag, int *seen, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    seen[i] = 1;
    if (x[i] > 0.5) flag[i] = 1;
}

__global__ void count(const int *flag, const int *seen, int *out, int n) {
    if (threadIdx.x != 0) return;
    int c = 0;
    for (int i = 0; i < n; i++) c += flag[i] + seen[i];
    *out = c;
}

int main() {
    const int n = 512;
    double *h = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) h[i] = drand48();
    double *x;
    int *flag, *seen, *out;
    cudaMalloc(&x, n * sizeof(double));
    cudaMalloc(&flag, n * sizeof(int));
    cudaMalloc(&seen, n * sizeof(int));
    cudaMalloc(&out, sizeof(int));
    cudaMemcpy(x, h, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(flag, 0, n * sizeof(int));
    threshold<<<n / 128, 128>>>(x, flag, seen, n);
    count<<<1, 32>>>(flag, seen, out, n);
    return 0;
}

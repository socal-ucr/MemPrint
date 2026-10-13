// A 1-D Jacobi stencil in ping-pong buffers: step t reads one buffer and writes the interior of the other,
// so step t+1 reads what step t wrote (RAW) and overwrites what step t-1 wrote and step t read (WAW, WAR);
// only the chain t -> t+1 is needed, the rest follow from it.
#include <cuda_runtime.h>

void probe(double v);

__global__ void jacobi(const float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < n - 1) b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0f;
}

int main() {
    const int n = 1024, steps = 4;
    float *a, *b;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMemset(a, 0, n * sizeof(float));
    cudaMemset(b, 0, n * sizeof(float));
    for (int t = 0; t < steps; t++) {
        jacobi<<<n / 128, 128>>>(a, b, n);
        float *tmp = a; a = b; b = tmp;
    }
    float h[4];
    cudaMemcpy(h, a, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    probe(h[1]);
    return 0;
}

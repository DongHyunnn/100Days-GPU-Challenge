#include <cuda_runtime.h>

__global__ void vector_scale(const float* x, float alpha, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = alpha * x[i];
    }
}

extern "C" void solve(const float* x, float alpha, float* y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_scale<<<blocksPerGrid, threadsPerBlock>>>(x, alpha, y, N);
    cudaDeviceSynchronize();
}

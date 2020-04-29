#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


__global__
void add_kernel(float *A, float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

void add_cuda(at::Tensor A, at::Tensor B, at::Tensor C) {

    int d = A.size(0);
    float *h_A = A.data_ptr<float>();
    float *h_B = B.data_ptr<float>();
    float *h_C = C.data_ptr<float>();

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, d * sizeof(float));
    cudaMalloc(&d_B, d * sizeof(float));
    cudaMalloc(&d_C, d * sizeof(float));

    cudaMemcpy(d_A, h_A, d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, d * sizeof(float), cudaMemcpyHostToDevice);

    add_kernel << < 1, d >> > (d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, d * sizeof(float), cudaMemcpyDeviceToHost);
}
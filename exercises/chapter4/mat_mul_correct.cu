#include <stdio.h>
#define N 1024  // Matrix size (N x N)

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    if (row < n && col < n) {
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matrixMultiply(float *A, float *B, float *C, int n) {
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Copy result matrix from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int n = N;
    size_t size = n * n * sizeof(float);

    // Allocate memory for matrices on host
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < n * n; ++i) {
        A[i] = 1.0f;  // or some meaningful values
        B[i] = 1.0f;  // or some meaningful values
    }

    // Perform matrix multiplication
    matrixMultiply(A, B, C, n);

    // Optionally, print the result matrix C
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         printf("%f ", C[i * n + j]);
    //     }
    //     printf("\n");
    // }

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
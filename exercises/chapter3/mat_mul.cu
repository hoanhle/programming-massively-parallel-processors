#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Kernel for row-wise matrix multiplication
__global__ void matrixMulRowKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        for (int col = 0; col < N; ++col) {
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// Kernel for column-wise matrix multiplication
__global__ void matrixMulColumnKernel(float* A, float* B, float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        for (int row = 0; row < N; ++row) {
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// Function to initialize matrices with random values
void initializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to print matrices
void printMatrix(float* matrix, int N, const char* name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    int N = 4; // Example size
    size_t size = N * N * sizeof(float);

    // Allocate host matrices
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host matrices
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    // Print initial matrices A and B
    printMatrix(h_A, N, "A");
    printMatrix(h_B, N, "B");

    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    // Launch row-wise kernel
    matrixMulRowKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back to host and print
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printMatrix(h_C, N, "C (Row-wise)");

    // Launch column-wise kernel
    matrixMulColumnKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back to host and print
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printMatrix(h_C, N, "C (Column-wise)");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

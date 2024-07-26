1. Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

```
A = [a11 a12]    B = [b11 b12]    C = [a11+b11 a12+b12]
    [a21 a22]        [b21 b22]        [a21+b21 a22+b22]
```

There is no reuse of elements across threads, each element is used exactly once. Therefore, there is no advantage in loading it into shared memory for multiple threads to access.

2. Draw the equivalent of Fig.5.7 for a 8x8 matrix multiplication with 2x2 tiling and 4x4 tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.

Note: `M(0,0)` is used twice by 2x2 tiling, and four times by 4x4 tiling. Reduction in global memory bandwidth is proportional to dimension size of tile.

3. What type of incorrect execution behavior can happen if one forgot to use one or both `__syncthreads()` in the kernel of Fig. 5.9?

In the first `__syncthreads()`
- Race condition: Some threads might start computing before all the data is loaded into shared memory.
- Reading Uninitialized data: Threads might read from `Mds` and `Nds` before the data has been written (using garbage value or data from previous iteration).
- Inconsistent State: Different threads might be working with different version of the shared memory data

In the second `__syncthreads()`
- Data corruption: Faster thread might start the next iteration and overwrite the shared memory before slower threads have finished using the current data
- Inconsistent iterations: some threads might be working on the next tile

5. Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer

Registers:
- Private to each thread

Shared memory:
- Shared among all threads in a block

While registers are faster, the ability of shared memory to facilitate data sharing and reuse across threads in a block make it invaluable for optimizing global memory access patterns.

6. Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?

1000 * 512 = 512000 versions

7. In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?

1000 versions

8. Consider performing a matrix multiplication of two input matrices with dimensions N x N. How many times is each element in the input matrices requested from global memory when:
a. There is no tiling? N times
b. Tiles of size TxT are used? N / T times

9. A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory-bound.
a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second 
b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second

The kernel's computational intensity: 35 operations / 7 * 4 bytes (32-bit) = 1.29 OP/B
The device's computational intensity
- a: 200 GFLOPS / 100GB/s = 2 OP/B (the kernel is memory bound)
- b: 300 GFLOPS / 250GB/s = 1.2 OP/s (the kernel is compute bound)

10. To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.
 
a. Out of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will this kernel function execute correctly on the device?
```C
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width / blockDim.x, A_height / blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void BlockTranspose(float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```

`blockA` is declared in shared memory

```C
__shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
```

Amount of shared memory available per block is limited. On most CUDA devices, this is 48KB (49,152 bytes)

Shared memory used = BLOCK_WIDTH * BLOCK_WIDTH * sizeof(float) = BLOCK_WIDTH^2 * 4

Therefore, BLOCK_WIDTH should be <= sqrt(49152 / 4) ~= 110.5

b. If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_SIZE values.

It might be because `baseIdx` can exceeds the bounds of the input array `A_elements`.

11. Consider the following CUDA kernel and the corresponding host function that calls it:

```C
__global__ void foo_kernel(float* a, float* b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x[4];

    __shared__ float y_s;
    __shared__ float b_s[128];

    for (unsigned int j = 0; j < 4; ++j) {
        x[j] = a[j * blockDim.x * gridDim.x + i];
    }
    if (threadIdx.x == 0) {
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();

    b[i] = 2.5f * x[0] + 3.7f * x[1] + 6.3f * x[2] + 8.5f * x[3]
           + y_s * b_s[threadIdx.x] + b_s[(threadIdx.x + 3) % 128];
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d);
}
```

a. How many versions of the variable i are there? 1024
b. How many versions of the array x[] are there? 1024
c. How many versions of the variable y_s are there? 8
d. How many versions of the array b_s[] are there? 8
e. What is the amount of shared memory used per block (in bytes)? 128 + 1 floats = 129 * 4 bytes = 516 bytes
f. What is the floating-point to global memory access ratio of the kernel (in OP/B)? 

Analyzing Floating-Point Operations (OP)

```C
y_s = 7.4f;
```

This is one float point operation

```C
b[i] = 2.5f * x[0] + 3.7f * x[1] + 6.3f * x[2] + 8.5f * x[3] + y_s * b_s[threadIdx.x] + b_s[(threadIdx.x + 3) % 128];
```

This involves 4 multiplications + 5 additions + 1 multiplication = 10 operations

Analyzing Global Memory Accesses (B)

```C
for (unsigned int j = 0; j < 4; ++j) {
    x[j] = a[j * blockDim.x * gridDim.x + i];
}
```

This results in 4 global memory reads.

```C
b_s[threadIdx.x] = b[i];
```

This results in 1 global memory access.

```C
b[i] = 2.5f * x[0] + 3.7f * x[1] + 6.3f * x[2] + 8.5f * x[3] + y_s * b_s[threadIdx.x] + b_s[(threadIdx.x + 3) % 128];
```

This result in 1 global memory write.

In total we have 6 global memory accesses per thread.

Floating point to global memory access: 11 / 6 * 4 = 24 ~= 0.458 OP/B

12. Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/SM.
b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/SM.

a. Can achieve full occupancy with no limiting factors.
b. Can achieve full thread occupancy, but is limited by the number of threads per block in terms of block utilization. It uses fewer, larger blocks to reach the thread limit.
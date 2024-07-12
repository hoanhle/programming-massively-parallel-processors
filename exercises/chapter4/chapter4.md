1. Consider the following CUDA kernel and the corresponding host function that calls it:

```C
__global__ void foo_kernel(int* a, int* b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;
    }
    if (i % 2 == 0) {
        a[i] = b[i] * 2;
    }
    for (unsigned int j = 0; j < 5 - (i % 3); ++j) {
        b[i] += j;
    }
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel<<< (N + 128 - 1) / 128, 128 >>>(a_d, b_d);
}
```

a. What is the number of warps per block?

Warp size: 128, block size: 32 --> 4 warps per block

b. Number of warps in the grid?

Number of blocks: 8 --> 32 warps

c. For the statement on line 04:
i. How many warps in the grid are active? 24 (3 warps per block)
ii. How many warps in the grid are divergent? 16
iii. What is the SIMD efficiency of warp 0 of block 0? 100%
iv. What is the SIMD efficiency of warp 1 of block 0? 25%
v. What is the SIMD efficiency of warp 3 of block 0? 75%

d. For the statement on line 07?
i. How many warps in the grid are active? 32
ii. How many warps in the grid are divergent? 32
iii. What is the SIMD efficiency (in %) of warp 0 of block 0? 50%

e. For the loop on line 09:
i. How many iterations have no divergence? 0 
ii. How many iterations have divergence? 1024

2. For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

num threads in the grid = 512 * 4 = 2048

3. For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length? 1

4. Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads’ total execution time is spent waiting for the barrier?
max([2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9]) * len([2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6 2.9]) - sum ([2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9]) = 3.0 * 8 - 19.9 = 4.1s
4.1s / 24s = 17.08%

5. A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the __syncthreads() instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain. 
No.Even though 32 threads constitute a single warp, and warps execute instructions in a SIMD (Single Instruction, Multiple Data) manner, this does not guarantee that all threads will execute in lockstep in all scenarios. The absence of `__syncthreads()` can lead to unpredictable behavior if threads depend on each other’s results.

6. If a CUDA device’s SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM? 
a. 128 threads per block
b. 256 threads per block 
c. 512 threads per block 
d. 1024 threads per block

c. 512 threads per block --> 1535 - 512 * 3 = 0

7. Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level.
a. 8 blocks with 128 threads each (50% occupancy)
b. 16 blocks with 64 threads each (50% occupancy)
c. 32 blocks with 32 threads each (50% occupancy)
d. 64 blocks with 32 threads each (100% occupancy)
e. 32 blocks with 64 threads each (100% occupancy)

8. Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64K (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.
a. The kernel uses 128 threads per block and 30 registers per thread.  (100% occupancy)
b. The kernel uses 32 threads per block and 29 registers per thread. (50% occupancy, limiting factor: number of blocks)
c. The kernel uses 256 threads per block and 34 registers per thread. (94% occupancy, limiting factor: number of registers per SM)

9. Astudentmentionsthattheywereabletomultiplytwo102431024matrices using a matrix multiplication kernel with 32 3 32 thread blocks. The student is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?
- The student’s kernel configuration uses 1024 threads per block, which exceeds the hardware limit of 512 threads per block.
- To make the kernel configuration valid, the student needs to reconfigure the thread block size to meet the hardware constraints.


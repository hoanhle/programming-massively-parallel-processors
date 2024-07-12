1. If we want to use each thread to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to data index?

A. i=threadIdx.x + threadIdx.y;
B. i=blockIdx.x + threadIdx.x;
C. i=blockIdx.x*blockDim.x + threadIdx.x;
D. i=blockIdx.x * threadIdx.x;
Correct answer: C

2. Assume that we want to use each thread to calculate two (adjacent) elements of a vector addition. What would be the expression for mapping the thread/block indices to i, the data index of the first element to be processed by a thread?

A. i=blockIdx.x*blockDim.x + threadIdx.x +2;
B. i=blockIdx.xthreadIdx.x2;
C. i=(blockIdx.x*blockDim.x + threadIdx.x)*2;
D. i=blockIdx.xblockDim.x2 + threadIdx.x;
Correct answer: C

```markdown
Each thread calculates 2 elements so the number of elements processed by the previous threads are doubled.  
``` 
3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes 2*blockDim.x consecutive elements that form two sections. All threads in each block will first process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?

A. i=blockIdx.x*blockDim.x + threadIdx.x +2;
B. i=blockIdx.xthreadIdx.x2;
C. i=(blockIdx.x*blockDim.x + threadIdx.x)*2;
D. i=blockIdx.xblockDim.x2 + threadIdx.x;
Correct answer: D

4. For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel launch to have a minimal number of thread blocks to cover all output elements. How many threads will be in the grid?

A. 8000
B. 8196
C. 8192
D. 8200
Correct answer: C


```markdown
ceil(8000/1024) = 8 blocks
8 blocks * 1024 threads per blocks = 8192 threads 
``` 

5. If we want to allocate an array of v integer elements in CUDA device global memory, what would be an appropriate expression for the second argument of the cudaMalloc call?

A. n
B. v
C. n * sizeof(int)
D. v * sizeof(int)
Correct answer: D

6. If we want to allocate an array of n floating-point elements and have a floating-point pointer variable d_A to point to the allocated memory, what would be an appropriate expression for the first argument of the cudaMalloc() call?

A. n
B. (void *) d_A
C. *d_A
D. (void **) &d_A
Correct answer: D

```markdown
cudaMalloc needs a generic pointer.
```

7. If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer to element 0 of the source array) to device array A_d (A_d is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?
A. cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);
B. cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceTHost);
C. cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);
D. cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);
Correct answer: C

8. How would one declare a variable err that can appropriately receive returned value of a CUDA API call?

A. int err;
B. cudaError err;
C. cudaError_t err;
D. cudaSuccess_t err;
Correct answer: C

9. Consider the following CUDA kernel and the corresponding host function that calls it:
```markdown
__global__ void foo_kernel(float* a, float* b, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        b[i] = 2.7f * a[i] - 4.3f;
    }
}

void foo(float* a_d, float* b_d) {
    unsigned int N = 200000;
    foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
}
```

a. What is the number of threads per block? 128
b. What is the number of threads in the grid? 200064
c. What is the number of block in the gird? 1563
d. What is the number of threads that execute the code on line 02? 200064
e. What is the number of threads that execute the code on line 04? 200000

10. A new summer intern was frustrated with CUDA. He has been complaining that CUDA is very tedious. He had to declare many functions that he plans to execute on both the host and the device twice, once as a host function and once as a device function. What is your response?

```markdown
You can declare both functions at the same time using "__host__" and "__device__" before function declaration.

__host__ __device__ function_return_type function_name(...) 
{
    ...
}
```
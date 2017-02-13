// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cstdlib>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////

__global__ void mykernel()  {}

__global__ void add(int *a, int *b, int *c)  {
    *c = *a + *b;
}

__global__ void blockAdd(int *a, int *b, int *c)  {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void threadAdd(int *a, int *b, int *c)  {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

__global__ void threadBlockAdd(int *a, int *b, int *c, int n)  {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////

#define N 2048*2048
#define THREADS_PER_BLOCK 512 

void runTest(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    int *a, *b, *c;
    int *da, *db, *dc;
    int size = N * sizeof(int);

    cudaMalloc((void **)&da, size);
    cudaMalloc((void **)&db, size);
    cudaMalloc((void **)&dc, size);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    srand(1234);
    for (int i=0; i<N; i++)  {
        a[i] = rand() % 3;
	b[i] = rand() % 3;
    }

    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

//    blockAdd<<<N,1>>>(da, db, dc);
//    threadAdd<<<1,N>>>(da, db, dc);
    threadBlockAdd<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(da, db, dc, N);

    cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<N; i++)  
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << "\n"; 

    free(a); free(b); free(c);
    cudaFree(da); cudaFree(db); cudaFree(dc);
}

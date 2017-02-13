// pointer to void

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

// includes CUDA
#include <cuda_runtime.h>

// random number support
#include <curand.h>
#include <curand_kernel.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

// prototypes
void runTest(int argc, char **argv);

extern "C"
void computeGold(double *reference, double *idata, const unsigned int len);

// yeah from http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html
#define fuknMAX 100

/* this GPU kernel function calculates a random number and stores it in the parameter */
__global__ void random(int* result) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t state;

  /* we have to initialize the state */
  curand_init(0, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  /* curand works like rand - except that it takes a state as a parameter */
  *result = curand(&state) % fuknMAX;
}

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel(double *g_idata, double *g_odata)
{
    // shared memory
    // the size is determined by the host application
    extern  __shared__  double sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // perform some computations
    sdata[tid] = (double) num_threads * sdata[tid];
    __syncthreads();

    // write data to global memory
    g_odata[tid] = sdata[tid];
}


///////////////////////// GET SOME ///////////////////////

__global__ void monte()  {}

__global__ void saxpy(int n, double a, double *x, double *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
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

void runTest(int argc, char **argv)  {
    printf("%s Starting...\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);
    std::cout << "devID = " << devID << "\n"; 

    /* allocate an int on the GPU */
    int* gpu_x;
    cudaMalloc((void**) &gpu_x, sizeof(int));

    /* invoke the GPU to initialize all of the random states */
    random<<<1, 1>>>(gpu_x);
  
    /* copy the random number back */
    int x;
    cudaMemcpy(&x, gpu_x, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Random number = %d.\n", x);

    /* free the memory we allocated */
    cudaFree(gpu_x);

//    int N = 1<<20;   // leftwise bitshift same as N = 2**20 = 1048576
//    double *x, *y, *d_x, *d_y;
//    x = (double*)malloc(N*sizeof(double));
//    y = (double*)malloc(N*sizeof(double));

//    cudaMalloc(&d_x, N*sizeof(double)); 
//    cudaMalloc(&d_y, N*sizeof(double));



    // Perform SAXPY on 1M elements
//    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);




//    cudaFree(d_x);
//    cudaFree(d_y);
//    free(x);
//    free(y);
}

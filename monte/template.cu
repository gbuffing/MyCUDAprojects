// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cstdlib>
#include <time.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <curand.h>
#include <curand_kernel.h>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void pi(int argc, char **argv);


#define RANDOM_MAX 100
 
__global__ void random(int *result)  {
    curandState_t state;

    curand_init(547, 
                745+threadIdx.x, /*sequence number is only important with multiple cores */
                1, /*offset is how much extra we advance in the sequence for each call, can be 0 */
                &state);

    /* curand works like rand - except that it takes a state as a parameter */

    int tmp = curand(&state) % RANDOM_MAX;
/*    result[0] = tmp;

    tmp = curand(&state) % RANDOM_MAX;
    result[1] = tmp;  */

    result[threadIdx.x] = tmp;
}

__global__ void roll(int n, int *hits, int *throws)  {
//    int *tmp;
//    random(tmp);

}

/*
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
*/
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    pi(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////

#define N 2048*2048
#define THREADS_PER_BLOCK 512 

void pi(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);


    int n = 1024;
    int size = n * sizeof(int);
    int *gpu_x;
    cudaMalloc((void **) &gpu_x, size);

    random<<<1,n>>>(gpu_x);

//    int x;
    int *x;
    x = (int *)malloc(size);
//    cudaMemcpy(&x, gpu_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(x, gpu_x, size, cudaMemcpyDeviceToHost);

//    std::cout << n*sizeof(int) << "  " << sizeof(*x) << "\n";

//    printf("%d\n", x);
    for (int i=0; i<n; i++)
        printf("%d\n", x[i]);

    cudaFree(gpu_x);

}

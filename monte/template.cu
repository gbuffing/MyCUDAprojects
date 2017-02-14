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

// declaration, forward
void pi(int argc, char **argv);

/********************** DEVICE CODE *********************/
// some hints here
// http://stackoverflow.com/questions/11832202/cuda-random-number-generating
//
// this is pretty good...look at multi core implementation at bottom
// http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html

__global__ void random(float *result)  {

    curandState_t state;

    curand_init(1234ULL, threadIdx.x, 0, &state);
// arg1 -> the seed controls the sequence of random values that are produced
// arg2 -> sequence number is only important with multiple cores
// arg3 -> offset, how much extra we advance in the sequence for each call, can be 0 
// curand works like rand - except that it takes a state as a parameter 
    
    float tmp = curand_uniform(&state);
    result[threadIdx.x] = tmp;
}

/*
__global__ void threadBlockAdd(int *a, int *b, int *c, int n)  {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}
*/

/********************** HOST CODE *********************/

void pi(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);


    int n = 1024;
    float size = n * sizeof(float);
    float *gpu_x;
    cudaMalloc((void **) &gpu_x, size);

    random<<<1,n>>>(gpu_x);

    float *x;
    x = (float *)malloc(size);
    cudaMemcpy(x, gpu_x, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<n; i++)
        std::cout << x[i] << "\n";

    cudaFree(gpu_x);

}


// Program main
int main(int argc, char **argv) {
    pi(argc, argv);
}


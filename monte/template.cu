// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cstdlib>
#include <time.h>

// includes CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <curand.h>
#include <curand_kernel.h>

// declaration, forward
void pi(int argc, char **argv);

/******************************** DEVICE CODE ********************************/

__global__ void init_random(unsigned int seed, curandState_t *states)  {
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void randoms(curandState_t *states, double *numbers)  {
    numbers[blockIdx.x] = curand_uniform_double(&states[blockIdx.x]);
}

/********************************* HOST CODE *********************************/

void pi(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    int n = 1024;
    curandState_t *states;
    cudaMalloc((void **) &states, n * sizeof(curandState_t));
    init_random<<<n,1>>>(time(0), states);

    double *host_nums;
    host_nums = new double [n];
    double *device_nums;
    int size = n * sizeof(double);
    cudaMalloc((void **) &device_nums, size);

    randoms<<<n,1>>>(states, device_nums);

    cudaMemcpy(host_nums, device_nums, n * sizeof(double), cudaMemcpyDeviceToHost);

    double total = 0;
    for (int i=0; i<n; i++)  {
        std::cout << i+1 << "    " << host_nums[i] << "\n";
        total += host_nums[i];
    }
    std::cout << "\naverage = " << total / double(n) << "\n";

    delete[] host_nums;
    cudaFree(states);
    cudaFree(device_nums);
}

// Program main
int main(int argc, char **argv) {
    pi(argc, argv);
}

/************************* stuff ***************************/
/*
__global__ void threadBlockAdd(int *a, int *b, int *c, int n)  {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}
*/

/*
 * __global__ void random(double *result)  {

    curandState_t state;  // need a local state for each thread, see cs.umw.edu

    curand_init(1234ULL, threadIdx.x, 0, &state);
// arg1 -> the seed controls the sequence of random values that are produced
// arg2 -> sequence number is only important with multiple cores
// arg3 -> offset, how much extra we advance in the sequence for each call, can be 0
// curand works like rand - except that it takes a state as a parameter

    double tmp = curand_uniform_double(&state);
    result[threadIdx.x] = tmp;
}
*/

// some hints here
// http://stackoverflow.com/questions/11832202/cuda-random-number-generating
//
// this is pretty good...look at multi core implementation at bottom
// http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html
//
// good article
// http://stackoverflow.com/questions/26650391/generate-random-number-within-a-function-with-curand-without-preallocation




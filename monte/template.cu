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

__global__ void init_monte(int *throws, int *hits)  {
    throws[blockIdx.x] = hits[blockIdx.x] = 0;
}

__global__ void monte(curandState_t *states, int *throws, int *hits)  {
	double x, y;
	x = curand_uniform_double(&states[blockIdx.x]);
	y = curand_uniform_double(&states[blockIdx.x]);
	throws[blockIdx.x]++;
	if (sqrt(x*x + y*y) <= 1.)  {
		hits[blockIdx.x]++;
	}
}

__global__ void monte2(curandState_t *states, int *throws, int *hits, int trials)  {
	double x, y;

	for (int i=0; i<trials; i++)  {
		x = curand_uniform_double(&states[blockIdx.x]);
		y = curand_uniform_double(&states[blockIdx.x]);
		throws[blockIdx.x]++;
		if (sqrt(x*x + y*y) <= 1.)  {
			hits[blockIdx.x]++;
		}
	}

}

/********************************* HOST CODE *********************************/

void pi(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    int n = 8*1024;
    curandState_t *states;
    cudaMalloc((void **) &states, n * sizeof(curandState_t));

    int *hits = new int [n];
    int *throws = new int [n];
    int size = n * sizeof(int);
    int *device_hits;
    cudaMalloc((void **) &device_hits, size);
    int *device_throws;
    cudaMalloc((void **) &device_throws, size);

    init_random<<<n,1>>>(time(0), states);
    init_monte<<<n,1>>>(device_throws, device_hits);
//    monte<<<n,1>>>(states, device_throws, device_hits);
    monte2<<<n,1>>>(states, device_throws, device_hits, 1024);

    cudaMemcpy(hits, device_hits, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(throws, device_throws, size, cudaMemcpyDeviceToHost);

    int total_hits = 0;
    int total_throws = 0;
    for (int i=0; i<n; i++)  {
    	total_hits += hits[i];
    	total_throws += throws[i];
    }

    double pie = 4. * double(total_hits) / double(total_throws);
    std::cout << pie << "    " << total_throws << "\n";

    delete[] hits;
    delete[] throws;
    cudaFree(states);
    cudaFree(device_hits);
    cudaFree(device_throws);
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




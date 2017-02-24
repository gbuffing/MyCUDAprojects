// includes, system
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <time.h>

// includes CUDA
#include <helper_cuda.h>  //needed for findCudaDevice()
#include <curand.h>
#include <curand_kernel.h>

/******************************** DEVICE CODE ********************************/

__global__ void init_random_blocks(int seed, curandState_t *state)  {
    curand_init(seed, blockIdx.x, 0, &state[blockIdx.x]);
}

__global__ void monteBlocks(curandState_t *states, int *throws, int *hits)  {
	double x, y;
	x = curand_uniform_double(&states[blockIdx.x]);
	y = curand_uniform_double(&states[blockIdx.x]);
	throws[blockIdx.x]++;
	if (sqrt(x*x + y*y) <= 1.)  {
		hits[blockIdx.x]++;
	}
}

__global__ void init_random_threads(int seed, curandState_t *state)  {
    curand_init(seed, threadIdx.x, 0, &state[threadIdx.x]);
}

__global__ void monteThreads(curandState_t *states, int *throws, int *hits)  {
	double x, y;
	x = curand_uniform_double(&states[threadIdx.x]);
	y = curand_uniform_double(&states[threadIdx.x]);
	throws[threadIdx.x]++;
	if (sqrt(x*x + y*y) <= 1.)  {
		hits[threadIdx.x]++;
	}
}

/********************************* HOST CODE *********************************/

void pi(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    int n = 256 * 1024;
    curandState_t *state;
    int state_size = n * sizeof(curandState_t);
    cudaMallocManaged(&state, state_size);

    unsigned int t = time(0);
    //t = 1234;
    init_random_blocks<<<n,1>>>(t, state);
//    init_random_threads<<<1,n>>>(t, state);
    cudaDeviceSynchronize();

    int size = n * sizeof(int);
    int *hits;
    cudaMallocManaged(&hits, size);
    int *throws;
    cudaMallocManaged(&throws, size);

    *hits = *throws = 0;

    monteBlocks<<<n,1>>>(state, throws, hits);
//      monteThreads<<<1,n>>>(state, throws, hits);

    cudaDeviceSynchronize();

    int total_hits = 0;
    int total_throws = 0;
    for (int i=0; i<n; i++)  {
    	total_hits += hits[i];
    	total_throws += throws[i];
    }

    double pie = 4. * double(total_hits) / double(total_throws);
    std::cout << pie << "    " << total_throws << "\n";

    cudaFree(state);
    cudaFree(hits);
    cudaFree(throws);
}

// Program main
int main(int argc, char **argv) {
    pi(argc, argv);
}

// some hints here
// http://stackoverflow.com/questions/11832202/cuda-random-number-generating
//
// this is pretty good...look at multi core implementation at bottom
// http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html
//
// good article
// http://stackoverflow.com/questions/26650391/generate-random-number-within-a-function-with-curand-without-preallocation




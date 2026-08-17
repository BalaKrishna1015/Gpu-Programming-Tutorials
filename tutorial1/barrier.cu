#include <stdio.h>
#include <cuda_runtime.h>

__device__  int count = 0;

__global__ void barrier()
{
    int tid = threadIdx.x;
    printf("Thread %d reached barrier\n", tid);
    atomicAdd(&count, 1);
    while (count < blockDim.x) ;

    __syncthreads();
    if (tid == 0)printf("All threads reached the barrier\n");
    printf("Thread %d passed barrier\n", tid);
}

int main()
{
    int threads = 8;
    barrier<<<1, threads>>>();
    cudaDeviceSynchronize();
    return 0;
}
#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
/**
 * This example illustrates implementation of custom atomic operations using
 * CUDA's built-in atomicCAS function to implement atomic signed 32-bit integer
 * addition.
 **/

__device__ int sqrt1(int *address)
{
    // Create an initial guess for the value stored at *address.
    int guess = *address;
    int nValue = (int)sqrtf(guess)+guess;
    int oldValue = atomicCAS(address, guess,nValue);

    // Loop while the guess is incorrect.
    while (oldValue != guess)
    {
        guess = oldValue;
        int nValue = (int)sqrtf(guess)+guess;
        oldValue = atomicCAS(address, guess, nValue);
    }

    return oldValue;
}

__device__ int sqrt2(int *address)
{
    // Create an initial guess for the value stored at *address.
    int guess = *address;
    int nValue =__fsqrt_rn(guess)+guess;
    int oldValue = atomicCAS(address, guess,nValue);

    // Loop while the guess is incorrect.
    while (oldValue != guess)
    {
        guess = oldValue;
        int nValue = __fsqrt_rn(guess)+guess;
        oldValue = atomicCAS(address, guess, nValue);
    } 

    return oldValue;
}

__global__ void kernel1(int *sharedInteger)
{
    sqrt1(sharedInteger);
}

__global__ void kernel2(int *sharedInteger)
{
   sqrt2(sharedInteger);
}





int main(int argc, char **argv)
{
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    int * value;
    int N = 25;

    if (argc > 1) N = atoi(argv[1]);

    CHECK(cudaMallocManaged((void **)&value, sizeof(int)));
    *value = N;
    CHECK(cudaEventRecord(start, 0));
    kernel1<<<1,1>>>(value);
    CHECK(cudaDeviceSynchronize());
   
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
 
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for kernel 1 execution = %f\n",
            elapsed_time / 1000.0f);

    printf("OLA sqrtf %d\n", *value);
    *value = N;
 
    CHECK(cudaEventRecord(start, 0));
    kernel2<<<1,1>>>(value);
    CHECK(cudaDeviceSynchronize());
   
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
 
    
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for kernel 2 execution = %f\n",
            elapsed_time / 1000.0f);

    printf("OLA __fsqrt_rn %d\n", *value);
    CHECK(cudaFree(value));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return 0;
}

/* 

Measured time for kernel 1 execution = 0.000078
OLA sqrtf 30
Measured time for kernel 2 execution = 0.000124
OLA __fsqrt_rn 30


*/

#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "../common/stopwatch.h"
#define LOG 0



void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i = 0; i < N; i++) {
		if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match)
		printf("Arrays match.\n\n");
}
void initialData(float *ip, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++) {
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}
void switchOnCPU(float *A, float *B, const int N)
{
	for (int i = 0; i < N; i++)
		B[N - i-1] = A[i];
}

__global__ void switchOnGPU(float *A, float *B, const int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) B[N-i-1] = A[i];
}

void print(float* A,float*B, const int N)
{
	printf("\n\n");
	for (int i = 0; i < N; i++)
		printf("%f ", A[i]);
	printf("\n");
	printf("\n");
	for (int i = 0; i < N; i++)
		printf("%f ", B[i]);
	printf("\n");
}

int main(int argc, char **argv)
{
	printf("%s Starting...\n", argv[0]);

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
	

	int power = 18;
	int nthreads = 512;   

	if (argc > 1)
	{
		nthreads = atoi(argv[1]); 
	}

	if (argc > 2)
	{
		power = atoi(argv[2]);  
	}

	int nElem = 1 << power;
	printf("Vector size %d\n", nElem);
	// alociranje host memorije
	size_t nBytes = nElem * sizeof(float);
	float *h_A, *h_B,*h_C;
	
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	h_C = (float *)malloc(nBytes);
	
	initialData(h_A, nElem);
	/*for (int i = 0; i < nElem; i++)
        h_A[i] =(float) i + 1; */
        
    dim3 block(nthreads);
    dim3 grid((nElem + block.x - 1) / block.x);
	Stopwatch s;
	switchOnCPU(h_A, h_B, nElem);
	
	printf("sumArraysOnCPU - Time elapsed %f: ",s.elapsed());
	// alociranje device globalne memorije
	//print(h_A, h_B, nElem);
	float *d_A, *d_B;
	cudaMalloc((float**)&d_A, nBytes);
	cudaMalloc((float**)&d_B, nBytes);
	
	// transfer podataka sa host-a na device
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	
	
	s.reset();
	switchOnGPU <<<grid, block>>> (d_A, d_B, nElem);
	cudaDeviceSynchronize();
	
	printf("\nsumArraysOnGPU <<<%d,%d>>> Time elapsed %f" \
		"sec\n", grid.x, block.x, s.elapsed());
	
	// kopiranje rezultata kernela nazad na host
	cudaMemcpy(h_C, d_B, nBytes, cudaMemcpyDeviceToHost);
	// provera device rezultata
	checkResult(h_B, h_C, nElem);
	// oslobadanje globalne memorije device-a
	cudaFree(d_A);
	cudaFree(d_B);

	// oslobadanje host memorije
	free(h_A);
	free(h_B);
	free(h_C);
	cudaDeviceReset();
	return(0);
}

/*
GeForce GTX 1050


Vector size 262144
sumArraysOnCPU - Time elapsed 0.001133:
sumArraysOnGPU <<<8192,32>>> Time elapsed 0.000464sec
Arrays match.

==8520== Profiling application: a 32
==8520== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   42.49%  157.92us         1  157.92us  157.92us  157.92us  [CUDA memcpy HtoD]
                   42.16%  156.70us         1  156.70us  156.70us  156.70us  [CUDA memcpy DtoH]
                   15.34%  57.023us         1  57.023us  57.023us  57.023us  switchOnGPU(float*, float*, int)
      API calls:   72.82%  208.47ms         2  104.23ms  12.400us  208.46ms  cudaMalloc
                   25.98%  74.386ms         1  74.386ms  74.386ms  74.386ms  cudaDeviceReset
                    0.43%  1.2370ms         2  618.50us  244.70us  992.30us  cudaMemcpy
                    0.24%  687.90us        97  7.0910us     200ns  314.60us  cuDeviceGetAttribute
                    0.21%  590.30us         1  590.30us  590.30us  590.30us  cudaGetDeviceProperties
                    0.14%  400.40us         1  400.40us  400.40us  400.40us  cudaDeviceSynchronize
                    0.13%  359.10us         2  179.55us  90.100us  269.00us  cudaFree
                    0.02%  59.000us         1  59.000us  59.000us  59.000us  cudaLaunchKernel
                    0.01%  37.700us         1  37.700us  37.700us  37.700us  cuDeviceTotalMem
                    0.00%  13.800us         1  13.800us  13.800us  13.800us  cuDeviceGetPCIBusId
                    0.00%  13.000us         1  13.000us  13.000us  13.000us  cudaSetDevice
                    0.00%  7.4000us         2  3.7000us     400ns  7.0000us  cuDeviceGet
                    0.00%  2.2000us         3     733ns     300ns  1.1000us  cuDeviceGetCount
                    0.00%  1.4000us         1  1.4000us  1.4000us  1.4000us  cuDeviceGetName
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetLuid
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetUuid






                    Vector size 262144
sumArraysOnCPU - Time elapsed 0.001133:
sumArraysOnGPU <<<8192,32>>> Time elapsed 0.000464sec
Arrays match.

==8520== Profiling application: a 32
==8520== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   42.49%  157.92us         1  157.92us  157.92us  157.92us  [CUDA memcpy HtoD]
                   42.16%  156.70us         1  156.70us  156.70us  156.70us  [CUDA memcpy DtoH]
                   15.34%  57.023us         1  57.023us  57.023us  57.023us  switchOnGPU(float*, float*, int)
      API calls:   72.82%  208.47ms         2  104.23ms  12.400us  208.46ms  cudaMalloc
                   25.98%  74.386ms         1  74.386ms  74.386ms  74.386ms  cudaDeviceReset
                    0.43%  1.2370ms         2  618.50us  244.70us  992.30us  cudaMemcpy
                    0.24%  687.90us        97  7.0910us     200ns  314.60us  cuDeviceGetAttribute
                    0.21%  590.30us         1  590.30us  590.30us  590.30us  cudaGetDeviceProperties
                    0.14%  400.40us         1  400.40us  400.40us  400.40us  cudaDeviceSynchronize
                    0.13%  359.10us         2  179.55us  90.100us  269.00us  cudaFree
                    0.02%  59.000us         1  59.000us  59.000us  59.000us  cudaLaunchKernel
                    0.01%  37.700us         1  37.700us  37.700us  37.700us  cuDeviceTotalMem
                    0.00%  13.800us         1  13.800us  13.800us  13.800us  cuDeviceGetPCIBusId
                    0.00%  13.000us         1  13.000us  13.000us  13.000us  cudaSetDevice
                    0.00%  7.4000us         2  3.7000us     400ns  7.0000us  cuDeviceGet
                    0.00%  2.2000us         3     733ns     300ns  1.1000us  cuDeviceGetCount
                    0.00%  1.4000us         1  1.4000us  1.4000us  1.4000us  cuDeviceGetName
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetLuid
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetUuid



Vector size 262144
sumArraysOnCPU - Time elapsed 0.001112:
sumArraysOnGPU <<<512,512>>> Time elapsed 0.000413sec
Arrays match.

==5896== Profiling application: a 512
==5896== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   46.63%  157.70us         1  157.70us  157.70us  157.70us  [CUDA memcpy HtoD]
                   46.46%  157.12us         1  157.12us  157.12us  157.12us  [CUDA memcpy DtoH]
                    6.91%  23.360us         1  23.360us  23.360us  23.360us  switchOnGPU(float*, float*, int)
      API calls:   73.60%  209.67ms         2  104.83ms  12.300us  209.65ms  cudaMalloc
                   25.22%  71.841ms         1  71.841ms  71.841ms  71.841ms  cudaDeviceReset
                    0.42%  1.2078ms         2  603.90us  243.60us  964.20us  cudaMemcpy
                    0.24%  693.50us        97  7.1490us     200ns  315.50us  cuDeviceGetAttribute
                    0.21%  595.20us         1  595.20us  595.20us  595.20us  cudaGetDeviceProperties
                    0.12%  346.60us         1  346.60us  346.60us  346.60us  cudaDeviceSynchronize
                    0.12%  334.90us         2  167.45us  87.400us  247.50us  cudaFree
                    0.02%  69.800us         1  69.800us  69.800us  69.800us  cuDeviceTotalMem
                    0.02%  61.600us         1  61.600us  61.600us  61.600us  cudaLaunchKernel
                    0.00%  13.500us         1  13.500us  13.500us  13.500us  cuDeviceGetPCIBusId
                    0.00%  12.700us         1  12.700us  12.700us  12.700us  cudaSetDevice
                    0.00%  7.1000us         2  3.5500us     400ns  6.7000us  cuDeviceGet
                    0.00%  2.6000us         3     866ns     400ns  1.2000us  cuDeviceGetCount
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuDeviceGetName
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetLuid
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetUuid



Vector size 262144
sumArraysOnCPU - Time elapsed 0.001207:
sumArraysOnGPU <<<256,1024>>> Time elapsed 0.000417sec
Arrays match.

==7556== Profiling application: a 1024
==7556== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   46.36%  157.47us         1  157.47us  157.47us  157.47us  [CUDA memcpy HtoD]
                   46.26%  157.12us         1  157.12us  157.12us  157.12us  [CUDA memcpy DtoH]
                    7.38%  25.056us         1  25.056us  25.056us  25.056us  switchOnGPU(float*, float*, int)
      API calls:   72.98%  211.48ms         2  105.74ms  14.800us  211.46ms  cudaMalloc
                   25.85%  74.896ms         1  74.896ms  74.896ms  74.896ms  cudaDeviceReset
                    0.44%  1.2658ms         2  632.90us  255.80us  1.0100ms  cudaMemcpy
                    0.24%  686.70us        97  7.0790us     200ns  312.90us  cuDeviceGetAttribute
                    0.21%  614.60us         1  614.60us  614.60us  614.60us  cudaGetDeviceProperties
                    0.12%  350.90us         1  350.90us  350.90us  350.90us  cudaDeviceSynchronize
                    0.12%  341.80us         2  170.90us  81.700us  260.10us  cudaFree
                    0.02%  62.000us         1  62.000us  62.000us  62.000us  cudaLaunchKernel
                    0.01%  41.000us         1  41.000us  41.000us  41.000us  cuDeviceTotalMem
                    0.00%  13.600us         1  13.600us  13.600us  13.600us  cuDeviceGetPCIBusId
                    0.00%  13.100us         1  13.100us  13.100us  13.100us  cudaSetDevice
                    0.00%  6.9000us         2  3.4500us     400ns  6.5000us  cuDeviceGet
                    0.00%  2.2000us         3     733ns     300ns  1.2000us  cuDeviceGetCount
                    0.00%  1.3000us         1  1.3000us  1.3000us  1.3000us  cuDeviceGetName
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetLuid
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetUuid


                    */
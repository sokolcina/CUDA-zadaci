#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "../common/stopwatch.h"
//#include <limit.h>
#define LOG 0




__global__ void Min(int *g_idata, int *g_odata,
	unsigned int isize)
{
	// set thread ID
	unsigned int tid = threadIdx.x;

	// convert global data pointer to the local pointer of this block
	int *idata = g_idata + blockIdx.x * blockDim.x;
	int *odata = &g_odata[blockIdx.x];

	// stop condition
	if (isize == 2 && tid == 0)
	{
		if (idata[0] < idata[1])
			g_odata[blockIdx.x] = idata[0];
			else
			g_odata[blockIdx.x] = idata[1];
		return;
	}

	// nested invoke
	int istride = isize >> 1;

	if (istride > 1 && tid < istride)
	{
		if (idata[tid + istride] < idata[tid])
			idata[tid] = idata[tid + istride];

		if (tid == 0)
		{
			Min <<<1, istride >>> (idata, odata, istride);
		}
	}

}

__global__ void Max(int *g_idata, int *g_odata,
	unsigned int isize)
{
	// set thread ID
	unsigned int tid = threadIdx.x;

	// convert global data pointer to the local pointer of this block
	int *idata = g_idata + blockIdx.x * blockDim.x;
	int *odata = &g_odata[blockIdx.x];

	// stop condition
	if (isize == 2 && tid == 0)
	{
		if (idata[0] > idata[1])
			g_odata[blockIdx.x] = idata[0];
			else
			g_odata[blockIdx.x] = idata[1];
		return;
	}

	// nested invoke
	int istride = isize >> 1;

	if (istride > 1 && tid < istride)
	{
		if (idata[tid + istride] > idata[tid])
			idata[tid] = idata[tid + istride];

		if (tid == 0)
		{
			Max <<<1, istride >>> (idata, odata, istride);
		}
	}
	
}
__global__ void maxMin(int *g_idata1, int*g_idata2, int *g_odata1, int* g_odata2,unsigned int isize)
{
	// set thread ID
	unsigned int tid = threadIdx.x;

	// convert global data pointer to the local pointer of this block
	int *idata1 = g_idata1 + blockIdx.x * blockDim.x;
	int *idata2 = g_idata2 + blockIdx.x * blockDim.x;

	int *odata1 = &g_odata1[blockIdx.x];
	int *odata2 = &g_odata2[blockIdx.x];
	
	if (isize == 2 && tid == 0)
	{
		if (idata1[0] < idata1[1])
			g_odata1[blockIdx.x] = idata1[0];
			else
			g_odata1[blockIdx.x] = idata1[1];
		if (idata2[0] > idata2[1])
			g_odata2[blockIdx.x] = idata2[0];
			else
			g_odata2[blockIdx.x] = idata2[1];
		return;
	}
	int istride = isize >> 1;

	if (istride > 1 && tid < istride)
	{
		if (idata1[tid] > idata1[tid + istride])
			idata1[tid] = idata1[tid + istride];
		if (idata2[tid] < idata2[tid + istride])
			idata2[tid] = idata2[tid + istride];

		if(tid==0)
		{
			maxMin<<<1,istride>>>(idata1,idata2,odata1,odata2,istride);
		}
	}



}

__global__ void maxMin1(int *g_idata1, int*g_idata2, int *g_odata1, int* g_odata2,unsigned int isize,dim3 grid)
{
	// set thread ID
	unsigned int tid = threadIdx.x;

	// convert global data pointer to the local pointer of this block
	int *idata1 = g_idata1 + blockIdx.x * blockDim.x;
	int *idata2 = g_idata2 + blockIdx.x * blockDim.x;

	int *odata1 = &g_odata1[blockIdx.x];
	int *odata2 = &g_odata2[blockIdx.x];
	int istride = isize >> 1;


	if (tid < istride)
	{
		if (idata1[tid] > idata1[tid + istride])
			idata1[tid] = idata1[tid + istride];
		if (idata2[tid] < idata2[tid + istride])
			idata2[tid] = idata2[tid + istride];
	}
	//__syncthreads();
	//istride = istride >> 1;
	__threadfence_block(); 
	//__threadfence();
	if (blockIdx.x == 0 && threadIdx.x==0)
	{
		Min <<<grid, blockDim.x/2>>> (idata1, odata1, istride);
		Max <<<grid, blockDim.x/2>>> (idata2, odata2, istride);
		//cudaDeviceSynchronize();
	}
	
	


}

void cpuMaxMin(int *data, int size)
{
	int max = data[0];
	int min = data[0];
	for (int i = 0; i < size; i++)
	{
		if (data[i] < min)
			min = data[i];
		else
			if (data[i] > max)
				max = data[i];
	}
	data[0] = min;
	data[1] = max;
}




// main from here
int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting reduction at ", argv[0]);
	printf("device %d: %s ", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));



	// set up execution configuration
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

	int size = 1<<power; // total number of elements to reduceNeighbored

	dim3 block(nthreads, 1);
	dim3 grid((size + block.x - 1) / block.x, 1);
	printf("Array %d grid %d block %d\n", size, grid.x, block.x);

	// allocate host memory
	size_t bytes = size * sizeof(int);
	int *h_idata = (int *)malloc(bytes);
	int *h_odata1 = (int *)malloc(grid.x*sizeof(int));
	int *h_odata2 = (int *)malloc(grid.x*sizeof(int));
	// initialize the array
	
	srand(10);
	for (int i = 0; i < size; i++)
	{
		h_idata[i] = (int)(rand()%bytes)+31;
		//h_idata[i] = i*2+1;
	}

	int *d_idata1 = NULL;
	int *d_odata1 = NULL;
	int *d_idata2 = NULL;
	int *d_odata2 = NULL;
	CHECK(cudaMalloc((void **)&d_idata1, bytes));
	CHECK(cudaMalloc((void **)&d_odata1, grid.x * sizeof(int)));
	CHECK(cudaMalloc((void **)&d_idata2, bytes));
	CHECK(cudaMalloc((void **)&d_odata2, grid.x * sizeof(int)));

	CHECK(cudaMemcpy(d_idata1, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_idata2, h_idata, bytes, cudaMemcpyHostToDevice));
	

	Stopwatch s;
	cpuMaxMin(h_idata, size);
	printf("CPU: %lf s\n",s.elapsed());
	int h_min = h_idata[0];
	int h_max = h_idata[1];


	s.reset();
	maxMin1 <<<grid, block >>> (d_idata1, d_idata2, d_odata1, d_odata2, block.x,grid);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError()); 
	printf("maxMin1<<<%d,%d>>>: %lf s\n",grid.x,block.x,s.elapsed());

	CHECK(cudaMemcpy(h_odata1, d_odata1, grid.x * sizeof(int),
		cudaMemcpyDeviceToHost));
	int d_min = h_odata1[0];
	for (int i = 1; i < grid.x; i++)
		if (h_odata1[i] < d_min) d_min = h_odata1[i];
		
		
	CHECK(cudaMemcpy(h_odata2, d_odata2, grid.x * sizeof(int),
		cudaMemcpyDeviceToHost));

		/*for (int i = 0; i < grid.x; i++)
		printf("%d \t",h_odata2[i]); */

	int d_max = h_odata2[0];
	for (int i = 1; i < grid.x; i++)
		if (h_odata2[i] > d_max) d_max = h_odata2[i];

	
		printf("\n %d \n",h_min);
		printf(" %d \n",h_max);
        printf("\n %d \n",d_min);
		printf(" %d \n",d_max);
	if (h_max == d_max && h_min == d_min)
		printf("SUPER\n");
	else printf("LOSE\n");

	

	CHECK(cudaMemcpy(d_idata1, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_idata2, h_idata, bytes, cudaMemcpyHostToDevice));


	s.reset();
	maxMin <<<grid, block >>> (d_idata1, d_idata2, d_odata1, d_odata2, block.x);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError()); 
	printf("maxMin<<<%d,%d>>>: %lf s\n",grid.x,block.x,s.elapsed());

	CHECK(cudaMemcpy(h_odata1, d_odata1, grid.x * sizeof(int),
		cudaMemcpyDeviceToHost));
	d_min = h_odata1[0];
	for (int i = 1; i < grid.x; i++)
		if (h_odata1[i] < d_min) d_min = h_odata1[i];
		
		
	CHECK(cudaMemcpy(h_odata2, d_odata2, grid.x * sizeof(int),
		cudaMemcpyDeviceToHost));

		/*for (int i = 0; i < grid.x; i++)
		printf("%d \t",h_odata2[i]); */

	d_max = h_odata2[0];
	for (int i = 1; i < grid.x; i++)
		if (h_odata2[i] > d_max) d_max = h_odata2[i];

	
	printf("\n %d \n",h_min);
		printf(" %d \n",h_max);
        printf("\n %d \n",d_min);
		printf(" %d \n",d_max);
	if (h_max == d_max && h_min == d_min)
		printf("SUPER\n");
	else printf("LOSE\n");
	
	// free host memory
	free(h_idata);
	free(h_odata1);
	free(h_odata2);
	// free device memory
	CHECK(cudaFree(d_idata1));
	CHECK(cudaFree(d_odata1));

	CHECK(cudaFree(d_idata2));
	CHECK(cudaFree(d_odata2));
	// reset device
	CHECK(cudaDeviceReset());


	return EXIT_SUCCESS;
}

/*

 GeForce GTX 1050


CPU: 0.000718 s
maxMin1<<<8192,32>>>: 0.046793 s

 31
 32798

 31
 32798
SUPER
maxMin<<<8192,32>>>: 0.036667 s

 31
 32798

 31
 32798
SUPER
==5576== Warning: 12018 records were dropped due to insufficient device buffer space. You can configure the buffer space using advanced options --device-buffer-size, --device-cdp-buffer-size
==5576== Profiling application: a 32
==5576== Profiling result:
            Type  Time(%)      Time  Calls (host)  Calls (device)       Avg       Min       Max  Name
 GPU activities:   35.75%  543.55ms             1           20750  26.194us  1.3440us  36.151ms  maxMin(int*, int*, int*, int*, unsigned int)
                   32.02%  486.95ms             0           24577  19.813us  1.2150us  22.797ms  Min(int*, int*, unsigned int)
                   31.97%  486.11ms             0           24577  19.778us  1.2160us  23.074ms  Max(int*, int*, unsigned int)
                    0.22%  3.3320ms             1               0  3.3320ms  3.3320ms  3.3320ms  maxMin1(int*, int*, int*, int*, unsigned int, dim3)
                    0.04%  651.52us             4               -  162.88us  157.66us  177.79us  [CUDA memcpy HtoD]
                    0.00%  24.064us             4               -  6.0160us  5.7920us  6.2720us  [CUDA memcpy DtoH]
      API calls:   61.06%  254.53ms             4               -  63.632ms  12.900us  254.02ms  cudaMalloc
                   19.98%  83.295ms             2               -  41.648ms  36.603ms  46.692ms  cudaDeviceSynchronize
                   17.92%  74.711ms             1               -  74.711ms  74.711ms  74.711ms  cudaDeviceReset
                    0.55%  2.2850ms             8               -  285.63us  149.20us  467.90us  cudaMemcpy
                    0.15%  643.70us            97               -  6.6360us     200ns  364.00us  cuDeviceGetAttribute
                    0.15%  618.50us             1               -  618.50us  618.50us  618.50us  cudaGetDeviceProperties
                    0.13%  561.50us             4               -  140.38us  29.100us  272.20us  cudaFree
                    0.04%  146.20us             2               -  73.100us  56.000us  90.200us  cudaLaunchKernel
                    0.01%  44.300us             1               -  44.300us  44.300us  44.300us  cuDeviceTotalMem
                    0.00%  15.700us             1               -  15.700us  15.700us  15.700us  cudaSetDevice
                    0.00%  14.200us             1               -  14.200us  14.200us  14.200us  cuDeviceGetPCIBusId
                    0.00%  7.0000us             2               -  3.5000us     400ns  6.6000us  cuDeviceGet
                    0.00%  2.6000us             3               -     866ns     300ns  1.5000us  cuDeviceGetCount
                    0.00%  1.8000us             2               -     900ns     900ns     900ns  cudaGetLastError
                    0.00%  1.6000us             1               -  1.6000us  1.6000us  1.6000us  cuDeviceGetName
                    0.00%     600ns             1               -     600ns     600ns     600ns  cuDeviceGetUuid
                    0.00%     600ns             1               -     600ns     600ns     600ns  cuDeviceGetLuid





CPU: 0.000717 s
maxMin1<<<1024,256>>>: 0.013657 s

 31
 32798

 31
 32798
SUPER
maxMin<<<1024,256>>>: 0.009427 s

 31
 32798

 31
 32798
SUPER
==7444== Profiling application: a 256
==7444== Profiling result:
            Type  Time(%)      Time  Calls (host)  Calls (device)       Avg       Min       Max  Name
 GPU activities:   41.52%  204.57ms             1            7168  28.535us  1.4080us  8.8570ms  maxMin(int*, int*, int*, int*, unsigned int)
                   29.33%  144.49ms             0            6145  23.514us  1.2480us  6.2739ms  Min(int*, int*, unsigned int)
                   28.99%  142.84ms             0            6145  23.245us  1.2480us  6.0183ms  Max(int*, int*, unsigned int)
                    0.14%  669.69us             4               -  167.42us  157.54us  196.48us  [CUDA memcpy HtoD]
                    0.02%  120.22us             1               0  120.22us  120.22us  120.22us  maxMin1(int*, int*, int*, int*, unsigned int, dim3)
                    0.00%  5.6640us             4               -  1.4160us  1.1520us  1.7280us  [CUDA memcpy DtoH]
      API calls:   72.89%  246.22ms             4               -  61.555ms  12.900us  245.58ms  cudaMalloc
                   19.17%  64.757ms             1               -  64.757ms  64.757ms  64.757ms  cudaDeviceReset
                    6.78%  22.913ms             2               -  11.457ms  9.3649ms  13.548ms  cudaDeviceSynchronize
                    0.58%  1.9599ms             8               -  244.99us  94.400us  448.90us  cudaMemcpy
                    0.20%  680.50us            97               -  7.0150us     200ns  309.40us  cuDeviceGetAttribute
                    0.18%  599.40us             1               -  599.40us  599.40us  599.40us  cudaGetDeviceProperties
                    0.14%  456.30us             4               -  114.08us  27.800us  222.10us  cudaFree
                    0.04%  151.50us             2               -  75.750us  53.900us  97.600us  cudaLaunchKernel
                    0.01%  41.500us             1               -  41.500us  41.500us  41.500us  cuDeviceTotalMem
                    0.00%  13.900us             1               -  13.900us  13.900us  13.900us  cuDeviceGetPCIBusId
                    0.00%  12.300us             1               -  12.300us  12.300us  12.300us  cudaSetDevice
                    0.00%  6.1000us             2               -  3.0500us     400ns  5.7000us  cuDeviceGet
                    0.00%  3.0000us             3               -  1.0000us     400ns  1.4000us  cuDeviceGetCount
                    0.00%  2.0000us             2               -  1.0000us     900ns  1.1000us  cudaGetLastError
                    0.00%  1.4000us             1               -  1.4000us  1.4000us  1.4000us  cuDeviceGetName
                    0.00%     700ns             1               -     700ns     700ns     700ns  cuDeviceGetLuid
                    0.00%     500ns             1               -     500ns     500ns     500ns  cuDeviceGetUuid





CPU: 0.000841 s
maxMin1<<<512,512>>>: 0.008662 s

 31
 32798

 31
 32798
SUPER
maxMin<<<512,512>>>: 0.005918 s

 31
 32798

 31
 32798
SUPER
==4928== Profiling application: a 512
==4928== Profiling result:
            Type  Time(%)      Time  Calls (host)  Calls (device)       Avg       Min       Max  Name
 GPU activities:   38.56%  110.75ms             1            4096  27.031us  1.4400us  5.0332ms  maxMin(int*, int*, int*, int*, unsigned int)
                   30.39%  87.261ms             0            3585  24.340us  1.2480us  3.7901ms  Min(int*, int*, unsigned int)
                   30.38%  87.251ms             0            3585  24.337us  1.2800us  3.6692ms  Max(int*, int*, unsigned int)
                    0.41%  1.1915ms             1               0  1.1915ms  1.1915ms  1.1915ms  maxMin1(int*, int*, int*, int*, unsigned int, dim3)
                    0.25%  715.84us             4               -  178.96us  157.89us  212.67us  [CUDA memcpy HtoD]
                    0.00%  5.8560us             4               -  1.4640us  1.2480us  1.8560us  [CUDA memcpy DtoH]
      API calls:   74.37%  239.55ms             4               -  59.887ms  15.200us  238.86ms  cudaMalloc
                   19.84%  63.920ms             1               -  63.920ms  63.920ms  63.920ms  cudaDeviceReset
                    4.45%  14.344ms             2               -  7.1721ms  5.8188ms  8.5253ms  cudaDeviceSynchronize
                    0.81%  2.6023ms             8               -  325.29us  109.20us  874.50us  cudaMemcpy
                    0.17%  534.50us             4               -  133.63us  28.900us  257.50us  cudaFree
                    0.16%  508.90us            97               -  5.2460us     200ns  248.30us  cuDeviceGetAttribute
                    0.12%  378.20us             1               -  378.20us  378.20us  378.20us  cudaGetDeviceProperties
                    0.06%  201.90us             2               -  100.95us  81.000us  120.90us  cudaLaunchKernel
                    0.01%  40.100us             1               -  40.100us  40.100us  40.100us  cuDeviceTotalMem
                    0.00%  14.100us             1               -  14.100us  14.100us  14.100us  cuDeviceGetPCIBusId
                    0.00%  12.200us             1               -  12.200us  12.200us  12.200us  cudaSetDevice
                    0.00%  6.0000us             2               -  3.0000us     400ns  5.6000us  cuDeviceGet
                    0.00%  2.3000us             3               -     766ns     400ns  1.1000us  cuDeviceGetCount
                    0.00%  1.8000us             2               -     900ns     900ns     900ns  cudaGetLastError
                    0.00%  1.5000us             1               -  1.5000us  1.5000us  1.5000us  cuDeviceGetName
                    0.00%     600ns             1               -     600ns     600ns     600ns  cuDeviceGetUuid
                    0.00%     600ns             1               -     600ns     600ns     600ns  cuDeviceGetLuid





CPU: 0.000729 s
maxMin1<<<256,1024>>>: 0.005571 s

 31
 32798

 31
 32798
SUPER
maxMin<<<256,1024>>>: 0.003473 s

 31
 32798

 31
 32798
SUPER
==2280== Profiling application: a 1024
==2280== Profiling result:
            Type  Time(%)      Time  Calls (host)  Calls (device)       Avg       Min       Max  Name
 GPU activities:   36.58%  61.023ms             1            2304  26.474us  1.3760us  2.6956ms  maxMin(int*, int*, int*, int*, unsigned int)
                   31.10%  51.867ms             0            2049  25.313us  1.2800us  2.2174ms  Min(int*, int*, unsigned int)
                   30.81%  51.386ms             0            2049  25.078us  1.2480us  2.1590ms  Max(int*, int*, unsigned int)
                    1.12%  1.8609ms             1               0  1.8609ms  1.8609ms  1.8609ms  maxMin1(int*, int*, int*, int*, unsigned int, dim3)
                    0.39%  658.01us             4               -  164.50us  157.66us  184.48us  [CUDA memcpy HtoD]
                    0.00%  5.0880us             4               -  1.2720us  1.1840us  1.4400us  [CUDA memcpy DtoH]
      API calls:   73.93%  243.69ms             4               -  60.924ms  12.700us  243.22ms  cudaMalloc
                   22.13%  72.944ms             1               -  72.944ms  72.944ms  72.944ms  cudaDeviceReset
                    2.69%  8.8829ms             2               -  4.4415ms  3.4106ms  5.4723ms  cudaDeviceSynchronize
                    0.65%  2.1400ms             8               -  267.50us  143.40us  432.00us  cudaMemcpy
                    0.21%  687.50us            97               -  7.0870us     200ns  317.60us  cuDeviceGetAttribute
                    0.18%  597.20us             1               -  597.20us  597.20us  597.20us  cudaGetDeviceProperties
                    0.15%  478.70us             4               -  119.68us  28.300us  223.50us  cudaFree
                    0.04%  141.30us             2               -  70.650us  53.700us  87.600us  cudaLaunchKernel
                    0.01%  42.500us             1               -  42.500us  42.500us  42.500us  cuDeviceTotalMem
                    0.00%  13.600us             1               -  13.600us  13.600us  13.600us  cuDeviceGetPCIBusId
                    0.00%  12.500us             1               -  12.500us  12.500us  12.500us  cudaSetDevice
                    0.00%  6.1000us             2               -  3.0500us     500ns  5.6000us  cuDeviceGet
                    0.00%  2.5000us             3               -     833ns     400ns  1.1000us  cuDeviceGetCount
                    0.00%  1.6000us             1               -  1.6000us  1.6000us  1.6000us  cuDeviceGetName
                    0.00%  1.6000us             2               -     800ns     800ns     800ns  cudaGetLastError
                    0.00%     800ns             1               -     800ns     800ns     800ns  cuDeviceGetLuid
                    0.00%     500ns             1               -     500ns     500ns     500ns  cuDeviceGetUuid
					*/
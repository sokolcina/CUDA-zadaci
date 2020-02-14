#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "../common/stopwatch.h"

#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))

#define BDIMX 16
#define BDIMY BDIMX


void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)(rand() & 0xFF) / 100.0f;
    }

    return;
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-3;
    bool match = 1;
   
	for (int i = 0; i < N; i++) {
		if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.8f gpu %5.8f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match)
		printf("Arrays match.\n\n");
}

void printData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%3.0f ", in[i]);
    }

    printf("\n");
    return;
}

__global__ void multGPU(float * out, float * A, float * B,int nrows, int ncols, int N)
{

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int outOffset=INDEX(row,col,ncols);
    if(row < nrows && col < ncols)
    {
       
        float sum=0.0f;
        for(int i=0;i<N;i++)
        {
            float a=A[INDEX(row,i,N)];
            float b=B[INDEX(i,col,ncols)];
            sum+=a*b;
            
        }
        out[outOffset] = sum;
    }
}


__global__ void multGPUSmem(float * out, float * A, float * B, int N)
{

    __shared__ float a[BDIMX][BDIMY], b[BDIMX][BDIMY];
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int outOffset=INDEX(row,col,N);
    int tx=threadIdx.x, ty=threadIdx.y,gd=gridDim.x;
    
    float sum = 0.0f;
    if(row < N && col < N)
    {
       
       for(int t=0; t<gd; t++)
       {
         
            int i = t*BDIMY + ty;
            int j = t*BDIMX + tx;
           
            a[ty][tx]=A[INDEX(row,j,N)];
            b[ty][tx]=B[INDEX(i,col,N)];

            __syncthreads();
            for(int k=0;k<BDIMX;k++)   
                sum+=a[ty][k]*b[k][tx];
             __syncthreads();
            
        }
        out[outOffset] = sum;
    }
}



void multCPU(float *out, float *A, float *B, int nrows, int ncols, int N) {
    for (int i = 0; i < nrows; ++i) 
    {
        for (int j = 0; j < ncols; ++j) 
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) 
            {
                sum += A[INDEX(i,k,N)] * B[INDEX(k,j,ncols)];
            }
            out[INDEX(i,j,ncols)] = sum;
        }
    }
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting matrix multiplication at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

   
    int nrows = 1 << 10;
    int N = 1 << 10;
    int ncols = 1 << 10;

    int blockx = 16;
    int blocky = 16;

    if (argc > 1) blockx = atoi(argv[1]);

    if (argc > 2) blocky  = atoi(argv[2]);

    if (argc > 3) nrows  = atoi(argv[3]);

    if (argc > 4) ncols  = atoi(argv[4]);

    if (argc > 5) N  = atoi(argv[5]);

    size_t nBytes = nrows * ncols * sizeof(float);
    size_t nBytes1 = nrows * N * sizeof(float);
    size_t nBytes2 = N * ncols * sizeof(float);
    // execution configuration
    dim3 block (blockx, blocky);
    dim3 grid  ((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes1);
    float *h_B = (float *)malloc(nBytes2);
    float *h_out = (float *)malloc(nBytes);
    float *g_out  = (float *)malloc(nBytes);

    // initialize host array
    initialData(h_A, nrows * N);
    initialData(h_B, N * ncols);
    Stopwatch s;
    multCPU(h_out, h_A, h_B, nrows, ncols,N);
    printf("\n multCPU elapsed %f sec \n\n",s.elapsed());
    // allocate device memory
    float *d_A, *d_B, *d_out;
    CHECK(cudaMalloc((float**)&d_A, nBytes1));
    CHECK(cudaMalloc((float**)&d_B, nBytes2));
    CHECK(cudaMalloc((float**)&d_out, nBytes));
    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes2, cudaMemcpyHostToDevice));
    s.reset();
    multGPU<<<grid, block>>>(d_out, d_A, d_B, nrows, ncols, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("multGPU<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
           block.y, s.elapsed());
    
    CHECK(cudaMemcpy(g_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, g_out, nrows * ncols);
       

    
// deljena memorija moze da se koristi samo nrows=N=ncols
    s.reset();
    multGPUSmem<<<grid, block>>>(d_out, d_A, d_B, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("multGPUSmem<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
           block.y, s.elapsed());
    
    CHECK(cudaMemcpy(g_out, d_out, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, g_out, nrows * ncols);

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_out));
    free(h_A);
    free(h_B);
    free(h_out);
    free(g_out);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

/* 

a starting matrix multiplication at device 0: GeForce GTX 1050
 multCPU elapsed 26.643980 sec

==8940== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==8940== Replaying kernel "multGPU(float*, float*, float*, int, int, int)" (done)
multGPU<<<grid (64,64) block (16,16)>>> elapsed 0.209968 sec
Arrays match.

==8940== Replaying kernel "multGPUSmem(float*, float*, float*, int)" (done)
multGPUSmem<<<grid (64,64) block (16,16)>>> elapsed 0.107873 sec
Arrays match.

==8940== Profiling application: a
==8940== Profiling result:
==8940== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avgtore
Device "GeForce GTX 1050 (0)"
    Kernel: multGPUSmem(float*, float*, float*, int)
          1                      dram_read_throughput                  Device Memory Read Throughput  29.003GB/s  29.003GB/s  29.003GB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  549.79MB/s  549.79MB/s  549.79MB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.200000    1.200000    1.200000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
    Kernel: multGPU(float*, float*, float*, int, int, int)
          1                      dram_read_throughput                  Device Memory Read Throughput  12.312GB/s  12.312GB/s  12.312GB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  206.02MB/s  206.02MB/s  206.02MB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000

*/
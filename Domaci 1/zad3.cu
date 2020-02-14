#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "../common/stopwatch.h"

void initialData(int *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (int)( rand() ) % 100; //100.0f;
    }

    return;
}

void printData(int *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%dth element: %d\n", i, in[i]);
    }

    return;
}

__global__ void reduceNeighbored (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void gpuRecursiveReduceNosync (int *g_idata, int *g_odata,
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
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // nested invoke
    int istride = isize >> 1;

    if(istride > 1 && tid < istride)
    {
    idata[tid] += idata[tid + istride];

    if(tid == 0)
    {
        gpuRecursiveReduceNosync<<<1, istride>>>(idata, odata, istride);
    }
    }
}

__global__ void skalarniPomGPU(int *g_idata, int *g_odata,unsigned int 
    bid, unsigned int isize)
{
	// set thread ID
	unsigned int tid = threadIdx.x;

	// convert global data pointer to the local pointer of this block
	int *idata = g_idata + bid * blockDim.x;
	

	int *odata = &g_odata[bid];
   
	

	int istride = isize >> 1;

	if ( tid < istride)
	{
       
        idata[tid]+=idata[tid + istride];

		if(tid==0)
		{
			gpuRecursiveReduceNosync<<<1,istride>>>(idata,odata,istride);
		}
	}

}

__global__ void skalarni(int* A, int* B, int *g_odata,unsigned int N)
{

    unsigned tid=threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if(i<N) {
        A[i]=A[i]*B[i];

    }
   // __syncthreads();
    
   //__threadfence();
    if(tid==0)
    {
        skalarniPomGPU<<<1,blockDim.x>>>(A,g_odata,blockIdx.x,blockDim.x);  
    }
   

}

__global__ void skalarniUnroll2(int* A, int* B, int *g_odata,unsigned int N)
{

    unsigned tid=threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

    if (i < N) A[i] = A[i] * B[i];
    if (i + blockDim.x < N) {
        A[i + blockDim.x] = A[i + blockDim.x] * B[i + blockDim.x];
    }

    if(tid==0)
    {
        skalarniPomGPU<<<1,blockDim.x>>>(A,g_odata,blockIdx.x,blockDim.x);  
    }
   

}

__global__ void skalarniUnroll4(int* A, int* B, int*C, int *g_odata,unsigned int N)
{

    unsigned tid=threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 4 + tid;

    if (i < N) C[i] = A[i] * B[i];
    if (i + blockDim.x < N) {
        C[i + blockDim.x]     = A[i + blockDim.x] * B[i + blockDim.x];
    }
    if (i + 2 * blockDim.x < N) {
        C[i + 2 * blockDim.x] = A[i+ 2 * blockDim.x] * B[i + 2 * blockDim.x];
    }
    if (i+ 3 * blockDim.x < N) {
        C[i + 3 * blockDim.x] = A[i + 3 * blockDim.x] * B[i + 3 * blockDim.x];
    }
    
    if(tid==0)
    {
        skalarniPomGPU<<<1,blockDim.x>>>(C,g_odata,blockIdx.x,blockDim.x);  
    }
   

}
__global__ void skalarni3(int* A, int* B, int* C, int *g_odata,unsigned int N)
{

    unsigned tid=threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    if(i<N) {
        C[i]=A[i]*B[i];
    }
    if(tid==0)
    {
        skalarniPomGPU<<<1,blockDim.x>>>(C,g_odata,blockIdx.x,blockDim.x);  
    }
}
/* Ovi ostali kerneli su po meni nesigurni 
jer se ne garantuje sinhronizacija izmedju blokova.
Ali daju korektan rezultat za razlicitu velicinu niza.
*/
/*
__global__ void skalarni2(int* A, int* B, int *g_odata,unsigned int N,int grid)
{

    unsigned tid=threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

   
    if(i<N) {
        A[i]=A[i]*B[i];    
    }
    
    if(blockIdx.x==0 && tid==0)
    {
        gpuRecursiveReduceNosync<<<grid,blockDim.x>>>(A,g_odata,blockDim.x);
    }
   

} */

__global__ void skalarni2Unroll4(int* A, int* B,int* C, int *g_odata, const unsigned int N,int grid)
{

    unsigned tid=threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 4 + tid;

    if (i < N) C[i] = A[i] * B[i];
    if (i + blockDim.x < N) {
        C[i + blockDim.x]     = A[i + blockDim.x] * B[i + blockDim.x];
    }
    if (i + 2 * blockDim.x < N) {
        C[i + 2 * blockDim.x] = A[i+ 2 * blockDim.x] * B[i + 2 * blockDim.x];
    }
    if (i+ 3 * blockDim.x < N) {
        C[i + 3 * blockDim.x] = A[i + 3 * blockDim.x] * B[i + 3 * blockDim.x];
    }
    
    if(blockIdx.x==0 && tid == 0)
    {
        gpuRecursiveReduceNosync<<<grid,blockDim.x>>>(C,g_odata,blockDim.x);
    }
   

}

__global__ void skalarni2Unroll2(int* A, int* B,int* C, int *g_odata, const unsigned int N,int grid)
{

    unsigned tid=threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

    if (i < N) C[i] = A[i] * B[i];
    if (i + blockDim.x < N) {
        C[i + blockDim.x] = A[i + blockDim.x] * B[i + blockDim.x];
    }
    
    if(blockIdx.x==0 && tid == 0)
    {
        gpuRecursiveReduceNosync<<<grid,blockDim.x>>>(C,g_odata,blockDim.x);
    }
   

}

__global__ void skalarni2(int* A, int* B,int* C, int *g_odata, const unsigned int N,int grid)
{

    unsigned tid=threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    if (i < N) C[i] = A[i] * B[i];
   
    if(i==N-1)
    {
        gpuRecursiveReduceNosync<<<grid,blockDim.x>>>(C,g_odata,blockDim.x);
    }
   

}


__global__ void skalarni4(int* A, int* B,int* C, int *g_odata, const unsigned int N,int grid)
{

    unsigned tid=threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    if (i < N) C[i] = A[i] * B[i];
    
    if(i==N-1)
    {
        reduceNeighbored<<<grid,blockDim.x>>>(C,g_odata,N);
    }

}


int cpuSkalarni(int* in1, int* in2, int size)
{
    int s=0;
    for(int i=0; i<size; i++)
        s+=in1[i]*in2[i];   
    return s;
}


int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int power=22;
    int nthreads=512;
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
	
    size_t nBytes = nElem * sizeof(int);
    
    int *h_A, *h_B, *h_Out;
	
	dim3 block(nthreads);
    dim3 grid((nElem + block.x - 1) / block.x);
   
    
	h_A = (int *)malloc(nBytes);
	h_B = (int *)malloc(nBytes);
    h_Out = (int *) malloc(grid.x*sizeof(int));
    
    initialData(h_A, nElem);
	initialData(h_B, nElem);
    
    

    Stopwatch s;
    int h_s = cpuSkalarni(h_A, h_B, nElem);
    printf("\nSkalarniOnCPU- Time elapsed %fsec \n", s.elapsed());
   
    //print(h_A, h_B, nElem);
    int *d_A, *d_B, *d_C, *d_Out;
    CHECK(cudaMalloc((int**)&d_A, nBytes));
    CHECK(cudaMalloc((int**)&d_B, nBytes));
    CHECK(cudaMalloc((int**)&d_C, nBytes));
    CHECK(cudaMalloc((int**)&d_Out, grid.x*sizeof(int)));
    //transfer podataka sa host-a na device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    //poziv kernela sa host-a

    
    //printData(h_A,nElem);
    
    //printData(h_B,nElem);
    s.reset();
    skalarni <<<grid, block>>> (d_A, d_B, d_Out, nElem);
   
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError()); 
    
    printf("\n skalarni <<<%d,%d>>> Time elapsed %f" \
    "sec\n", grid.x, block.x, s.elapsed());


    CHECK(cudaMemcpy(h_Out, d_Out, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    int d_s=0;
    for(int i=0;i<grid.x;i++) d_s+=h_Out[i];

    printf(" %d",h_s);
    printf("\n %d\n",d_s);


    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_Out,0,grid.x*sizeof(int)));
    s.reset();
    skalarniUnroll2 <<<grid, block>>> (d_A, d_B, d_Out, nElem);
   
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError()); 
    
    printf("\n skalarniUnroll2 <<<%d,%d>>> Time elapsed %f" \
    "sec\n", grid.x, block.x, s.elapsed());

    CHECK(cudaMemcpy(h_Out, d_Out, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    d_s=0;
    for(int i=0;i<grid.x;i++) d_s+=h_Out[i];

    printf(" %d",h_s);
    printf("\n %d\n",d_s);

    

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_Out,0,grid.x*sizeof(int)));
    s.reset();
    skalarniUnroll4 <<<grid, block>>> (d_A, d_B,d_C, d_Out, nElem);
   
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError()); 
    
    printf("\n skalarniUnroll4 <<<%d,%d>>> Time elapsed %f" \
    "sec\n", grid.x, block.x, s.elapsed());

    CHECK(cudaMemcpy(h_Out, d_Out, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    d_s=0;
    for(int i=0;i<grid.x;i++) d_s+=h_Out[i];

    printf(" %d",h_s);
    printf("\n %d\n",d_s);
    
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_Out,0,grid.x*sizeof(int)));
    s.reset();
    skalarni2<<<grid,block>>>(d_A,d_B,d_C,d_Out,nElem,grid.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError()); 
    
    printf("\n Skalarni2 <<<%d,%d>>> Time elapsed %f" \
    "sec\n", grid.x, block.x, s.elapsed());

// kopiranje rezultata kernela nazad na host
    CHECK(cudaMemcpy(h_Out, d_Out, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    d_s=0;
    for(int i=0;i<grid.x;i++) d_s+=h_Out[i];

    printf(" %d",h_s);
    printf("\n %d\n",d_s); 
    
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_Out,0,grid.x*sizeof(int)));
    s.reset();
    skalarni3<<<grid,block>>>(d_A,d_B,d_C,d_Out,nElem);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError()); 
    
    printf("\n Skalarni3 <<<%d,%d>>> Time elapsed %f" \
    "sec\n", grid.x, block.x, s.elapsed());

// kopiranje rezultata kernela nazad na host
    CHECK(cudaMemcpy(h_Out, d_Out, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    d_s=0;
    for(int i=0;i<grid.x;i++) d_s+=h_Out[i];

    printf(" %d",h_s);
    printf("\n %d\n",d_s);

    CHECK(cudaMemset(d_Out,0,grid.x*sizeof(int)));
    s.reset();
    skalarni2Unroll2<<<grid.x/2,block>>>(d_A,d_B,d_C,d_Out,nElem,grid.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError()); 
    
    printf("\n skalarni2Unroll2 <<<%d,%d>>> Time elapsed %f" \
    "sec\n", grid.x, block.x, s.elapsed());

// kopiranje rezultata kernela nazad na host
    CHECK(cudaMemcpy(h_Out, d_Out, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    d_s=0;
    for(int i=0;i<grid.x;i++) 
    {
        d_s+=h_Out[i];
       // printf("\t %d",h_Out[i]);
    }

    printf(" %d",h_s);
    printf("\n %d \n",d_s);
    CHECK(cudaMemset(d_Out,0,grid.x*sizeof(int)));
    s.reset();
    skalarni2Unroll4<<<grid.x/4,block>>>(d_A,d_B,d_C,d_Out,nElem,grid.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError()); 
    
    printf("\n skalarni2Unroll4 <<<%d,%d>>> Time elapsed %f" \
    "sec\n", grid.x, block.x, s.elapsed());

// kopiranje rezultata kernela nazad na host
    CHECK(cudaMemcpy(h_Out, d_Out, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    d_s=0;
    for(int i=0;i<grid.x;i++) d_s+=h_Out[i];

    printf(" %d",h_s);
    printf("\n %d \n",d_s);

    CHECK(cudaMemset(d_Out,0,grid.x*sizeof(int)));
    
    s.reset();
    skalarni4<<<grid,block>>>(d_A,d_B,d_C,d_Out,nElem,grid.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError()); 
    memset(h_Out,0,grid.x*sizeof(int));
    printf("\n skalarni4 <<<%d,%d>>> Time elapsed %f" \
    "sec\n", grid.x, block.x, s.elapsed());

// kopiranje rezultata kernela nazad na host
    CHECK(cudaMemcpy(h_Out, d_Out, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    d_s=0;
    for(int i=0;i<grid.x;i++) d_s+=h_Out[i];

    printf(" %d",h_s);
    printf("\n %d \n",d_s);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Out);
  
    free(h_A);
    free(h_B);
    free(h_Out);
    cudaDeviceReset();
    return(0);
}

/* 
GeForce GTX 1050

SkalarniOnCPU- Time elapsed 0.012361sec

 skalarni <<<131072,32>>> Time elapsed 0.645046sec
 1677013280
 1677013280

 skalarniUnroll2 <<<131072,32>>> Time elapsed 0.637585sec
 1677013280
 1677013280

 skalarniUnroll4 <<<131072,32>>> Time elapsed 0.637881sec
 1677013280
 1677013280

 Skalarni2 <<<131072,32>>> Time elapsed 0.429906sec
 1677013280
 1677013280

 Skalarni3 <<<131072,32>>> Time elapsed 0.636618sec
 1677013280
 1677013280

 skalarni2Unroll2 <<<131072,32>>> Time elapsed 0.429518sec
 1677013280
 1677013280

 skalarni2Unroll4 <<<131072,32>>> Time elapsed 0.428573sec
 1677013280
 1677013280

 skalarni4 <<<131072,32>>> Time elapsed 0.009061sec
 1677013280
 1677013280
==1524== Warning: 4124404 records were dropped due to insufficient device buffer space. You can configure the buffer space using advanced options --device-buffer-size, --device-cdp-buffer-size
==1524== Profiling application: a 32
==1524== Profiling result:
            Type  Time(%)      Time  Calls (host)  Calls (device)       Avg       Min       Max  Name
 GPU activities:   29.36%  1.28975s             0           55872  23.084us  1.1840us  107.42us  gpuRecursiveReduceNosync(int*, int*, unsigned int)
                   14.66%  643.99ms             1               0  643.99ms  643.99ms  643.99ms  skalarni(int*, int*, int*, unsigned int)
                   14.50%  637.14ms             1               0  637.14ms  637.14ms  637.14ms  skalarniUnroll4(int*, int*, int*, int*, unsigned int)
                   14.50%  636.85ms             1               0  636.85ms  636.85ms  636.85ms  skalarniUnroll2(int*, int*, int*, unsigned int)
                   14.48%  635.87ms             1               0  635.87ms  635.87ms  635.87ms  skalarni3(int*, int*, int*, int*, unsigned int)
                    9.25%  406.42ms             0           14032  28.963us  16.992us  107.20us  skalarniPomGPU(int*, int*, unsigned int, unsigned int)
                    1.37%  60.250ms            10               -  6.0250ms  5.4468ms  6.5732ms  [CUDA memcpy HtoD]
                    1.08%  47.394ms             1               0  47.394ms  47.394ms  47.394ms  skalarni2Unroll2(int*, int*, int*, int*, unsigned int, int)
                    0.65%  28.343ms             1               0  28.343ms  28.343ms  28.343ms  skalarni2Unroll4(int*, int*, int*, int*, unsigned int, int)
                    0.07%  3.1848ms             1               0  3.1848ms  3.1848ms  3.1848ms  skalarni2(int*, int*, int*, int*, unsigned int, int)
                    0.07%  2.9479ms             1               0  2.9479ms  2.9479ms  2.9479ms  skalarni4(int*, int*, int*, int*, unsigned int, int)
                    0.02%  663.80us             8               -  82.975us  78.719us  86.271us  [CUDA memcpy DtoH]
                    0.00%  4.8960us             7               -     699ns     640ns     896ns  [CUDA memset]
      API calls:   90.80%  3.85326s             8               -  481.66ms  8.7100ms  644.96ms  cudaDeviceSynchronize
                    5.88%  249.34ms             4               -  62.335ms  557.60us  246.32ms  cudaMalloc
                    1.66%  70.373ms            18               -  3.9096ms  394.70us  7.0672ms  cudaMemcpy
                    1.58%  66.956ms             1               -  66.956ms  66.956ms  66.956ms  cudaDeviceReset
                    0.02%  1.0196ms             4               -  254.90us  153.00us  521.80us  cudaFree
                    0.02%  787.80us             7               -  112.54us  51.700us  238.60us  cudaMemset
                    0.02%  692.10us            97               -  7.1350us     200ns  322.30us  cuDeviceGetAttribute
                    0.01%  598.30us             1               -  598.30us  598.30us  598.30us  cudaGetDeviceProperties
                    0.01%  533.80us             8               -  66.725us  37.700us  218.10us  cudaLaunchKernel
                    0.00%  44.000us             1               -  44.000us  44.000us  44.000us  cuDeviceTotalMem
                    0.00%  13.300us             1               -  13.300us  13.300us  13.300us  cuDeviceGetPCIBusId
                    0.00%  12.300us             1               -  12.300us  12.300us  12.300us  cudaSetDevice
                    0.00%  7.5000us             8               -     937ns     800ns  1.0000us  cudaGetLastError
                    0.00%  6.8000us             2               -  3.4000us     400ns  6.4000us  cuDeviceGet
                    0.00%  2.1000us             3               -     700ns     400ns  1.1000us  cuDeviceGetCount
                    0.00%  1.3000us             1               -  1.3000us  1.3000us  1.3000us  cuDeviceGetName
                    0.00%     500ns             1               -     500ns     500ns     500ns  cuDeviceGetUuid
                    0.00%     500ns             1               -     500ns     500ns     500ns  cuDeviceGetLuid





SkalarniOnCPU- Time elapsed 0.012476sec

 skalarni <<<16384,256>>> Time elapsed 0.140732sec
 1677013280
 1677013280

 skalarniUnroll2 <<<16384,256>>> Time elapsed 0.130778sec
 1677013280
 1677013280

 skalarniUnroll4 <<<16384,256>>> Time elapsed 0.132044sec
 1677013280
 1677013280

 Skalarni2 <<<16384,256>>> Time elapsed 0.105880sec
 1677013280
 1677013280

 Skalarni3 <<<16384,256>>> Time elapsed 0.131352sec
 1677013280
 1677013280

 skalarni2Unroll2 <<<16384,256>>> Time elapsed 0.105701sec
 1677013280
 1677013280

 skalarni2Unroll4 <<<16384,256>>> Time elapsed 0.105816sec
 1677013280
 1677013280

 skalarni4 <<<16384,256>>> Time elapsed 0.004052sec
 1677013280
 1677013280
==1240== Warning: 798452 records were dropped due to insufficient device buffer space. You can configure the buffer space using advanced options --device-buffer-size, --device-cdp-buffer-size
==1240== Profiling application: a 256
==1240== Profiling result:
            Type  Time(%)      Time  Calls (host)  Calls (device)       Avg       Min       Max  Name
 GPU activities:   62.92%  1.50742s             0           61145  24.653us  1.2160us  64.927us  gpuRecursiveReduceNosync(int*, int*, unsigned int)
                   10.36%  248.27ms             0            8759  28.344us  18.336us  61.824us  skalarniPomGPU(int*, int*, unsigned int, unsigned int)
                    5.83%  139.69ms             1               0  139.69ms  139.69ms  139.69ms  skalarni(int*, int*, int*, unsigned int)
                    5.47%  131.12ms             1               0  131.12ms  131.12ms  131.12ms  skalarniUnroll4(int*, int*, int*, int*, unsigned int)
                    5.44%  130.34ms             1               0  130.34ms  130.34ms  130.34ms  skalarni3(int*, int*, int*, int*, unsigned int)
                    5.43%  130.00ms             1               0  130.00ms  130.00ms  130.00ms  skalarniUnroll2(int*, int*, int*, unsigned int)
                    2.55%  61.031ms            10               -  6.1031ms  5.6811ms  6.6876ms  [CUDA memcpy HtoD]
                    1.24%  29.623ms             1               0  29.623ms  29.623ms  29.623ms  skalarni2Unroll2(int*, int*, int*, int*, unsigned int, int)
                    0.65%  15.552ms             1               0  15.552ms  15.552ms  15.552ms  skalarni2Unroll4(int*, int*, int*, int*, unsigned int, int)
                    0.05%  1.2330ms             1               0  1.2330ms  1.2330ms  1.2330ms  skalarni4(int*, int*, int*, int*, unsigned int, int)
                    0.05%  1.2298ms             1               0  1.2298ms  1.2298ms  1.2298ms  skalarni2(int*, int*, int*, int*, unsigned int, int)
                    0.00%  87.327us             8               -  10.915us  10.560us  11.360us  [CUDA memcpy DtoH]
                    0.00%  21.023us             7               -  3.0030us  2.7190us  3.3920us  [CUDA memset]
      API calls:   68.90%  855.15ms             8               -  106.89ms  3.9697ms  140.64ms  cudaDeviceSynchronize
                   19.61%  243.44ms             4               -  60.859ms  583.20us  240.45ms  cudaMalloc
                    5.62%  69.803ms            18               -  3.8779ms  244.70us  7.1879ms  cudaMemcpy
                    5.56%  69.051ms             1               -  69.051ms  69.051ms  69.051ms  cudaDeviceReset
                    0.10%  1.2515ms             4               -  312.88us  223.60us  429.20us  cudaFree
                    0.09%  1.0970ms             8               -  137.13us  69.600us  301.10us  cudaLaunchKernel
                    0.04%  497.70us            97               -  5.1300us     200ns  215.70us  cuDeviceGetAttribute
                    0.03%  372.80us             7               -  53.257us  21.900us  204.30us  cudaMemset
                    0.03%  370.70us             1               -  370.70us  370.70us  370.70us  cudaGetDeviceProperties
                    0.00%  41.700us             1               -  41.700us  41.700us  41.700us  cuDeviceTotalMem
                    0.00%  13.400us             1               -  13.400us  13.400us  13.400us  cuDeviceGetPCIBusId
                    0.00%  12.400us             1               -  12.400us  12.400us  12.400us  cudaSetDevice
                    0.00%  6.6000us             2               -  3.3000us     500ns  6.1000us  cuDeviceGet
                    0.00%  6.5000us             8               -     812ns     500ns  1.1000us  cudaGetLastError
                    0.00%  2.4000us             3               -     800ns     300ns  1.1000us  cuDeviceGetCount
                    0.00%  1.4000us             1               -  1.4000us  1.4000us  1.4000us  cuDeviceGetName
                    0.00%     800ns             1               -     800ns     800ns     800ns  cuDeviceGetLuid
                    0.00%     500ns             1               -     500ns     500ns     500ns  cuDeviceGetUuid



SkalarniOnCPU- Time elapsed 0.012613sec

 skalarni <<<8192,512>>> Time elapsed 0.086526sec
 1677013280
 1677013280

 skalarniUnroll2 <<<8192,512>>> Time elapsed 0.076569sec
 1677013280
 1677013280

 skalarniUnroll4 <<<8192,512>>> Time elapsed 0.076975sec
 1677013280
 1677013280

 Skalarni2 <<<8192,512>>> Time elapsed 0.065407sec
 1677013280
 1677013280

 Skalarni3 <<<8192,512>>> Time elapsed 0.076832sec
 1677013280
 1677013280

 skalarni2Unroll2 <<<8192,512>>> Time elapsed 0.065597sec
 1677013280
 1677013280

 skalarni2Unroll4 <<<8192,512>>> Time elapsed 0.064851sec
 1677013280
 1677013280

 skalarni4 <<<8192,512>>> Time elapsed 0.004588sec
 1677013280
 1677013280
==3456== Warning: 421620 records were dropped due to insufficient device buffer space. You can configure the buffer space using advanced options --device-buffer-size, --device-cdp-buffer-size
==3456== Profiling application: a 512
==3456== Profiling result:
            Type  Time(%)      Time  Calls (host)  Calls (device)       Avg       Min       Max  Name
 GPU activities:   71.08%  1.54750s             0           62110  24.915us  1.2160us  50.400us  gpuRecursiveReduceNosync(int*, int*, unsigned int)
                    9.79%  213.13ms             0            7794  27.344us  19.136us  47.711us  skalarniPomGPU(int*, int*, unsigned int, unsigned int)
                    3.93%  85.534ms             1               0  85.534ms  85.534ms  85.534ms  skalarni(int*, int*, int*, unsigned int)
                    3.50%  76.160ms             1               0  76.160ms  76.160ms  76.160ms  skalarniUnroll4(int*, int*, int*, int*, unsigned int)
                    3.49%  76.075ms             1               0  76.075ms  76.075ms  76.075ms  skalarni3(int*, int*, int*, int*, unsigned int)
                    3.48%  75.695ms             1               0  75.695ms  75.695ms  75.695ms  skalarniUnroll2(int*, int*, int*, unsigned int)
                    2.76%  59.995ms            10               -  5.9995ms  5.4042ms  6.7181ms  [CUDA memcpy HtoD]
                    1.20%  26.159ms             1               0  26.159ms  26.159ms  26.159ms  skalarni2Unroll2(int*, int*, int*, int*, unsigned int, int)
                    0.63%  13.655ms             1               0  13.655ms  13.655ms  13.655ms  skalarni2Unroll4(int*, int*, int*, int*, unsigned int, int)
                    0.09%  1.9138ms             1               0  1.9138ms  1.9138ms  1.9138ms  skalarni2(int*, int*, int*, int*, unsigned int, int)
                    0.06%  1.2708ms             1               0  1.2708ms  1.2708ms  1.2708ms  skalarni4(int*, int*, int*, int*, unsigned int, int)
                    0.00%  50.591us             8               -  6.3230us  5.6960us  7.5520us  [CUDA memcpy DtoH]
                    0.00%  12.864us             7               -  1.8370us  1.7280us  2.0160us  [CUDA memset]
      API calls:   57.40%  516.24ms             8               -  64.531ms  4.4770ms  86.437ms  cudaDeviceSynchronize
                   27.08%  243.56ms             4               -  60.890ms  467.40us  240.62ms  cudaMalloc
                    7.66%  68.855ms            18               -  3.8253ms  201.80us  7.3965ms  cudaMemcpy
                    7.40%  66.553ms             1               -  66.553ms  66.553ms  66.553ms  cudaDeviceReset
                    0.19%  1.6732ms             4               -  418.30us  170.00us  920.20us  cudaFree
                    0.11%  981.90us             8               -  122.74us  69.200us  227.20us  cudaLaunchKernel
                    0.06%  575.00us             7               -  82.142us  22.600us  291.50us  cudaMemset
                    0.05%  463.20us            97               -  4.7750us     200ns  207.70us  cuDeviceGetAttribute
                    0.04%  370.10us             1               -  370.10us  370.10us  370.10us  cudaGetDeviceProperties
                    0.00%  40.700us             1               -  40.700us  40.700us  40.700us  cuDeviceTotalMem
                    0.00%  13.400us             1               -  13.400us  13.400us  13.400us  cuDeviceGetPCIBusId
                    0.00%  12.900us             1               -  12.900us  12.900us  12.900us  cudaSetDevice
                    0.00%  6.7000us             8               -     837ns     500ns  1.0000us  cudaGetLastError
                    0.00%  6.3000us             2               -  3.1500us     400ns  5.9000us  cuDeviceGet
                    0.00%  2.5000us             3               -     833ns     400ns  1.1000us  cuDeviceGetCount
                    0.00%  1.5000us             1               -  1.5000us  1.5000us  1.5000us  cuDeviceGetName
                    0.00%     800ns             1               -     800ns     800ns     800ns  cuDeviceGetLuid
                    0.00%     500ns             1               -     500ns     500ns     500ns  cuDeviceGetUuid






SkalarniOnCPU- Time elapsed 0.012840sec

 skalarni <<<4096,1024>>> Time elapsed 0.051539sec
 1677013280
 1677013280

 skalarniUnroll2 <<<4096,1024>>> Time elapsed 0.050108sec
 1677013280
 1677013280

 skalarniUnroll4 <<<4096,1024>>> Time elapsed 0.046660sec
 1677013280
 1677013280

 Skalarni2 <<<4096,1024>>> Time elapsed 0.040571sec
 1677013280
 1677013280

 Skalarni3 <<<4096,1024>>> Time elapsed 0.046409sec
 1677013280
 1677013280

 skalarni2Unroll2 <<<4096,1024>>> Time elapsed 0.040531sec
 1677013280
 1677013280

 skalarni2Unroll4 <<<4096,1024>>> Time elapsed 0.040056sec
 1677013280
 1677013280

 skalarni4 <<<4096,1024>>> Time elapsed 0.005426sec
 1677013280
 1677013280
==5908== Warning: 204532 records were dropped due to insufficient device buffer space. You can configure the buffer space using advanced options --device-buffer-size, --device-cdp-buffer-size
==5908== Profiling application: a 1024
==5908== Profiling result:
            Type  Time(%)      Time  Calls (host)  Calls (device)       Avg       Min       Max  Name
 GPU activities:   77.23%  1.59776s             0           62899  25.401us  1.2800us  183.58us  gpuRecursiveReduceNosync(int*, int*, unsigned int)
                    9.38%  194.08ms             0            7005  27.706us  19.392us  176.99us  skalarniPomGPU(int*, int*, unsigned int, unsigned int)
                    2.88%  59.628ms            10               -  5.9628ms  5.2801ms  6.6448ms  [CUDA memcpy HtoD]
                    2.45%  50.593ms             1               0  50.593ms  50.593ms  50.593ms  skalarni(int*, int*, int*, unsigned int)
                    2.38%  49.227ms             1               0  49.227ms  49.227ms  49.227ms  skalarniUnroll2(int*, int*, int*, unsigned int)
                    2.21%  45.784ms             1               0  45.784ms  45.784ms  45.784ms  skalarniUnroll4(int*, int*, int*, int*, unsigned int)
                    2.19%  45.308ms             1               0  45.308ms  45.308ms  45.308ms  skalarni3(int*, int*, int*, int*, unsigned int)
                    0.72%  14.826ms             1               0  14.826ms  14.826ms  14.826ms  skalarni2Unroll2(int*, int*, int*, int*, unsigned int, int)
                    0.41%  8.5484ms             1               0  8.5484ms  8.5484ms  8.5484ms  skalarni2Unroll4(int*, int*, int*, int*, unsigned int, int)
                    0.08%  1.7003ms             1               0  1.7003ms  1.7003ms  1.7003ms  skalarni4(int*, int*, int*, int*, unsigned int, int)
                    0.06%  1.3143ms             1               0  1.3143ms  1.3143ms  1.3143ms  skalarni2(int*, int*, int*, int*, unsigned int, int)
                    0.00%  26.688us             8               -  3.3360us  2.9440us  4.0960us  [CUDA memcpy DtoH]
                    0.00%  9.4720us             7               -  1.3530us     768ns  1.5680us  [CUDA memset]
      API calls:   45.30%  319.94ms             8               -  39.993ms  5.3326ms  51.447ms  cudaDeviceSynchronize
                   35.36%  249.76ms             4               -  62.439ms  546.70us  246.77ms  cudaMalloc
                    9.43%  66.636ms            18               -  3.7020ms  201.60us  7.1874ms  cudaMemcpy
                    9.29%  65.641ms             1               -  65.641ms  65.641ms  65.641ms  cudaDeviceReset
                    0.15%  1.0551ms             8               -  131.89us  70.000us  237.20us  cudaLaunchKernel
                    0.14%  1.0145ms             7               -  144.93us  22.400us  656.30us  cudaMemset
                    0.13%  888.40us             4               -  222.10us  183.50us  309.20us  cudaFree
                    0.10%  687.90us            97               -  7.0910us     200ns  313.20us  cuDeviceGetAttribute
                    0.08%  597.90us             1               -  597.90us  597.90us  597.90us  cudaGetDeviceProperties
                    0.01%  40.200us             1               -  40.200us  40.200us  40.200us  cuDeviceTotalMem
                    0.00%  13.800us             1               -  13.800us  13.800us  13.800us  cuDeviceGetPCIBusId
                    0.00%  12.800us             1               -  12.800us  12.800us  12.800us  cudaSetDevice
                    0.00%  7.7000us             8               -     962ns     900ns  1.1000us  cudaGetLastError
                    0.00%  7.6000us             2               -  3.8000us     400ns  7.2000us  cuDeviceGet
                    0.00%  1.9000us             3               -     633ns     300ns  1.1000us  cuDeviceGetCount
                    0.00%  1.4000us             1               -  1.4000us  1.4000us  1.4000us  cuDeviceGetName
                    0.00%     700ns             1               -     700ns     700ns     700ns  cuDeviceGetLuid
                    0.00%     500ns             1               -     500ns     500ns     500ns  cuDeviceGetUuid
                    */
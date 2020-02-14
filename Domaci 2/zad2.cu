#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "../common/stopwatch.h"



void initialData1(int *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (int)(rand() % 1000)+1;
    }

    return;
}
void initialData2(int *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (int)(rand() % 80)+1;
    }

    return;
}


int maxCPU(int a, int b) { return (a>b)? a : b;}

void printData(int *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", in[i]);
    }

    printf("\n");
    return;
}

int KPCPU1 (int *v, int *w, int *dp, int N, int W)
{
    for(int i=0;i<N;i++)
    {
        for(int j=W; j>=w[i]; j--) 
            dp[j]=maxCPU(dp[j],dp[j-w[i]]+v[i]);
       
    }
        return dp[W];
}

int KPCPU2 (int *v, int *w, int **dp, int N, int W)
{
    
   
    for(int i=0;i<=N;i++)
        dp[i][0]=0;
    
    for(int i=0;i<=W;i++)
        dp[0][i]=0;

    for(int i=1;i<=N;i++)
    {
        for(int j=1; j<=W;j++)
        {
            if(j<w[i-1])
            dp[i][j]=dp[i-1][j];
            else
            dp[i][j]=maxCPU(dp[i-1][j],dp[i-1][j-w[i-1]]+v[i-1]);
            //printf("%d ",dp[i][j]);
           
        }
        //printf("\n");
    }
   
        return dp[N][W];
}

__device__ int maxGPU (int a, int b) { return (a>b)? a : b;}

__global__ void KPGPU(int *prev, int *tmp,int *w, int *v,int N, int W, int i)
{

    unsigned int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<=W)
    {
        if(j<w[i])
        tmp[j]=prev[j];
        else
        tmp[j]=maxGPU(prev[j],prev[j-w[i]]+v[i]);
    }
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting main at ", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    


    int B=512;
    int W=400,N=100000;

    if (argc > 1) B = atoi(argv[1]);

    if (argc > 2) N = atoi(argv[2]);
   
    if (argc > 3) W = atoi(argv[2]);

  

   
    dim3 block (B);
    dim3 grid ((W + block.x - 1) / block.x); 

    int *h_DP, *h_v, *h_w,**h_mat,*h_res;
    int *d_v, *d_w,*d_prev,*d_tmp;
    size_t nBytes = (W+1) * sizeof(int);
    h_DP = (int *)malloc(nBytes);
    h_res = (int *)malloc(nBytes);
    cudaMalloc((int**)&d_prev, nBytes);
    cudaMalloc((int**)&d_tmp, nBytes);

    cudaMemset(d_prev,0,nBytes);
    cudaMemset(d_tmp,0,nBytes);

    h_mat=(int**)malloc((N+1)*sizeof(int*));
    for(int i=0;i<=N;i++)
        h_mat[i]=(int*)malloc((nBytes));
    
    memset(h_DP,0,nBytes);
    nBytes = N*sizeof(int);
    h_v = (int *)malloc(nBytes);
    h_w = (int *)malloc(nBytes);
    initialData1(h_v,N);
    initialData2(h_w,N);

    
	cudaMalloc((int**)&d_v, nBytes);
    cudaMalloc((int**)&d_w, nBytes);

    CHECK(cudaMemcpy(d_v, h_v, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, h_w, nBytes, cudaMemcpyHostToDevice));
    Stopwatch s;
    printf("%d\n",KPCPU1(h_v,h_w,h_DP,N,W));
    printf("KPCPU1 elapsed %f sec \n",s.elapsed());
    s.reset();
    printf("%d",KPCPU2(h_v,h_w,h_mat,N,W));
    printf("\nKPCPU2 elapsed %f sec \n\n",s.elapsed());

    s.reset();
    for(int i=0;i<=N;i++)
        {
            KPGPU<<<grid,block>>>(d_prev,d_tmp,d_w,d_v,N,W,i);
            CHECK(cudaStreamSynchronize(0));
            CHECK(cudaGetLastError());
            int *t=d_prev;
            d_prev=d_tmp;
            d_tmp=t;
        }
    printf("\n%d * KPGPU<<<%d,%d>>>  elapsed %f sec \n",N,grid.x,block.x,s.elapsed());
    CHECK(cudaMemcpy(h_res, d_prev, (W+1) *sizeof(int), cudaMemcpyDeviceToHost));
    printf("%d\n",h_res[W]);
    for(int i=0;i<=N;i++)
    free(h_mat[i]);
    free(h_mat);
    free(h_DP);
    free(h_v);
    free(h_w);

    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_prev);
    cudaFree(d_tmp);
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}


/* 

a starting main at device 0: GeForce GTX 1050
327704
KPCPU1 elapsed 0.232883 sec
327704
KPCPU2 elapsed 0.371544 sec


100000 * KPGPU<<<13,32>>>  elapsed 9.027132 sec
327704
==6696== Profiling application: a 32
==6696== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.90%  123.64ms    100001  1.2360us  1.1830us  2.5920us  KPGPU(int*, int*, int*, int*, int, int, int)
                    0.10%  122.53us         2  61.264us  61.184us  61.344us  [CUDA memcpy HtoD]
                    0.00%  2.2720us         2  1.1360us     800ns  1.4720us  [CUDA memset]
                    0.00%  1.3760us         1  1.3760us  1.3760us  1.3760us  [CUDA memcpy DtoH]
      API calls:   79.55%  7.22795s    100001  72.278us  23.400us  1.7266ms  cudaStreamSynchronize
                   17.01%  1.54571s    100001  15.456us  13.600us  1.2323ms  cudaLaunchKernel
                    2.38%  216.29ms         4  54.072ms  9.4000us  216.22ms  cudaMalloc
                    0.66%  59.521ms         1  59.521ms  59.521ms  59.521ms  cudaDeviceReset
                    0.38%  34.281ms    100001     342ns     200ns  281.30us  cudaGetLastError
                    0.01%  723.80us         3  241.27us  143.10us  352.40us  cudaMemcpy
                    0.01%  717.50us        97  7.3960us     200ns  315.70us  cuDeviceGetAttribute
                    0.01%  602.80us         1  602.80us  602.80us  602.80us  cudaGetDeviceProperties
                    0.00%  397.10us         4  99.275us  24.900us  255.60us  cudaFree
                    0.00%  45.100us         1  45.100us  45.100us  45.100us  cuDeviceTotalMem
                    0.00%  31.000us         2  15.500us  5.0000us  26.000us  cudaMemset
                    0.00%  13.500us         1  13.500us  13.500us  13.500us  cudaSetDevice
                    0.00%  13.400us         1  13.400us  13.400us  13.400us  cuDeviceGetPCIBusId
                    0.00%  4.6000us         2  2.3000us     400ns  4.2000us  cuDeviceGet
                    0.00%  2.3000us         3     766ns     300ns  1.0000us  cuDeviceGetCount
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuDeviceGetName
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetUuid
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetLuid





a starting main at device 0: GeForce GTX 1050
327704
KPCPU1 elapsed 0.249529 sec
327704
KPCPU2 elapsed 0.389207 sec


100000 * KPGPU<<<2,256>>>  elapsed 9.194475 sec
327704
==6328== Profiling application: a 256
==6328== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.90%  122.36ms    100001  1.2230us  1.1520us  2.4320us  KPGPU(int*, int*, int*, int*, int, int, int)
                    0.10%  122.62us         2  61.312us  61.152us  61.472us  [CUDA memcpy HtoD]
                    0.00%  2.2400us         2  1.1200us     800ns  1.4400us  [CUDA memset]
                    0.00%  1.3440us         1  1.3440us  1.3440us  1.3440us  [CUDA memcpy DtoH]
      API calls:   79.55%  7.36634s    100001  73.662us  26.800us  5.2921ms  cudaStreamSynchronize
                   16.96%  1.57101s    100001  15.709us  13.500us  415.40us  cudaLaunchKernel
                    2.46%  227.53ms         4  56.884ms  9.1000us  227.47ms  cudaMalloc
                    0.63%  58.153ms         1  58.153ms  58.153ms  58.153ms  cudaDeviceReset
                    0.37%  34.661ms    100001     346ns     200ns  231.40us  cudaGetLastError
                    0.01%  798.40us         3  266.13us  198.20us  357.20us  cudaMemcpy
                    0.01%  656.90us         1  656.90us  656.90us  656.90us  cudaGetDeviceProperties
                    0.01%  609.30us        97  6.2810us     200ns  257.00us  cuDeviceGetAttribute
                    0.01%  475.50us         4  118.88us  24.500us  304.00us  cudaFree
                    0.00%  41.500us         1  41.500us  41.500us  41.500us  cuDeviceTotalMem
                    0.00%  27.200us         2  13.600us  5.3000us  21.900us  cudaMemset
                    0.00%  18.900us         1  18.900us  18.900us  18.900us  cudaSetDevice
                    0.00%  13.700us         1  13.700us  13.700us  13.700us  cuDeviceGetPCIBusId
                    0.00%  4.3000us         2  2.1500us     400ns  3.9000us  cuDeviceGet
                    0.00%  2.8000us         3     933ns     300ns  1.5000us  cuDeviceGetCount
                    0.00%  1.4000us         1  1.4000us  1.4000us  1.4000us  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetUuid
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetLuid








                    a starting main at device 0: GeForce GTX 1050
327704
KPCPU1 elapsed 0.233630 sec
327704
KPCPU2 elapsed 0.370733 sec


100000 * KPGPU<<<1,512>>>  elapsed 9.369431 sec
327704
==4884== Profiling application: a
==4884== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.90%  125.31ms    100001  1.2530us  1.1830us  6.6870us  KPGPU(int*, int*, int*, int*, int, int, int)
                    0.10%  122.14us         2  61.072us  61.024us  61.120us  [CUDA memcpy HtoD]
                    0.00%  2.3040us         2  1.1520us     800ns  1.5040us  [CUDA memset]
                    0.00%  1.5680us         1  1.5680us  1.5680us  1.5680us  [CUDA memcpy DtoH]
      API calls:   79.02%  7.44478s    100001  74.447us  23.200us  12.277ms  cudaStreamSynchronize
                   17.66%  1.66368s    100001  16.636us  13.500us  6.9754ms  cudaLaunchKernel
                    2.26%  212.78ms         4  53.194ms  9.9000us  212.71ms  cudaMalloc
                    0.64%  60.005ms         1  60.005ms  60.005ms  60.005ms  cudaDeviceReset
                    0.39%  36.685ms    100001     366ns     200ns  1.0904ms  cudaGetLastError
                    0.01%  1.0136ms         3  337.87us  213.00us  437.30us  cudaMemcpy
                    0.01%  681.70us        97  7.0270us     200ns  304.50us  cuDeviceGetAttribute
                    0.01%  602.70us         1  602.70us  602.70us  602.70us  cudaGetDeviceProperties
                    0.00%  470.60us         4  117.65us  25.200us  297.80us  cudaFree
                    0.00%  39.200us         1  39.200us  39.200us  39.200us  cuDeviceTotalMem
                    0.00%  26.800us         2  13.400us  5.1000us  21.700us  cudaMemset
                    0.00%  13.500us         1  13.500us  13.500us  13.500us  cuDeviceGetPCIBusId
                    0.00%  12.500us         1  12.500us  12.500us  12.500us  cudaSetDevice
                    0.00%  4.2000us         2  2.1000us     500ns  3.7000us  cuDeviceGet
                    0.00%  2.2000us         3     733ns     300ns  1.1000us  cuDeviceGetCount
                    0.00%  1.4000us         1  1.4000us  1.4000us  1.4000us  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetLuid
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetUuid

*/
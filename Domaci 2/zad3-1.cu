#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "../common/stopwatch.h"


__device__ bool prime=1;
__global__ void primeGPU(int N)
{
    if(N==1) {prime=0; return;}
    if(N==2 || N==3) { return;}

    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>1 && N%i==0 && i<=sqrt((float)N)) { prime=0; return;}
    
}

__global__ void primeGPUUnroll2(int N)
{
    if(N==1) {prime=0; return;}
    if(N==2 || N==3) { return;}

    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int j=i+blockDim.x;
    
    if(i>1 && N%i==0 && i<=sqrt((float)N)) { prime=0; return;}
    if(j>1 && N%j==0 && j<=sqrt((float)N)) { prime=0; return;}
}

__global__ void primeGPUUnroll4(int N)
{
    if(N==1) {prime=0; return;}
    if(N==2 || N==3) { return;}

    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int j=i+blockDim.x;
    const unsigned int k=j+blockDim.x;
    const unsigned int l=k+blockDim.x;

    if(i>1 && N%i==0 && i<=sqrt((float)N)) { prime=0; return;}
    if(j>1 && N%j==0 && j<=sqrt((float)N)) { prime=0; return;}
    if(k>1 && N%k==0 && k<=sqrt((float)N)) { prime=0; return;}
    if(l>1 && N%l==0 && l<=sqrt((float)N)) { prime=0; return;}
}

__global__ void eratostenGPU1(bool *p, int N, int i)
{
    const unsigned int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>1 && j<=(N/i) && p[i])
        p[i*j]=0;
}

__global__ void eratostenGPU2(bool *p, int N, int i)
{
    const unsigned int k=blockIdx.x*blockDim.x+threadIdx.x;
    int j=i*i+(k*i);
    if(j<N && p[i])
        p[j]=0;
}



bool primeCPU(int N)
{
    if(N==1) return 0;
    if(N==2) return 1;
    if(N==3) return 1;
    if(N%2==0) return 0;
    for(int i=3;i<=sqrt(N);i++)
    {
        if(N%i==0) return 0;
    }
    return 1;
}

void eratostenCPU1(bool *p,int N)
{
    p[0]=0;
    p[1]=0;
    for(int i=2;i<N;i++)
    {
        if(p[i])
        {
            for(int j=2;j<=N/i;j++)
                p[i*j]=0;
        }
    }
}
void eratostenCPU2(bool *p, int N)
{
    p[0]=0;
    p[1]=0;
    for(int i=2;i<=sqrt(N);i++)
    {
        if(p[i])
        {
            int j=i*i;
            int k=1;
            while(j<N)
            {
                p[j]=0;
                j=i*i+(k*i);
                k++;
            }
        }
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
    int N=83159;
    if (argc > 1) B = atoi(argv[1]);

    if (argc > 2) N = atoi(argv[2]);
   

  
    Stopwatch s;
    bool tmp=primeCPU(N);
    printf("\nprimeCPU %f sec \n",s.elapsed());
    dim3 block (B);
    dim3 grid ((N + block.x - 1) / block.x); 

    s.reset();
    primeGPU<<<grid,block>>>(N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("primeGPU<<<%d,%d>>> elapsed %f sec \n", grid.x, block.x, s.elapsed());
    bool tr=1;
    bool pr=0;
    CHECK(cudaMemcpyFromSymbol(&pr, prime, sizeof(bool)));
    if(tmp!=pr)
    printf("Bad\n");
    else printf("Good\n");
    cudaMemcpyToSymbol(prime, &tr, sizeof(bool));

    s.reset();
    primeGPUUnroll2<<<grid.x/2,block>>>(N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("primeGPUUnroll2<<<%d,%d>>> elapsed %f sec \n", grid.x/2, block.x, s.elapsed());
    CHECK(cudaMemcpyFromSymbol(&pr, prime, sizeof(bool)));
    if(tmp!=pr)
    printf("Bad\n");
    else printf("Good\n");
    cudaMemcpyToSymbol(prime, &tr, sizeof(bool));


    s.reset();
    primeGPUUnroll4<<<grid.x/4,block>>>(N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("primeGPUUnroll4<<<%d,%d>>> elapsed %f sec \n", grid.x/2, block.x, s.elapsed());
    CHECK(cudaMemcpyFromSymbol(&pr, prime, sizeof(bool)));
    if(tmp!=pr)
    printf("Bad\n");
    else printf("Good\n");
    cudaMemcpyToSymbol(prime, &tr, sizeof(bool));
   
  /*  for(int i=9999000;i<10000000;i++)
    {
    bool tmp=primeCPU(i);

    dim3 block (B);
    dim3 grid ((i + block.x * 2- 1) / (block.x*2));
    primeGPUUnroll4<<<grid.x/4,block>>>(i);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    pr=0;
    CHECK(cudaMemcpyFromSymbol(&pr, prime, sizeof(bool)));
    if(tmp!=pr)
    printf("Bad\n %d \n",i);

    cudaMemcpyToSymbol(prime, &tr, sizeof(bool));
    } 
*/
    int nBytes=N* sizeof(bool);
    bool *h_A, *h_B, *h_C;
    
    h_A = (bool *)malloc(nBytes);
    h_B = (bool *)malloc(nBytes); 
    h_C = (bool *)malloc(nBytes); 
    memset(h_A,1,nBytes);
    memset(h_B,1,nBytes);

    s.reset();
    eratostenCPU1(h_A,N);
    printf("\neratostenCPU1 elapsed %f sec \n",s.elapsed());

    s.reset();
    eratostenCPU2(h_B,N);
    printf("eratostenCPU2 elapsed %f sec \n",s.elapsed());

    for(int i=0;i<N;i++)
        if(h_A[i]!=h_B[i])
        {
            printf("CPU1 BAD i = %d \n",i);
            break;
        }
    memset(h_B,0,nBytes);
    s.reset();
    for(int i=2;i<N;i++)
        {
            if(primeCPU(i))
                h_B[i]=1;
        }
    printf("primeCPU for all numbers to N elapsed %f sec \n",s.elapsed());

    for(int i=0;i<N;i++)
        if(h_A[i]!=h_B[i])
        {
            printf("CPU2 BAD i = %d \n",i);
            break;
        }
    memset(h_B,0,nBytes);
    s.reset();
    for(int i=2;i<N;i++)
    {
        pr=0;
        primeGPU<<<grid,block>>>(i);
        CHECK(cudaStreamSynchronize(0));
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpyFromSymbol(&pr, prime, sizeof(bool)));
        cudaMemcpyToSymbol(prime, &tr, sizeof(bool));
        h_B[i]=pr;
    }
 
    printf("primeGPU for all numbers to N elapsed %f sec \n",s.elapsed());
    for(int i=0;i<N;i++)
        if(h_A[i]!=h_B[i])
        {
            printf("CPU and GPU BAD i = %d \n",i);
            break;
        }

   

    bool *d_A;
    cudaMalloc((bool**)&d_A, nBytes);
    cudaMemset(d_A,1,nBytes);
    cudaMemset(d_A,0,2*sizeof(bool));
    s.reset();
    for(int i=2;i<N;i++)
       {
        eratostenGPU1<<<grid,block>>>(d_A,N,i);
        CHECK(cudaStreamSynchronize(0));
        CHECK(cudaGetLastError());
       }
    
    printf("eratostenGPU1<<<%d,%d>>> elapsed %f sec \n", grid.x/2, block.x, s.elapsed());
    CHECK(cudaMemcpy(h_C, d_A, nBytes, cudaMemcpyDeviceToHost));
    for(int i=0;i<N;i++)
    if(h_A[i]!=h_C[i])
    {
        printf("GPU1 BAD i = %d \n",i);
        break;
    }
   
    cudaMemset(d_A,1,nBytes);
    cudaMemset(d_A,0,2*sizeof(bool));
    s.reset();
    for(int i=2;i<=sqrt(N);i++)
       {
        eratostenGPU2<<<grid,block>>>(d_A,N,i);
        CHECK(cudaStreamSynchronize(0));
        CHECK(cudaGetLastError());
       }
     
    printf("eratostenGPU2<<<%d,%d>>> elapsed %f sec \n", grid.x/2, block.x, s.elapsed());
    CHECK(cudaMemcpy(h_C, d_A, nBytes, cudaMemcpyDeviceToHost));
    for(int i=0;i<N;i++)
    if(h_A[i]!=h_C[i])
    {
        printf("GPU1 BAD i = %d \n",i);
        break;
    }
   
    cudaFree(d_A);
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}


/*


 
a starting main at device 0: GeForce GTX 1050

primeCPU 0.000003 sec
primeGPU<<<2599,32>>> elapsed 0.359991 sec
Good
primeGPUUnroll2<<<1299,32>>> elapsed 0.000256 sec
Good
primeGPUUnroll4<<<1299,32>>> elapsed 0.000176 sec
Good

eratostenCPU1 elapsed 0.001477 sec
eratostenCPU2 elapsed 0.000735 sec
primeCPU for all numbers to N elapsed 0.039269 sec
primeGPU for all numbers to N elapsed 29.132383 sec
eratostenGPU1<<<1299,32>>> elapsed 9.663573 sec
eratostenGPU2<<<1299,32>>> elapsed 0.032215 sec
==8660== Profiling application: a 32
==8660== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.88%  1.45163s     83158  17.456us  12.800us  231.36us  primeGPU(int)
                   45.62%  1.32778s     83157  15.967us  15.680us  18.336us  eratostenGPU1(bool*, int, int)
                    2.37%  68.885ms     83162     828ns     351ns  13.696us  [CUDA memcpy DtoH]
                    2.00%  58.342ms     83160     701ns     671ns  1.3440us  [CUDA memcpy HtoD]
                    0.13%  3.8509ms       287  13.417us  13.184us  15.456us  eratostenGPU2(bool*, int, int)
                    0.00%  11.455us         1  11.455us  11.455us  11.455us  primeGPUUnroll2(int)
                    0.00%  8.6720us         1  8.6720us  8.6720us  8.6720us  primeGPUUnroll4(int)
                    0.00%  2.8800us         4     720ns     640ns     768ns  [CUDA memset]
      API calls:   47.49%  18.3736s    166601  110.28us  25.400us  39.870ms  cudaStreamSynchronize
                   34.53%  13.3617s     83160  160.67us  92.700us  51.580ms  cudaMemcpyFromSymbol
                    9.06%  3.50415s    166604  21.032us  13.500us  359.82ms  cudaLaunchKernel
                    8.57%  3.31517s     83160  39.864us  26.300us  16.170ms  cudaMemcpyToSymbol
                    0.17%  66.694ms    166604     400ns     200ns  325.00us  cudaGetLastError
                    0.17%  66.042ms         1  66.042ms  66.042ms  66.042ms  cudaDeviceReset
                    0.00%  1.3106ms         1  1.3106ms  1.3106ms  1.3106ms  cudaMalloc
                    0.00%  816.10us         2  408.05us  190.90us  625.20us  cudaMemcpy
                    0.00%  716.70us        97  7.3880us     200ns  340.30us  cuDeviceGetAttribute
                    0.00%  615.90us         1  615.90us  615.90us  615.90us  cudaGetDeviceProperties
                    0.00%  489.70us         3  163.23us  141.30us  206.00us  cudaDeviceSynchronize
                    0.00%  437.40us         1  437.40us  437.40us  437.40us  cudaFree
                    0.00%  187.20us         4  46.800us  12.200us  97.800us  cudaMemset
                    0.00%  44.000us         1  44.000us  44.000us  44.000us  cuDeviceTotalMem
                    0.00%  14.800us         1  14.800us  14.800us  14.800us  cudaSetDevice
                    0.00%  13.600us         1  13.600us  13.600us  13.600us  cuDeviceGetPCIBusId
                    0.00%  4.5000us         2  2.2500us     400ns  4.1000us  cuDeviceGet
                    0.00%  2.5000us         3     833ns     400ns  1.3000us  cuDeviceGetCount
                    0.00%  1.3000us         1  1.3000us  1.3000us  1.3000us  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetUuid
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetLuid


                    


a starting main at device 0: GeForce GTX 1050

primeCPU 0.000003 sec
primeGPU<<<325,256>>> elapsed 0.222129 sec
Good
primeGPUUnroll2<<<162,256>>> elapsed 0.000191 sec
Good
primeGPUUnroll4<<<162,256>>> elapsed 0.000163 sec
Good

eratostenCPU1 elapsed 0.001537 sec
eratostenCPU2 elapsed 0.000803 sec
primeCPU for all numbers to N elapsed 0.038325 sec
primeGPU for all numbers to N elapsed 29.891912 sec
eratostenGPU1<<<162,256>>> elapsed 11.910505 sec
eratostenGPU2<<<162,256>>> elapsed 0.038316 sec
==384== Profiling application: a 256
==384== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   50.24%  713.70ms     83158  8.5820us  3.6800us  9.9840us  primeGPU(int)
                   40.70%  578.16ms     83157  6.9520us  6.8790us  8.6400us  eratostenGPU1(bool*, int, int)
                    4.87%  69.151ms     83162     831ns     351ns  13.855us  [CUDA memcpy DtoH]
                    4.11%  58.334ms     83160     701ns     671ns  2.6560us  [CUDA memcpy HtoD]
                    0.08%  1.1224ms       287  3.9100us  3.8390us  5.3760us  eratostenGPU2(bool*, int, int)
                    0.00%  6.8480us         1  6.8480us  6.8480us  6.8480us  primeGPUUnroll2(int)
                    0.00%  5.9200us         1  5.9200us  5.9200us  5.9200us  primeGPUUnroll4(int)
                    0.00%  3.5520us         4     888ns     736ns  1.2160us  [CUDA memset]
      API calls:   48.30%  20.0218s    166601  120.18us  23.900us  79.108ms  cudaStreamSynchronize
                   33.99%  14.0888s     83160  169.42us  92.500us  68.523ms  cudaMemcpyFromSymbol
                    9.54%  3.95403s    166604  23.733us  13.600us  221.96ms  cudaLaunchKernel
                    7.78%  3.22361s     83160  38.763us  25.900us  38.369ms  cudaMemcpyToSymbol
                    0.21%  89.002ms    166604     534ns     200ns  7.2098ms  cudaGetLastError
                    0.14%  59.873ms         1  59.873ms  59.873ms  59.873ms  cudaDeviceReset
                    0.02%  8.7402ms         1  8.7402ms  8.7402ms  8.7402ms  cudaMalloc
                    0.00%  1.6014ms         2  800.70us  391.00us  1.2104ms  cudaMemcpy
                    0.00%  685.60us        97  7.0680us     200ns  312.20us  cuDeviceGetAttribute
                    0.00%  604.50us         1  604.50us  604.50us  604.50us  cudaGetDeviceProperties
                    0.00%  540.30us         1  540.30us  540.30us  540.30us  cudaFree
                    0.00%  447.30us         3  149.10us  137.90us  156.90us  cudaDeviceSynchronize
                    0.00%  235.50us         4  58.875us  10.900us  150.60us  cudaMemset
                    0.00%  44.600us         1  44.600us  44.600us  44.600us  cuDeviceTotalMem
                    0.00%  14.500us         1  14.500us  14.500us  14.500us  cudaSetDevice
                    0.00%  13.800us         1  13.800us  13.800us  13.800us  cuDeviceGetPCIBusId
                    0.00%  4.7000us         2  2.3500us     400ns  4.3000us  cuDeviceGet
                    0.00%  2.2000us         3     733ns     300ns  1.1000us  cuDeviceGetCount
                    0.00%  1.4000us         1  1.4000us  1.4000us  1.4000us  cuDeviceGetName
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetLuid
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetUuid




                    a starting main at device 0: GeForce GTX 1050

primeCPU 0.000003 sec
primeGPU<<<163,512>>> elapsed 0.250712 sec
Good
primeGPUUnroll2<<<81,512>>> elapsed 0.000285 sec
Good
primeGPUUnroll4<<<81,512>>> elapsed 0.000119 sec
Good

eratostenCPU1 elapsed 0.001655 sec
eratostenCPU2 elapsed 0.000893 sec
primeCPU for all numbers to N elapsed 0.044534 sec
primeGPU for all numbers to N elapsed 27.361498 sec
eratostenGPU1<<<81,512>>> elapsed 9.183002 sec
eratostenGPU2<<<81,512>>> elapsed 0.028100 sec
==2444== Profiling application: a
==2444== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   50.15%  724.30ms     83158  8.7090us  3.9040us  10.208us  primeGPU(int)
                   40.98%  591.82ms     83157  7.1160us  7.0070us  8.6400us  eratostenGPU1(bool*, int, int)
                    4.77%  68.902ms     83162     828ns     351ns  15.424us  [CUDA memcpy DtoH]
                    4.02%  58.090ms     83160     698ns     671ns  4.2560us  [CUDA memcpy HtoD]
                    0.08%  1.1962ms       287  4.1670us  4.0640us  5.5360us  eratostenGPU2(bool*, int, int)
                    0.00%  7.0720us         1  7.0720us  7.0720us  7.0720us  primeGPUUnroll2(int)
                    0.00%  5.6320us         1  5.6320us  5.6320us  5.6320us  primeGPUUnroll4(int)
                    0.00%  2.8160us         4     704ns     672ns     768ns  [CUDA memset]
      API calls:   46.63%  16.9469s    166601  101.72us  24.200us  68.883ms  cudaStreamSynchronize
                   35.11%  12.7603s     83160  153.44us  92.300us  39.068ms  cudaMemcpyFromSymbol
                    9.12%  3.31491s    166604  19.896us  13.500us  250.56ms  cudaLaunchKernel
                    8.74%  3.17733s     83160  38.207us  26.100us  15.787ms  cudaMemcpyToSymbol
                    0.20%  73.581ms    166604     441ns     200ns  2.3302ms  cudaGetLastError
                    0.18%  65.247ms         1  65.247ms  65.247ms  65.247ms  cudaDeviceReset
                    0.00%  1.3568ms        97  13.987us     200ns  729.10us  cuDeviceGetAttribute
                    0.00%  1.2735ms         1  1.2735ms  1.2735ms  1.2735ms  cudaMalloc
                    0.00%  790.30us         1  790.30us  790.30us  790.30us  cudaGetDeviceProperties
                    0.00%  522.60us         1  522.60us  522.60us  522.60us  cudaFree
                    0.00%  430.10us         2  215.05us  182.40us  247.70us  cudaMemcpy
                    0.00%  365.20us         3  121.73us  90.100us  140.30us  cudaDeviceSynchronize
                    0.00%  285.20us         4  71.300us  13.100us  150.30us  cudaMemset
                    0.00%  44.600us         1  44.600us  44.600us  44.600us  cuDeviceTotalMem
                    0.00%  19.300us         1  19.300us  19.300us  19.300us  cudaSetDevice
                    0.00%  13.800us         1  13.800us  13.800us  13.800us  cuDeviceGetPCIBusId
                    0.00%  4.3000us         2  2.1500us     500ns  3.8000us  cuDeviceGet
                    0.00%  2.7000us         3     900ns     400ns  1.4000us  cuDeviceGetCount
                    0.00%  1.3000us         1  1.3000us  1.3000us  1.3000us  cuDeviceGetName
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetLuid
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetUuid

*/
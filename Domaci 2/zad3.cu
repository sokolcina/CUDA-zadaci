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
    int N=831595;
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
   
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

/*

a starting main at device 0: GeForce GTX 1050

primeCPU 0.000003 sec
==8896== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==8896== primeGPU<<<163,512>>> elapsed 0.285190 sec
Replaying kernel "primeGPU(int)" (done)
Good
primeGPUUnroll2<<<81,512>>> elapsed 0.056494 sec
==8896== Replaying kernel "primeGPUUnroll2(int)" (done)
Good
==8896== primeGPUUnroll4<<<81,512>>> elapsed 0.057359 sec
Replaying kernel "primeGPUUnroll4(int)" (done)
Good
==8896== Profiling application: a
==8896== Profiling result:
==8896== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "GeForce GTX 1050 (0)"
    Kernel: primeGPUUnroll4(int)
          1                      dram_read_throughput                  Device Memory Read Throughput  752.26MB/s  752.26MB/s  752.26MB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  77.653MB/s  77.653MB/s  77.653MB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
    Kernel: primeGPUUnroll2(int)
          1                      dram_read_throughput                  Device Memory Read Throughput  377.32MB/s  377.32MB/s  377.32MB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  49.757MB/s  49.757MB/s  49.757MB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
    Kernel: primeGPU(int)
          1                      dram_read_throughput                  Device Memory Read Throughput  28.435MB/s  28.435MB/s  28.435MB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  14.217MB/s  14.217MB/s  14.217MB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000






          a starting main at device 0: GeForce GTX 1050

primeCPU 0.000003 sec
==7680== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==7680== primeGPU<<<2599,32>>> elapsed 0.264729 sec
Replaying kernel "primeGPU(int)" (done)
Good
==7680== primeGPUUnroll2<<<1299,32>>> elapsed 0.057661 sec
Replaying kernel "primeGPUUnroll2(int)" (done)
Good
primeGPUUnroll4<<<1299,32>>> elapsed 0.058291 sec
==7680== Replaying kernel "primeGPUUnroll4(int)" (done)
Good
==7680== Profiling application: a 32
==7680== Profiling result:
==7680== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "GeForce GTX 1050 (0)"
    Kernel: primeGPUUnroll4(int)
          1                      dram_read_throughput                  Device Memory Read Throughput  443.02MB/s  443.02MB/s  443.02MB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  67.636MB/s  67.636MB/s  67.636MB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
    Kernel: primeGPUUnroll2(int)
          1                      dram_read_throughput                  Device Memory Read Throughput  263.91MB/s  263.91MB/s  263.91MB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  42.652MB/s  42.652MB/s  42.652MB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
    Kernel: primeGPU(int)
          1                      dram_read_throughput                  Device Memory Read Throughput  28.575MB/s  28.575MB/s  28.574MB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  7.1436MB/s  7.1436MB/s  7.1436MB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000





a starting main at device 0: GeForce GTX 1050

primeCPU 0.000003 sec
==4824== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==4824== primeGPU<<<325,256>>> elapsed 0.267788 sec
Replaying kernel "primeGPU(int)" (done)
Good
==4824== primeGPUUnroll2<<<162,256>>> elapsed 0.056880 sec
Replaying kernel "primeGPUUnroll2(int)" (done)
Good
==4824== primeGPUUnroll4<<<162,256>>> elapsed 0.059133 sec
Replaying kernel "primeGPUUnroll4(int)" (done)
Good
==4824== Profiling application: a 256
==4824== Profiling result:
==4824== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "GeForce GTX 1050 (0)"
    Kernel: primeGPUUnroll4(int)
          1                      dram_read_throughput                  Device Memory Read Throughput  506.22MB/s  506.22MB/s  506.22MB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  54.237MB/s  54.237MB/s  54.237MB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
    Kernel: primeGPUUnroll2(int)
          1                      dram_read_throughput                  Device Memory Read Throughput  413.46MB/s  413.46MB/s  413.46MB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
    Kernel: primeGPU(int)
          1                      dram_read_throughput                  Device Memory Read Throughput  291.45MB/s  291.45MB/s  291.45MB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  36.893MB/s  36.893MB/s  36.893MB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000

          */


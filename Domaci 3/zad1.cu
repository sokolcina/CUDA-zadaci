#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "../common/stopwatch.h"

#define NSTREAM 4



void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)( rand() & 0xFF ) / 10.0f; //100.0f;
    }

    return;
}


__global__ void a1  (float *A,float *B,float *C, const int ncols,const int nrows)
{
    const unsigned int j=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int i=blockIdx.y*blockDim.y+threadIdx.y;
    const unsigned int id=ncols*i+j;
    const unsigned int N = nrows * ncols;
    if(id<N)
        C[id]=A[id]+B[id];
}

__global__ void a2  (float *A,float *B,float *C, const int ncols,const int nrows)
{
    const unsigned int j=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int i=blockIdx.y*blockDim.y+threadIdx.y;
    const unsigned int id=ncols*i+j;
    const unsigned int N = nrows * ncols;
    if(id<N)
        C[id]=A[id]+B[id];
}

__global__ void s1  (float *A,float *B,float *C, const int ncols,const int nrows)
{
    const unsigned int j=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int i=blockIdx.y*blockDim.y+threadIdx.y;
    const unsigned int id=ncols*i+j;
    const unsigned int N = nrows * ncols;
    if(id<N)
        C[id]=A[id]-B[id];
}
__global__ void s2  (float *A,float *B,float *C, const int ncols,const int nrows)
{
    const unsigned int j=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int i=blockIdx.y*blockDim.y+threadIdx.y;
    const unsigned int id=ncols*i+j;
    const unsigned int N = nrows * ncols;
    if(id<N)
        C[id]=A[id]-B[id];
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
    

   
    int nx = 1 << 12;
    int ny = 1 << 12;

   
    int blockx = 16;
    int blocky = 16;

    if (argc > 1) blockx = atoi(argv[1]);

    if (argc > 2) blocky  = atoi(argv[2]);

    if (argc > 3) nx  = atoi(argv[3]);

    if (argc > 4) ny  = atoi(argv[4]);


    size_t nBytes = nx * ny * sizeof(float);

    int N = nx * ny;
    // execution configuration
    dim3 block (blockx, blocky);
    dim3 grid  ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *h_res1 = (float *)malloc(nBytes);
    float *h_res2  = (float *)malloc(nBytes);
    float *h_res3  = (float *)malloc(nBytes);
    float *h_res4  = (float *)malloc(nBytes);

    float *d_A1, *d_A2, *d_A3, *d_A4;
    float *d_B1, *d_B2, *d_B3, *d_B4;
    float *d_C1, *d_C2, *d_C3, *d_C4;

    CHECK(cudaMalloc((void **)&d_A1, nBytes));
    CHECK(cudaMalloc((void **)&d_A2, nBytes));
    CHECK(cudaMalloc((void **)&d_A3, nBytes));
    CHECK(cudaMalloc((void **)&d_A4, nBytes));

    CHECK(cudaMalloc((void **)&d_B1, nBytes));
    CHECK(cudaMalloc((void **)&d_B2, nBytes));
    CHECK(cudaMalloc((void **)&d_B3, nBytes));
    CHECK(cudaMalloc((void **)&d_B4, nBytes));

    CHECK(cudaMalloc((void **)&d_C1, nBytes));
    CHECK(cudaMalloc((void **)&d_C2, nBytes));
    CHECK(cudaMalloc((void **)&d_C3, nBytes));
    CHECK(cudaMalloc((void **)&d_C4, nBytes));

    initialData(h_A, N);
    initialData(h_B, N);


    CHECK(cudaMemcpy(d_A1, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_A2, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_A3, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_A4, h_A, nBytes, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(d_B1, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B2, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B3, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B4, h_B, nBytes, cudaMemcpyHostToDevice));
    // creat events
    cudaEvent_t start, stop, e_A,e_S;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    Stopwatch s;
    // record start event
    CHECK(cudaEventRecord(start, 0));

    a1<<<grid,block>>>(d_A1,d_B1,d_C1,nx,ny);
    a2<<<grid,block>>>(d_A2,d_B2,d_C2,nx,ny);
    s1<<<grid,block>>>(d_A3,d_B3,d_C3,nx,ny);
    s2<<<grid,block>>>(d_A4,d_B4,d_C4,nx,ny);

    // record stop event
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    float elapsed_time;
    // calculate elapsed time
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for sequence execution = %f\nstopwatch = %f\n",
           elapsed_time / 1000.0f,s.elapsed());


    CHECK(cudaMemcpy(h_res1, d_C1, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_res2, d_C2, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_res3, d_C3, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_res4, d_C4, nBytes, cudaMemcpyDeviceToHost));
    for(int i=0;i<N;i++)
    {
        if(h_res1[i]!=h_res2[i] || h_res3[i]!=h_res4[i])
            printf("BAD\n");
    }
    // Allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *) malloc(NSTREAM * sizeof(
        cudaStream_t));

    for (int i = 0 ; i < NSTREAM ; i++)
    {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

     // record start event
     s.reset();
     CHECK(cudaEventRecord(start, 0));
   
     a1<<<grid,block,0,streams[0]>>>(d_A1,d_B1,d_C1,nx,ny);
     a2<<<grid,block,0,streams[1]>>>(d_A2,d_B2,d_C2,nx,ny);
     s1<<<grid,block,0,streams[2]>>>(d_A3,d_B3,d_C3,nx,ny);
     s2<<<grid,block,0,streams[3]>>>(d_A4,d_B4,d_C4,nx,ny);


     // record stop event
     CHECK(cudaEventRecord(stop, 0));
     CHECK(cudaEventSynchronize(stop));
 
     // calculate elapsed time
     CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
     printf("Measured time for parallel execution = %f\nstopwatch = %f\n",
            elapsed_time / 1000.0f,s.elapsed());
 
    CHECK(cudaMemcpy(h_res1, d_C1, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_res2, d_C2, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_res3, d_C3, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_res4, d_C4, nBytes, cudaMemcpyDeviceToHost));
        for(int i=0;i<N;i++)
            {
                if(h_res1[i]!=h_res2[i] || h_res3[i]!=h_res4[i])
                    printf("BAD\n");
            }

    CHECK(cudaEventCreateWithFlags(&e_A,cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&e_S,cudaEventDisableTiming));
    
    // record start event
    s.reset();
    CHECK(cudaEventRecord(start, 0));
    
    a1<<<grid,block,0,streams[0]>>>(d_A1,d_B1,d_C1,nx,ny);
    CHECK(cudaEventRecord(e_A, 0));
    CHECK(cudaStreamWaitEvent(streams[0], e_A, 0));
    a2<<<grid,block,0,streams[1]>>>(d_A2,d_B2,d_C2,nx,ny);
    
    
    s1<<<grid,block,0,streams[2]>>>(d_A3,d_B3,d_C3,nx,ny);
    CHECK(cudaEventRecord(e_S, 0));
    CHECK(cudaStreamWaitEvent(streams[2], e_S, 0));
    s2<<<grid,block,0,streams[3]>>>(d_A4,d_B4,d_C4,nx,ny);
    

    // record stop event
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // calculate elapsed time
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution with events = %f\nstopwatch = %f\n",
           elapsed_time / 1000.0f,s.elapsed());


    CHECK(cudaMemcpy(h_res1, d_C1, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_res2, d_C2, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_res3, d_C3, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_res4, d_C4, nBytes, cudaMemcpyDeviceToHost));
    for(int i=0;i<N;i++)
    {
        if(h_res1[i]!=h_res2[i] || h_res3[i]!=h_res4[i])
            printf("BAD\n");
    }
     // release all stream
     for (int i = 0 ; i < NSTREAM ; i++)
     {
         CHECK(cudaStreamDestroy(streams[i]));
     }
 
     free(streams);
    

    // initialize host array
  
    free(h_A);
    free(h_B);
    free(h_res1);
    free(h_res2);
    free(h_res3);
    free(h_res4);
    CHECK(cudaFree(d_A1));CHECK(cudaFree(d_A2));CHECK(cudaFree(d_A3));CHECK(cudaFree(d_A4));
    CHECK(cudaFree(d_B1));CHECK(cudaFree(d_B2));CHECK(cudaFree(d_B3));CHECK(cudaFree(d_B4));
    CHECK(cudaFree(d_C1));CHECK(cudaFree(d_C2));CHECK(cudaFree(d_C3));CHECK(cudaFree(d_C4));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    
    CHECK(cudaEventDestroy(e_A));
    CHECK(cudaEventDestroy(e_S));
    
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

/* 

a starting main at device 0: GeForce GTX 1050
Measured time for sequence execution = 0.008140
stopwatch = 0.008676
Measured time for parallel execution = 0.008136
stopwatch = 0.008441
Measured time for parallel execution with events = 0.008160
stopwatch = 0.008459

*/
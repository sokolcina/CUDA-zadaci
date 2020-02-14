#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "../common/stopwatch.h"
/*
 * Various memory access pattern optimizations applied to a matrix transpose
 * kernel.
 */

#define BDIMX 16
#define BDIMY 16

void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)( rand() & 0xFF ) / 10.0f; //100.0f;
    }

    return;
}

void printData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%dth element: %f\n", i, in[i]);
    }

    return;
}

void checkResult(float *h_out, float *g_out, const int size, int showme)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < size; i++)
    {
        if (abs(h_out[i] - g_out[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, h_out[i],
                    g_out[i]);
            break;
        }

        if (showme && i > size / 2 && i < size / 2 + 5)
        {
            // printf("%dth element: host %f gpu %f\n",i,h_out[i],g_out[i]);
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
   
}

void transposeHost(float *out, float *in, const int nx, const int ny)
{
    for( int iy = 0; iy < ny; ++iy)
    {
        for( int ix = 0; ix < nx; ++ix)
        {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}



__global__ void transposeNaiveRow(float *out, float *in, const int nx,
                                  const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeNaiveCol(float *out, float *in, const int nx,
                                  const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}


__global__ void transposeUnroll4Row(float *out, float *in, const int nx,
                                    const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        out[to]                   = in[ti];
        out[to + ny * blockDim.x]   = in[ti + blockDim.x];
        out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
    }
}

__global__ void transposeUnroll4Col(float *out, float *in, const int nx,
                                    const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        out[ti]                = in[to];
        out[ti +   blockDim.x] = in[to +   blockDim.x * ny];
        out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
        out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
    }
}


__global__ void transposeDiagonalRow(float *out, float *in, const int nx,
                                     const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}


__global__ void transposeDiagonalCol(float *out, float *in, const int nx,
                                     const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

// main functions
int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

   
    int nx = 1 << 13;
    int ny = 1 << 13;

   
    int blockx = 16;
    int blocky = 16;

    if (argc > 1) blockx = atoi(argv[1]);

    if (argc > 2) blocky  = atoi(argv[2]);

    if (argc > 3) nx  = atoi(argv[3]);

    if (argc > 4) ny  = atoi(argv[4]);


    size_t nBytes = nx * ny * sizeof(float);

    // execution configuration
    dim3 block (blockx, blocky);
    dim3 grid  ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *h_out = (float *)malloc(nBytes);
    float *g_out  = (float *)malloc(nBytes);

    // initialize host array
    initialData(h_A, nx * ny);

    Stopwatch s;
    transposeHost(h_out, h_A, nx, ny);
    printf("\ntransposeHost elapsed %f sec \n",s.elapsed());
    // allocate device memory
    float *d_A, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    s.reset();
    transposeNaiveRow<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("transposeNaiveRow<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
           block.y, s.elapsed());
    
    CHECK(cudaMemcpy(g_out, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, g_out, nx * ny, 1);
       

    s.reset();
    transposeNaiveCol<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("transposeNaiveCol<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y, s.elapsed());

    CHECK(cudaMemcpy(g_out, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, g_out, nx * ny, 1);
   
    
    s.reset();
    transposeDiagonalRow<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("transposeDiagonalRow<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y,s.elapsed());

    CHECK(cudaMemcpy(g_out, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, g_out, nx * ny, 1);

  
    s.reset();
    transposeDiagonalCol<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("transposeDiagonalCol<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y,s.elapsed());

    CHECK(cudaMemcpy(g_out, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, g_out, nx * ny, 1);

    grid.x = (nx + block.x * 4 - 1) / (block.x * 4);

    s.reset();
    transposeUnroll4Row<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("transposeUnroll4Row<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y,s.elapsed());

    CHECK(cudaMemcpy(g_out, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, g_out, nx * ny, 1);

    s.reset();
    transposeUnroll4Col<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("transposeUnroll4Col<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y,s.elapsed());

    CHECK(cudaMemcpy(g_out, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out, g_out, nx * ny, 1);


    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_out);
    free(g_out);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}


/* 
 GeForce GTX 1050



transposeHost elapsed 1.477946 sec
transposeNaiveRow<<<grid (1024,1024) block (8,8)>>> elapsed 0.014048 sec
transposeNaiveCol<<<grid (1024,1024) block (8,8)>>> elapsed 0.013361 sec
transposeDiagonalRow<<<grid (1024,1024) block (8,8)>>> elapsed 0.022513 sec
transposeDiagonalCol<<<grid (1024,1024) block (8,8)>>> elapsed 0.022696 sec
transposeUnroll4Row<<<grid (256,1024) block (8,8)>>> elapsed 0.017934 sec
transposeUnroll4Col<<<grid (256,1024) block (8,8)>>> elapsed 0.013618 sec
==8728== Profiling application: a 8 8
==8728== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   77.48%  659.48ms         6  109.91ms  93.765ms  170.35ms  [CUDA memcpy DtoH]
                   10.50%  89.350ms         1  89.350ms  89.350ms  89.350ms  [CUDA memcpy HtoD]
                    2.63%  22.410ms         1  22.410ms  22.410ms  22.410ms  transposeDiagonalCol(float*, float*, int, int)
                    2.62%  22.277ms         1  22.277ms  22.277ms  22.277ms  transposeDiagonalRow(float*, float*, int, int)
                    2.08%  17.695ms         1  17.695ms  17.695ms  17.695ms  transposeUnroll4Row(float*, float*, int, int)
                    1.59%  13.513ms         1  13.513ms  13.513ms  13.513ms  transposeNaiveRow(float*, float*, int, int)
                    1.56%  13.305ms         1  13.305ms  13.305ms  13.305ms  transposeUnroll4Col(float*, float*, int, int)
                    1.54%  13.131ms         1  13.131ms  13.131ms  13.131ms  transposeNaiveCol(float*, float*, int, int)
      API calls:   44.86%  753.60ms         7  107.66ms  89.485ms  171.39ms  cudaMemcpy
                   44.66%  750.22ms         2  375.11ms  10.407ms  739.81ms  cudaMalloc
                    6.17%  103.68ms         6  17.280ms  13.284ms  22.618ms  cudaDeviceSynchronize
                    3.69%  61.997ms         1  61.997ms  61.997ms  61.997ms  cudaDeviceReset
                    0.49%  8.1968ms         2  4.0984ms  2.2642ms  5.9326ms  cudaFree
                    0.04%  687.60us        97  7.0880us     200ns  313.00us  cuDeviceGetAttribute
                    0.04%  595.80us         1  595.80us  595.80us  595.80us  cudaGetDeviceProperties
                    0.02%  405.20us         6  67.533us  61.800us  86.500us  cudaLaunchKernel
                    0.02%  383.80us         1  383.80us  383.80us  383.80us  cuDeviceGetPCIBusId
                    0.00%  45.500us         1  45.500us  45.500us  45.500us  cuDeviceTotalMem
                    0.00%  19.200us         1  19.200us  19.200us  19.200us  cudaSetDevice
                    0.00%  6.8000us         2  3.4000us     400ns  6.4000us  cuDeviceGet
                    0.00%  4.6000us         6     766ns     700ns     900ns  cudaGetLastError
                    0.00%  2.8000us         3     933ns     300ns  1.5000us  cuDeviceGetCount
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetLuid
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetUuid




transposeHost elapsed 1.476614 sec
transposeNaiveRow<<<grid (512,512) block (16,16)>>> elapsed 0.019974 sec
transposeNaiveCol<<<grid (512,512) block (16,16)>>> elapsed 0.010157 sec
transposeDiagonalRow<<<grid (512,512) block (16,16)>>> elapsed 0.019736 sec
transposeDiagonalCol<<<grid (512,512) block (16,16)>>> elapsed 0.012333 sec
transposeUnroll4Row<<<grid (128,512) block (16,16)>>> elapsed 0.019680 sec
transposeUnroll4Col<<<grid (128,512) block (16,16)>>> elapsed 0.010632 sec
==9876== Profiling application: a
==9876== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   78.02%  641.94ms         6  106.99ms  92.389ms  169.72ms  [CUDA memcpy DtoH]
                   10.97%  90.253ms         1  90.253ms  90.253ms  90.253ms  [CUDA memcpy HtoD]
                    2.37%  19.461ms         1  19.461ms  19.461ms  19.461ms  transposeNaiveRow(float*, float*, int, int)
                    2.36%  19.451ms         1  19.451ms  19.451ms  19.451ms  transposeUnroll4Row(float*, float*, int, int)
                    2.36%  19.449ms         1  19.449ms  19.449ms  19.449ms  transposeDiagonalRow(float*, float*, int, int)
                    1.47%  12.078ms         1  12.078ms  12.078ms  12.078ms  transposeDiagonalCol(float*, float*, int, int)
                    1.26%  10.334ms         1  10.334ms  10.334ms  10.334ms  transposeUnroll4Col(float*, float*, int, int)
                    1.20%  9.8546ms         1  9.8546ms  9.8546ms  9.8546ms  transposeNaiveCol(float*, float*, int, int)
      API calls:   65.58%  736.00ms         7  105.14ms  90.463ms  170.88ms  cudaMemcpy
                   20.09%  225.46ms         2  112.73ms  10.263ms  215.19ms  cudaMalloc
                    8.20%  92.006ms         6  15.334ms  10.065ms  19.877ms  cudaDeviceSynchronize
                    5.20%  58.380ms         1  58.380ms  58.380ms  58.380ms  cudaDeviceReset
                    0.77%  8.6344ms         2  4.3172ms  2.5144ms  6.1200ms  cudaFree
                    0.06%  683.90us        97  7.0500us     200ns  310.10us  cuDeviceGetAttribute
                    0.06%  620.30us         1  620.30us  620.30us  620.30us  cudaGetDeviceProperties
                    0.04%  414.00us         6  69.000us  62.100us  87.100us  cudaLaunchKernel
                    0.00%  38.900us         1  38.900us  38.900us  38.900us  cuDeviceTotalMem
                    0.00%  13.800us         1  13.800us  13.800us  13.800us  cuDeviceGetPCIBusId
                    0.00%  12.800us         1  12.800us  12.800us  12.800us  cudaSetDevice
                    0.00%  7.6000us         2  3.8000us     500ns  7.1000us  cuDeviceGet
                    0.00%  4.5000us         6     750ns     600ns     900ns  cudaGetLastError
                    0.00%  2.6000us         3     866ns     400ns  1.2000us  cuDeviceGetCount
                    0.00%  1.4000us         1  1.4000us  1.4000us  1.4000us  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetLuid
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetUuid



                    transposeHost elapsed 1.568921 sec
transposeNaiveRow<<<grid (256,256) block (32,32)>>> elapsed 0.020485 sec
transposeNaiveCol<<<grid (256,256) block (32,32)>>> elapsed 0.016247 sec
transposeDiagonalRow<<<grid (256,256) block (32,32)>>> elapsed 0.020285 sec
transposeDiagonalCol<<<grid (256,256) block (32,32)>>> elapsed 0.016534 sec
transposeUnroll4Row<<<grid (64,256) block (32,32)>>> elapsed 0.019553 sec
transposeUnroll4Col<<<grid (64,256) block (32,32)>>> elapsed 0.014690 sec
==3396== Profiling application: a 32 32
==3396== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   76.42%  635.41ms         6  105.90ms  92.661ms  168.80ms  [CUDA memcpy DtoH]
                   10.84%  90.111ms         1  90.111ms  90.111ms  90.111ms  [CUDA memcpy HtoD]
                    2.40%  19.994ms         1  19.994ms  19.994ms  19.994ms  transposeDiagonalRow(float*, float*, int, int)
                    2.40%  19.976ms         1  19.976ms  19.976ms  19.976ms  transposeNaiveRow(float*, float*, int, int)
                    2.32%  19.259ms         1  19.259ms  19.259ms  19.259ms  transposeUnroll4Row(float*, float*, int, int)
                    1.96%  16.262ms         1  16.262ms  16.262ms  16.262ms  transposeDiagonalCol(float*, float*, int, int)
                    1.93%  16.008ms         1  16.008ms  16.008ms  16.008ms  transposeNaiveCol(float*, float*, int, int)
                    1.73%  14.400ms         1  14.400ms  14.400ms  14.400ms  transposeUnroll4Col(float*, float*, int, int)
      API calls:   63.94%  728.96ms         7  104.14ms  90.279ms  169.79ms  cudaMemcpy
                   20.67%  235.70ms         2  117.85ms  10.325ms  225.38ms  cudaMalloc
                    9.41%  107.29ms         6  17.882ms  14.612ms  20.391ms  cudaDeviceSynchronize
                    5.11%  58.266ms         1  58.266ms  58.266ms  58.266ms  cudaDeviceReset
                    0.71%  8.0873ms         2  4.0436ms  2.2560ms  5.8313ms  cudaFree
                    0.06%  676.10us        97  6.9700us     200ns  307.10us  cuDeviceGetAttribute
                    0.05%  594.50us         1  594.50us  594.50us  594.50us  cudaGetDeviceProperties
                    0.03%  399.00us         6  66.500us  62.200us  84.800us  cudaLaunchKernel
                    0.00%  42.100us         1  42.100us  42.100us  42.100us  cuDeviceTotalMem
                    0.00%  13.400us         1  13.400us  13.400us  13.400us  cuDeviceGetPCIBusId
                    0.00%  12.400us         1  12.400us  12.400us  12.400us  cudaSetDevice
                    0.00%  7.9000us         2  3.9500us     400ns  7.5000us  cuDeviceGet
                    0.00%  3.7000us         6     616ns     500ns     700ns  cudaGetLastError
                    0.00%  2.0000us         3     666ns     300ns  1.1000us  cuDeviceGetCount
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuDeviceGetName
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetLuid
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetUuid



                    */
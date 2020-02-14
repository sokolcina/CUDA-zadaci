#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "../common/stopwatch.h"

#define RADIUS 9
#define BDIMX 16
#define BDIMY BDIMX

#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))

#define IPAD 2

__constant__ float coef[RADIUS];

#define c1 1f
#define c2 1f
#define c3 1f
#define c4 1f
#define c5 1f
#define c6 1f
#define c7 1f
#define c8 1f
#define c9 1f

void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void printData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%3.4f ", in[i]);
    }

    printf("\n");
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

void cpuCopy(float *out, float *in, const int nrows, const int ncols)
{
    for(int i=0;i<nrows;i++)
     for(int j=0;j<ncols;j++)
     {
        int inOffset=INDEX(i,j,ncols);
        int outOffset=INDEX(i+1,j+1,ncols+2);
        out[outOffset] = in[inOffset];
     }
}



void cpuCalculate(float* withZeros, float* in,float* out, float const *cpuCoef, const int nrows, const int ncols)
{
   
    cpuCopy(withZeros,in,nrows,ncols);
   
    int ncols1=ncols+2;
    for(int i=1; i<(nrows+1); i++)
     for(int j=1; j<(ncols+1); j++)
     {
        int outOffset=INDEX(i,j,ncols1);
        int inOffset=INDEX(i-1,j-1,ncols);
        float tmp=cpuCoef[0]*withZeros[outOffset];
        tmp+=cpuCoef[1]*withZeros[outOffset+1];
        tmp+=cpuCoef[2]*withZeros[outOffset-1];
        tmp+=cpuCoef[3]*withZeros[outOffset+ncols1];
        tmp+=cpuCoef[4]*withZeros[outOffset+ncols1+1];
        tmp+=cpuCoef[5]*withZeros[outOffset+ncols1-1];
        tmp+=cpuCoef[6]*withZeros[outOffset-ncols1];
        tmp+=cpuCoef[7]*withZeros[outOffset-ncols1+1];
        tmp+=cpuCoef[8]*withZeros[outOffset-ncols1-1];
        out[inOffset] = tmp;  
        
     } 
}

/*void setup_coef_constant (void)
{
    const float h_Boef[] = {c1,c2,c3,c4,c5,c6,c7,c8,c9};
    CHECK(cudaMemcpyToSymbol(coef, h_Boef, RADIUS  * sizeof(float)));
} 

ova funkcija nece nesto
*/


__global__ void copy(float *out, float *in, const int nrows,
    const int ncols)
{
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int row1=row+1;
    unsigned int col1=col+1;

    int inOffset=INDEX(row,col,ncols);
    int outOffset=INDEX(row1,col1,ncols+2);

    if (col < ncols && row < nrows)
    {
        out[outOffset] = in[inOffset];
    }
}



__global__ void copySmem(float *out, float *in, const int nrows,
    const int ncols)
    {
        __shared__ float tile[BDIMY][BDIMX];
    
        unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
        unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    
        unsigned int row1=row+1;
        unsigned int col1=col+1;
    
        int r=threadIdx.y;
        int c=threadIdx.x;
      
        int inOffset=INDEX(row,col,ncols);
        int outOffset=INDEX(row1,col1,ncols+2);    
        
        if (col < ncols && row < nrows)
        {
            tile[r][c]=in[inOffset];
            __syncthreads();
            out[outOffset]=tile[r][c];
        }
      
    }

__global__ void copySmemPad(float *out, float *in, const int nrows,
        const int ncols)
        {
            __shared__ float tile[BDIMY][BDIMX+IPAD];
        
            unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
            unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
        
            unsigned int row1=row+1;
            unsigned int col1=col+1;
        
            int r=threadIdx.y;
            int c=threadIdx.x;
          
            int inOffset=INDEX(row,col,ncols);
            int outOffset=INDEX(row1,col1,ncols+2);    
            
            if (col < ncols && row < nrows)
            {
                tile[r][c]=in[inOffset];
                __syncthreads();
                out[outOffset]=tile[r][c];
            }
          
        }
    


__global__ void copyBack(float *out, float *in, const int nrows,
const int ncols)
{

    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int row1=row+1;
    unsigned int col1=col+1;
  
    int inOffset=INDEX(row1,col1,ncols+2);
    int outOffset=INDEX(row,col,ncols);
    
    if (col < ncols && row < nrows)
    {
        out[outOffset] = in[inOffset];
    }
}


__global__ void copyBackSmem(float *out, float *in, const int nrows,
    const int ncols)
    {
        __shared__ float tile[BDIMY][BDIMX];
    
        unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
        unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    
        unsigned int row1=row+1;
        unsigned int col1=col+1;
    
        int r=threadIdx.y;
        int c=threadIdx.x;
      
       
        int inOffset=INDEX(row1,col1,ncols+2);
        int outOffset=INDEX(row,col,ncols);
        if (col < ncols && row < nrows)
        {
            tile[r][c]=in[inOffset];
            __syncthreads();
            out[outOffset]=tile[r][c];
        }
      
    }


__global__ void copyBackSmemPad(float *out, float *in, const int nrows,
        const int ncols)
        {
            __shared__ float tile[BDIMY][BDIMX+IPAD];
        
            unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
            unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
        
            unsigned int row1=row+1;
            unsigned int col1=col+1;
        
            int r=threadIdx.y;
            int c=threadIdx.x;
          
           
            int inOffset=INDEX(row1,col1,ncols+2);
            int outOffset=INDEX(row,col,ncols);
            if (col < ncols && row < nrows)
            {
                tile[r][c]=in[inOffset];
                __syncthreads();
                out[outOffset]=tile[r][c];
            }
          
        }

__global__ void calculate(float *out, float *in, const int nrows,
const int ncols)
{
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    int offset=INDEX(row,col,ncols);

    if (col <ncols-1 && row < nrows-1 && col>0 && row>0)
    {
        float tmp=coef[0]*in[offset];
        tmp+=coef[1]*in[offset+1];
        tmp+=coef[2]*in[offset-1];
        tmp+=coef[3]*in[offset+ncols];
        tmp+=coef[4]*in[offset+ncols+1];
        tmp+=coef[5]*in[offset+ncols-1];
        tmp+=coef[6]*in[offset-ncols];
        tmp+=coef[7]*in[offset-ncols+1];
        tmp+=coef[8]*in[offset-ncols-1];
        out[offset] = tmp;  
    }
}

// ovi kerneli ipak ne mogu jer je nezgodno da se podeli
// u parcice za obradjivanje jer uvek postoje elementi koji su zavisni
// u nekom prethodnom parcetu ili narednom rade samo za matrice
// koje mogu da obuhvate celu deljenu memoriju

__global__ void calculateSmem(float *out, float *in, int nrows, int ncols)
{

    __shared__ float tile[BDIMY+1][BDIMX+1];


    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    int r=threadIdx.y;
    int c=threadIdx.x;
    int bdx=blockDim.x;
    int bdy=blockDim.y;
    unsigned int offset = INDEX(row, col, ncols);
    if (row < nrows && col < ncols)
    {
      tile[r][c] = in[offset];
    
    }
    if(r==bdy-1)
    {
        offset = INDEX(row+1, col, ncols);
        tile[r+1][c] = in[offset];
    }
    if(c==bdx-1)
    {
        offset = INDEX(row, col+1, ncols);
        tile[r][c+1] = in[offset];
    }
    __syncthreads();

    if (col < ncols-1 && row < nrows-1 && col>0 && row>0)
    {
       
        float tmp=coef[0]*tile[r][c];
        tmp+=coef[1]*tile[r-1][c];
        tmp+=coef[2]*tile[r+1][c];
        tmp+=coef[3]*tile[r][c-1];
        tmp+=coef[4]*tile[r-1][c-1];
        tmp+=coef[5]*tile[r+1][c-1];
        tmp+=coef[6]*tile[r][c+1];
        tmp+=coef[7]*tile[r-1][c+1];
        tmp+=coef[8]*tile[r+1][c+1];
        out[offset] = tmp; 
    }
}


__global__ void calculateSmemPad(float *out, float *in, int nrows, int ncols)
{

    __shared__ float tile[BDIMY][BDIMX+IPAD];


    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    int r=threadIdx.y;
    int c=threadIdx.x;
    
    unsigned int offset = INDEX(row, col, ncols);
    if (row < nrows && col < ncols)
    {
      tile[r][c] = in[offset];
    }
    __syncthreads();

    if (col < ncols-1 && row < nrows-1 && col>0 && row>0)
    {
       
        float tmp=coef[0]*tile[r][c];
        tmp+=coef[1]*tile[r-1][c];
        tmp+=coef[2]*tile[r+1][c];
        tmp+=coef[3]*tile[r][c-1];
        tmp+=coef[4]*tile[r-1][c-1];
        tmp+=coef[5]*tile[r+1][c-1];
        tmp+=coef[6]*tile[r][c+1];
        tmp+=coef[7]*tile[r-1][c+1];
        tmp+=coef[8]*tile[r+1][c+1];
        out[offset] = tmp; 
    }
}


__global__ void calculateSmemDyn(float *out, float *in, int nrows, int ncols)
{

    extern __shared__ float tile[];


    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    int r=threadIdx.y;
    int c=threadIdx.x;
    
    unsigned int offset = INDEX(row, col, ncols);
    unsigned int offset2 = INDEX(r, c, blockDim.x);
    if (row < nrows && col < ncols)
    {
      tile[offset2] = in[offset];
    }
    __syncthreads();

    if (col < ncols-1 && row < nrows-1 && col>0 && row>0)
    {
       
        float tmp=coef[0]*tile[offset2];
        tmp+=coef[1]*tile[offset2-ncols];
        tmp+=coef[2]*tile[offset2+ncols];
        tmp+=coef[3]*tile[offset2-1];
        tmp+=coef[4]*tile[offset2-ncols-1];
        tmp+=coef[5]*tile[offset2+ncols-1];
        tmp+=coef[6]*tile[offset2+1];
        tmp+=coef[7]*tile[offset2-ncols+1];
        tmp+=coef[8]*tile[offset2+ncols+1];
        out[offset] = tmp; 
    }
}


__global__ void calculateSmemDynPads(float *out, float *in, int nrows, int ncols)
{

    extern __shared__ float tile[];


    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    int r=threadIdx.y;
    int c=threadIdx.x;
    
    unsigned int offset = INDEX(row, col, ncols);
    unsigned int offset2 = INDEX(r, c, ncols);
    if (row < nrows && col < ncols)
    {
      tile[offset2] = in[offset];
    }
    __syncthreads();

    if (col < ncols-1 && row < nrows-1 && col>0 && row>0)
    {
       
        float tmp=coef[0]*tile[offset2];
        tmp+=coef[1]*tile[offset2-ncols];
        tmp+=coef[2]*tile[offset2+ncols];
        tmp+=coef[3]*tile[offset2-1];
        tmp+=coef[4]*tile[offset2-ncols-1];
        tmp+=coef[5]*tile[offset2+ncols-1];
        tmp+=coef[6]*tile[offset2+1];
        tmp+=coef[7]*tile[offset2-ncols+1];
        tmp+=coef[8]*tile[offset2+ncols+1];
        out[offset] = tmp; 
    }
}



__global__ void calculateSmemUnroll(float *out, float *in, const int nrows, 
    const int ncols) 
{
   
    __shared__ float tile[BDIMY][BDIMX * 2];

    
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int row2 = row;
    unsigned int col2 = col + blockDim.x;

    int r=threadIdx.y;
    int c=threadIdx.x;
    int bd=blockDim.x;
   
    unsigned int offset = INDEX(row, col, ncols);
    unsigned int offset2 = INDEX(row2, col2, ncols);


    if (row < nrows && col < ncols)
    {
        tile[r][c] = in[offset];
    }
    if (row2 < nrows && col2 < ncols)
    {
        tile[r][bd + c] = in[offset2];
    }
    __syncthreads();

    if (col < ncols-1 && row < nrows-1 && col>0 && row>0)
    {
        
        float tmp=coef[0]*tile[r][c];
        tmp+=coef[1]*tile[r-1][c];
        tmp+=coef[2]*tile[r+1][c];
        tmp+=coef[3]*tile[r][c-1];
        tmp+=coef[4]*tile[r-1][c-1];
        tmp+=coef[5]*tile[r+1][c-1];
        tmp+=coef[6]*tile[r][c+1];
        tmp+=coef[7]*tile[r-1][c+1];
        tmp+=coef[8]*tile[r+1][c+1];
        out[offset] = tmp; 
    }
    if (col2 < ncols-1 && row2 < nrows-1 && col>0 && row>0)
    { 
        float tmp=coef[0]*tile[r][c+bd];
        tmp+=coef[1]*tile[r-1][c+bd];
        tmp+=coef[2]*tile[r+1][c+bd];
        tmp+=coef[3]*tile[r][c-1+bd];
        tmp+=coef[4]*tile[r-1][c-1+bd];
        tmp+=coef[5]*tile[r+1][c-1+bd];
        tmp+=coef[6]*tile[r][c+1+bd];
        tmp+=coef[7]*tile[r-1][c+1+bd];
        tmp+=coef[8]*tile[r+1][c+1+bd];
        out[offset2] = tmp; 
    }
}


__global__ void calculateSmemUnrollPad(float *out, float *in, const int nrows, 
    const int ncols) 
{
   
    __shared__ float tile[BDIMY][BDIMX * 2+IPAD];

    
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int row2 = row;
    unsigned int col2 = col + blockDim.x;

    int r=threadIdx.y;
    int c=threadIdx.x;
    int bd=blockDim.x;
   
    unsigned int offset = INDEX(row, col, ncols);
    unsigned int offset2 = INDEX(row2, col2, ncols);


    if (row < nrows && col < ncols)
    {
        tile[r][c] = in[offset];
    }
    if (row2 < nrows && col2 < ncols)
    {
        tile[r][bd + c] = in[offset2];
    }
    __syncthreads();

    if (col < ncols-1 && row < nrows-1 && col>0 && row>0)
    {
        
        float tmp=coef[0]*tile[r][c];
        tmp+=coef[1]*tile[r-1][c];
        tmp+=coef[2]*tile[r+1][c];
        tmp+=coef[3]*tile[r][c-1];
        tmp+=coef[4]*tile[r-1][c-1];
        tmp+=coef[5]*tile[r+1][c-1];
        tmp+=coef[6]*tile[r][c+1];
        tmp+=coef[7]*tile[r-1][c+1];
        tmp+=coef[8]*tile[r+1][c+1];
        out[offset] = tmp; 
    }
    if (col2 < ncols-1 && row2 < nrows-1 && col>0 && row>0)
    { 
        float tmp=coef[0]*tile[r][c+bd];
        tmp+=coef[1]*tile[r-1][c+bd];
        tmp+=coef[2]*tile[r+1][c+bd];
        tmp+=coef[3]*tile[r][c-1+bd];
        tmp+=coef[4]*tile[r-1][c-1+bd];
        tmp+=coef[5]*tile[r+1][c-1+bd];
        tmp+=coef[6]*tile[r][c+1+bd];
        tmp+=coef[7]*tile[r-1][c+1+bd];
        tmp+=coef[8]*tile[r+1][c+1+bd];
        out[offset2] = tmp; 
    }
}





int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting main at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int powc=12;
    int powr=12;
    if (argc > 1) powr = atoi(argv[1]);

    if (argc > 2) powc = atoi(argv[2]);
    

    int nrows = (1 << powr)-2;
    int ncols = (1 << powc)-2;

    

   

    printf(" with matrcol nrows %d ncols %d\n", nrows+2, ncols+2);
    size_t ncells = nrows*ncols;
    size_t nBytes = ncells * sizeof(float);

    size_t ncells1 = (nrows+2) * (ncols+2);
    size_t nBytes1 = ncells1 * sizeof(float);
   

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *h_out  = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes1);
    float *d_out  = (float *)malloc(nBytes);
    memset(h_B,0,nBytes1);
    //  initialize host array
    initialData(h_A, ncells);
    //printf("\nh_a\n");
    //printData(h_A,ncells);
   
    //setup_coef_constant();

    const float h_Coef[9] = {5,3,4,2,5,2,5,2,4};
    CHECK(cudaMemcpyToSymbol(coef, h_Coef, RADIUS  * sizeof(float)));

    dim3 block (BDIMX, BDIMY);
    dim3 grid ((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
    dim3 grid2 ((grid.x + 2 - 1) / 2, grid.y);
    // allocate device memory
    float *d_A, *d_C, *d_B,*d_D;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes1));
    CHECK(cudaMalloc((float**)&d_C, nBytes1));
    CHECK(cudaMalloc((float**)&d_D, nBytes));

    CHECK(cudaMemset(d_C, 0, nBytes1));
    CHECK(cudaMemset(d_B, 0, nBytes1));
    
    Stopwatch s;
    cpuCalculate(h_B,h_A,h_out,h_Coef,nrows,ncols);
    printf("\ncpuCalculate %f sec \n\n",s.elapsed());
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    dim3 grid3 ((ncols+2 + block.x - 1) / block.x, (nrows+2 + block.y - 1) / block.y);

  /*
    copy<<<grid,block>>>(d_B,d_A,nrows,ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

   
    calculateSmem<<<grid3,block>>>(d_C,d_B,nrows+2,ncols+2);
    //calculateSmemDyn<<<grid3,block,BDIMX*BDIMY*sizeof(float)>>>(d_C,d_B,nrows+2,ncols+2);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

   // CHECK(cudaMemcpy(h_B, d_C, nBytes1, cudaMemcpyDeviceToHost));
   // printData(h_B,ncells1);
    copyBack<<<grid,block>>>(d_D,d_C,nrows,ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError()); 

    CHECK(cudaMemcpy(d_out, d_D, nBytes, cudaMemcpyDeviceToHost));
    checkResult(h_out,d_out,ncells);
    printf("\n\n");
    printData(h_out,ncells);
    printf("\n\n");
    printData(d_out,ncells);
   
    calculateSmem<<<grid3,block>>>(d_C,d_B,nrows+2,ncols+2);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    copyBack<<<grid,block>>>(d_D,d_C,nrows,ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(d_out, d_D, nBytes, cudaMemcpyDeviceToHost));
    checkResult(d_out,h_out,ncells); */

    s.reset();
    copy<<<grid,block>>>(d_B,d_A,nrows,ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("copy<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y, s.elapsed());
    
    s.reset();
    calculate<<<grid3,block>>>(d_C,d_B,nrows+2,ncols+2);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("calculate<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y, s.elapsed());
    s.reset();
    copyBack<<<grid,block>>>(d_D,d_C,nrows,ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("copyBack<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y, s.elapsed());
    CHECK(cudaMemcpy(d_out, d_D, nBytes, cudaMemcpyDeviceToHost));
    checkResult(d_out,h_out,ncells); 
   

    s.reset();
    copySmem<<<grid,block>>>(d_B,d_A,nrows,ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("copySmem<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y, s.elapsed());
    
    s.reset();
    calculate<<<grid3,block>>>(d_C,d_B,nrows+2,ncols+2);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("calculate<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y, s.elapsed());
    s.reset();
    copyBackSmem<<<grid,block>>>(d_D,d_C,nrows,ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("copyBackSmem<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y, s.elapsed());
    CHECK(cudaMemcpy(d_out, d_D, nBytes, cudaMemcpyDeviceToHost));
    checkResult(d_out,h_out,ncells); 
   
   

    s.reset();
    copySmemPad<<<grid,block>>>(d_B,d_A,nrows,ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("copySmem<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y, s.elapsed());
    
    s.reset();
    calculate<<<grid3,block>>>(d_C,d_B,nrows+2,ncols+2);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("calculate<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y, s.elapsed());
    s.reset();
    copyBackSmemPad<<<grid,block>>>(d_D,d_C,nrows,ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("copyBackSmem<<<grid (%d,%d) block (%d,%d)>>> elapsed %f sec \n", grid.x, grid.y, block.x,
    block.y, s.elapsed());
    CHECK(cudaMemcpy(d_out, d_D, nBytes, cudaMemcpyDeviceToHost));
    checkResult(d_out,h_out,ncells); 
   
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_D));
    free(h_A);
    free(h_B);
    free(d_out);
    free(h_out);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

/* 

a starting main at device 0: GeForce GTX 1050  with matrcol nrows 4096 ncols 4096

cpuCalculate 0.536247 sec

==8548== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==8548== copy<<<grid (256,256) block (16,16)>>> elapsed 0.160981 sec
Replaying kernel "copy(float*, float*, int, int)" (done)
==8548== calculate<<<grid (256,256) block (16,16)>>> elapsed 0.108635 sec
Replaying kernel "calculate(float*, float*, int, int)" (done)
==8548== copyBack<<<grid (256,256) block (16,16)>>> elapsed 0.105520 sec
Replaying kernel "copyBack(float*, float*, int, int)" (done)
Arrays match.

==8548== copySmem<<<grid (256,256) block (16,16)>>> elapsed 0.105121 sec
Replaying kernel "copySmem(float*, float*, int, int)" (done)
==8548== calculate<<<grid (256,256) block (16,16)>>> elapsed 0.112580 sec
Replaying kernel "calculate(float*, float*, int, int)" (done)
copyBackSmem<<<grid (256,256) block (16,16)>>> elapsed 0.106316 sec
==8548== Replaying kernel "copyBackSmem(float*, float*, int, int)" (done)
Arrays match.

==8548== copySmem<<<grid (256,256) block (16,16)>>> elapsed 0.112830 sec
Replaying kernel "copySmemPad(float*, float*, int, int)" (done)
calculate<<<grid (256,256) block (16,16)>>> elapsed 0.108445 sec
==8548== Replaying kernel "calculate(float*, float*, int, int)" (done)
==8548== copyBackSmem<<<grid (256,256) block (16,16)>>> elapsed 0.106764 sec
Replaying kernel "copyBackSmemPad(float*, float*, int, int)" (done)
Arrays match.

==8548== Profiling application: a
==8548== Profiling result:
==8548== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "GeForce GTX 1050 (0)"
    Kernel: copySmemPad(float*, float*, int, int)
          1                      dram_read_throughput                  Device Memory Read Throughput  32.192GB/s  32.192GB/s  32.192GB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  31.831GB/s  31.831GB/s  31.831GB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.996094    1.996094    1.996094
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.996094    1.996094    1.996094
    Kernel: calculate(float*, float*, int, int)
          3                      dram_read_throughput                  Device Memory Read Throughput  24.440GB/s  24.514GB/s  24.470GB/s
          3                     dram_write_throughput                 Device Memory Write Throughput  24.155GB/s  24.171GB/s  24.162GB/s
          3      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          3     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
    Kernel: copy(float*, float*, int, int)
          1                      dram_read_throughput                  Device Memory Read Throughput  38.718GB/s  38.718GB/s  38.718GB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  38.391GB/s  38.391GB/s  38.391GB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
    Kernel: copyBackSmemPad(float*, float*, int, int)
          1                      dram_read_throughput                  Device Memory Read Throughput  31.857GB/s  31.857GB/s  31.857GB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  31.526GB/s  31.526GB/s  31.526GB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.996094    1.996094    1.996094
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.996094    1.996094    1.996094
    Kernel: copyBack(float*, float*, int, int)
          1                      dram_read_throughput                  Device Memory Read Throughput  38.005GB/s  38.005GB/s  38.005GB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  37.680GB/s  37.680GB/s  37.680GB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
    Kernel: copySmem(float*, float*, int, int)
          1                      dram_read_throughput                  Device Memory Read Throughput  32.824GB/s  32.824GB/s  32.824GB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  32.672GB/s  32.672GB/s  32.672GB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
    Kernel: copyBackSmem(float*, float*, int, int)
          1                      dram_read_throughput                  Device Memory Read Throughput  34.424GB/s  34.424GB/s  34.424GB/s
          1                     dram_write_throughput                 Device Memory Write Throughput  32.412GB/s  32.412GB/s  32.412GB/s
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000


*/
#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"

/*
 * A simple example of performing matrix-vector multiplication using the cuBLAS
 * library and some randomly generated inputs.
 */

/*
 * N = # of rows
 * M = # of columns
 */
int N = 500;
int M = 500;



void generate_random_dense_matrix(int N, int M, float *A)
{
    int i, j;
    double rMax = (double)RAND_MAX;

    //int id=1;
    for (i = 0; i < N; i++)
    {
        
        for (j = 0; j < M; j++)
        {
            double dr = (double)rand();
            A[i * M + j] = (dr / rMax) * 100.0; 
            //A[i * M + j]=id++;
        }
    }
}



void compute_coefficients(int N, int M, float* A, float* y)
{
    for (int i = 0; i < N; i++)
    {
        float s=0.0f;
        for (int j = 0; j < M; j++)
        {
            s=s+A[j * N + i];
        }
        y[i]=s;
    }
}

void rearrange(float *vec, int *pivotArray, int N){
    for (int i = 0; i < N; i++) {
        float temp = vec[i];
        vec[i] = vec[pivotArray[i] - 1];
        vec[pivotArray[i] - 1] = temp;
    }   
}

int main(int argc, char **argv)
{
   
    float *h_A, *h_x, *h_y;
    float *d_A, *d_x, *d_y;
    float beta;
    float alpha;
    cublasHandle_t handle = 0;


    alpha = 1.0f;
    //beta = 0.0f;
    if (argc > 1) {
        N = atoi(argv[1]);
        M = N;
    }
    // Generate inputs
    srand(9384);

    //CHECK(cudaDeviceSynchronize());
    h_A = (float*)malloc(N * M * sizeof(float));
    h_y = (float*)malloc(N * sizeof(float));
    h_x = (float*)malloc(M * sizeof(float));

    generate_random_dense_matrix(N, M, h_A);
    compute_coefficients(N, M, h_A, h_y);
    memset(h_x,0,M * sizeof(float));
    CHECK(cudaMalloc(&d_y, N * sizeof ( float )));
    CHECK(cudaMalloc(&d_x, M * sizeof ( float )));
    CHECK(cudaMalloc(&d_A, N * M * sizeof ( float )));
    CHECK_CUBLAS(cublasSetMatrix(N,M,sizeof(float),h_A,N,d_A,N));
    /*
    for(int i=0; i<M; i++)
        h_x[i]=2.0f; */

     

    cudaMemset(d_x, 0, M * sizeof(float));
    
    CHECK(cudaMemcpy(d_y, h_y, sizeof(float) * N, cudaMemcpyHostToDevice));

    CHECK_CUBLAS(cublasCreate(&handle));

    //priprema za cublasSgetrfBatched da dobijemo pivotArray
    //za LU dekompoziciju
    int *P, *Info;
   
    float ** h_AA = (float **) malloc(sizeof(float *));
    *h_AA = d_A;
    float ** d_AA;
    CHECK(cudaMalloc(&d_AA,sizeof(float*)));
    CHECK(cudaMemcpy(d_AA,h_AA,sizeof(float*),cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&P, N * sizeof(int)));
    CHECK(cudaMalloc(&Info, sizeof(int)));
    
    
    
    CHECK_CUBLAS(cublasSgetrfBatched(handle,N,d_AA,N,P,Info,1));
    CHECK(cudaDeviceSynchronize());

    int INFOh = 0;
    CHECK(cudaMemcpy(&INFOh,Info,sizeof(int),cudaMemcpyDeviceToHost));

    printf("%d\n",INFOh);
    if(INFOh == N)
    {
        printf("Factorization Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    int *h_P=(int*)malloc(N*sizeof(int));
    CHECK(cudaMemcpy(h_P,P,N*sizeof(int),cudaMemcpyDeviceToHost));
    /*
    for(int i=0;i<N;i++)
    printf("%f ",h_y[i]);
    printf("\n");
   
    for(int i=0;i<N;i++)
    printf("%f ",h_y[i]);
    printf("\n"); */

    //da se preraspodeli
    rearrange(h_y,h_P,N);
    CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
    //CHECK_CUBLAS(cublasSgetriBatched(handle,N,d_AA,N,P,Info,));
    
    //donji deo matrice
    CHECK_CUBLAS(cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, 1, &alpha, d_A, N, d_y, N));
    CHECK(cudaDeviceSynchronize());
    // gornji deo matrice
    CHECK_CUBLAS(cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, 1, &alpha, d_A, N, d_y, N));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_x, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\n");
    /*for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("%2.6f ", h_A[j * N + i]);
        }
        printf("\n");
    }*/
    /*for(int i=0;i<N;i++)
        printf("%d ",h_P[i]);
    printf("\n"); */
    printf("X: \n");
    for(int i=0;i<N;i++)
        printf("%2.3f ",h_x[i]);
   
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_AA);
    CHECK(cudaFree(d_AA[0]));
    CHECK(cudaFree(d_AA));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(Info));
    CHECK(cudaFree(P));
  
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}


/* 

0

X:
1.000 1.000 1.000 1.000 1.001 1.000 1.000 1.000 1.000 1.000 
1.001 0.999 0.999 1.000 0.999 1.001 1.001 1.000 1.000 1.000 
1.001 1.000 1.001 1.000 0.999 0.999 1.001 0.999 0.999 0.998 
1.000 1.001 1.000 0.999 1.000 1.000 1.001 1.000 1.000 1.000 
1.000 1.001 1.000 0.999 1.001 1.001 1.000 0.999 1.000 0.999 
0.999 1.000 0.999 0.999 0.998 1.000 1.000 1.000 1.000 1.000 
0.999 1.000 1.001 0.999 1.000 1.000 1.001 1.000 0.998 1.000 
1.000 1.000 1.001 1.000 1.000 1.000 0.999 1.001 1.000 1.001 
1.000 1.000 1.001 1.000 1.000 0.999 1.000 1.000 1.001 1.001 
1.000 1.001 1.000 1.000 0.999 1.000 1.000 1.001 1.000 1.000 
1.000 1.000 0.999 0.999 0.998 1.000 1.000 1.001 1.001 1.000 
1.000 1.000 0.998 1.000 0.999 1.001 1.000 0.999 1.002 0.999 
1.000 0.998 0.999 1.000 1.000 1.000 1.000 0.999 1.000 0.999 
1.001 0.999 1.001 1.001 1.000 1.000 1.000 1.001 1.001 1.000 
1.000 1.001 1.001 1.001 1.000 1.001 1.001 1.001 1.000 0.999 
1.000 0.999 1.000 1.000 1.001 1.000 1.000 1.000 1.000 0.999 
0.999 1.000 1.000 0.999 1.000 1.000 1.001 1.000 1.000 1.000 
1.000 1.000 0.999 1.002 1.001 0.999 1.000 0.999 1.000 1.000 
1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 0.999 
1.001 0.999 0.999 1.000 1.001 1.000 0.999 1.000 0.998 0.999 
0.999 1.001 1.001 1.001 1.000 1.000 1.000 1.000 1.000 0.999 
1.000 1.001 1.000 1.000 1.000 1.001 0.999 0.999 1.000 0.999 
1.000 1.000 1.002 1.001 1.000 0.999 1.001 1.001 1.001 1.000 
1.000 1.000 1.001 1.000 1.001 1.000 0.999 1.000 0.998 1.000 
1.001 0.999 1.000 0.999 0.998 1.000 0.999 1.000 1.000 1.001 
1.001 0.999 1.000 0.999 1.001 1.000 1.000 1.000 0.999 1.000 
1.000 1.001 1.000 1.002 0.999 1.000 1.000 0.998 0.999 1.000 
1.000 1.000 1.000 1.001 1.000 1.000 1.000 1.000 1.001 1.001 
1.000 1.000 1.001 0.999 1.001 1.001 1.000 1.000 1.000 1.001 
1.001 1.000 1.001 0.999 1.000 1.001 1.000 1.000 0.999 1.000 
1.000 1.000 1.001 0.999 0.999 1.000 1.000 1.000 1.001 1.000 
1.001 0.999 1.001 1.000 0.999 1.000 1.001 1.000 0.999 1.000 
1.001 0.999 1.001 1.000 0.999 1.000 1.000 0.999 1.000 1.000 
1.001 1.000 1.000 1.000 1.000 1.001 0.999 1.001 1.000 0.999 
1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.001 1.000 
1.000 1.000 1.002 1.000 0.999 1.001 1.001 0.999 1.001 1.000 
1.000 0.999 1.002 1.001 0.998 0.999 1.000 0.999 1.000 0.999 
1.000 1.000 1.001 1.000 1.000 1.001 1.000 1.000 1.000 0.999 
0.999 1.001 0.999 0.999 1.001 1.000 1.000 1.000 1.000 1.000 
1.001 0.998 1.000 1.001 1.001 1.001 1.000 1.000 0.998 0.999 
1.000 0.999 1.000 0.999 1.000 1.000 1.000 1.000 1.001 1.000 
1.000 1.001 0.999 0.998 0.998 1.000 1.001 1.001 0.999 1.001 
0.999 1.000 1.000 1.000 1.001 1.000 1.001 1.001 1.002 1.000 
1.000 1.000 1.000 1.000 1.001 1.000 1.000 1.001 1.001 1.001 
1.000 0.999 1.001 1.000 1.001 1.000 1.000 1.000 1.002 1.001 
1.000 0.999 1.000 1.001 1.000 0.999 1.001 0.998 1.002 1.000 
1.001 0.999 1.000 1.001 1.000 1.000 1.000 1.000 0.999 1.001 
0.999 1.000 1.001 1.000 1.001 1.000 0.999 0.999 1.000 1.000 
1.000 1.001 1.000 0.999 1.000 1.001 0.999 1.001 1.000 1.000 
0.999 0.999 1.000 1.002 1.000 1.000 1.000 1.000 0.999 1.000 

*/
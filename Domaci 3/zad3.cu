#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse_v2.h>
#include <cuda.h>

/*
 * This is an example demonstrating usage of the cuSPARSE library to perform a
 * sparse matrix-vector multiplication on randomly generated data.
 */

/*
 * M = # of rows
 * N = # of columns
 */
int M = 5;
int N = 5;

/*
 * Generate a vector of length N with random single-precision floating-point
 * values between 0 and 100.
 */
void generate_random_vector(int N, float **outX)
{
    int i;
    double rMax = (double)RAND_MAX;
    float *X = (float *)malloc(sizeof(float) * N);

    for (i = 0; i < N; i++)
    {
        int r = rand();
        double dr = (double)r;
        X[i] = (dr / rMax) * 100.0;
    }

    *outX = X;
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

/*
 * Generate random dense matrix A in column-major order, while rounding some
 * elements down to zero to ensure it is sparse.
 */
int generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);
    //cudaMemset(A,0,M*N*sizeof(float));
    int totalNnz = 0;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
        
            int r = rand();
            float *curr = A + (j * M + i);

            if (r % 3 > 0)
            {
                *curr = 0.0f;
            }
            else
            {
                double dr = (double)r;
                *curr = (dr / rMax) * 5.0;
            }
            //if(i==j) *curr=1.0f;
            if (*curr != 0.0f)
            {
                totalNnz++;
            }
           
        }
    }

    *outA = A;
    return totalNnz;
}

void print_partial_matrix(float *M, int nrows, int ncols, int max_row,
    int max_col)
{
int row, col;

for (row = 0; row < max_row; row++)
{
    for (col = 0; col < max_col; col++)
    {
        printf("%2.2f ", M[row * ncols + col]);
    }
    printf("...\n");
}
printf("...\n");
}

int main(int argc, char **argv)
{
    int row;
    float *A, *dA;
    int *dNnzPerRow;
    float *dCsrValA;
    int *dCsrRowPtrA;
    int *dCsrColIndA;
    int totalNnz;
    float alpha = 1.0f;
    //float beta = 0.0f;
    float *dX, *X;
    float *dY, *Y;
    int structural_zero;
    int numerical_zero;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;
    csrsv2Info_t  info_A  = 0;
    const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    int lworkInBytes;
    void * d_work = NULL;
    // Generate input
    srand(9384);
    int trueNnz = generate_random_dense_matrix(M, N, &A);
    print_partial_matrix(A,M,M,M,M);
    
   
      // generate_random_vector(M, &Y);
    X=(float*)malloc(N*sizeof(float));
    Y=(float*)malloc(M*sizeof(float));
    compute_coefficients(M,N,A,Y);
    for(int i=0;i<N;i++)
        {
            printf("%f ",Y[i]);
            Y[i]=1;
        }
        printf("\n");
    //memset(X,0,N*sizeof(float));
    // Create the cuSPARSE handlef
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateCsrsv2Info(&info_A));
    // Allocate device memory for vectors and the dense form of the matrix A
    CHECK(cudaMalloc((void **)&dX, sizeof(float) * N));
    CHECK(cudaMalloc((void **)&dY, sizeof(float) * M));
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dNnzPerRow, sizeof(int) * M));

    // Construct a descriptor of the matrix A
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    // Transfer the input vectors and dense matrix A to the device
    //CHECK(cudaMemcpy(dX, X, sizeof(float) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dY, Y, sizeof(float) * M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));
/*
    // Compute the number of non-zero elements in A
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descr, dA,
                                M, dNnzPerRow, &totalNnz));
*/
    totalNnz=trueNnz;
    if (totalNnz != trueNnz)
    {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
                "value: expected %d but got %d\n", trueNnz, totalNnz);
        return 1;
    }

    // Allocate device memory to store the sparse CSR representation of A
    CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * totalNnz));
    CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (M + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalNnz));

    // Convert A from a dense formatting to a CSR formatting, using the GPU
    CHECK_CUSPARSE(cusparseSdense2csr(handle, M, N, descr, dA, M, dNnzPerRow,
                                      dCsrValA, dCsrRowPtrA, dCsrColIndA));

    for(int i=0;i<N;i++)
        X[i]=1;
                                     
    CHECK(cudaMemcpy(dX, X, sizeof(float) * N, cudaMemcpyHostToDevice));

    /*
    CHECK_CUSPARSE(cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        M, N, totalNnz, &alpha, descr, dCsrValA,
                                        dCsrRowPtrA, dCsrColIndA, dX, &beta, dY));
                             
                                
                                   
    CHECK(cudaMemcpy(Y, dY, sizeof(float) * M, cudaMemcpyDeviceToHost));
                                  
    for (row = 0; row < 5; row++)
        {
         printf("%f ", Y[row]);
        }
        printf("%\n mnozenje\n");
 */  
        compute_coefficients(M,N,A,Y);
        CHECK(cudaMemcpy(dY, Y, sizeof(float) * N, cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseScsrsv2_bufferSize(
                                handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                M,
                                totalNnz,
                                descr,
                                dCsrValA,
                                dCsrRowPtrA,
                                dCsrColIndA,
                                info_A,
                                &lworkInBytes));

    if (NULL != d_work) { cudaFree(d_work); }
    CHECK(cudaMalloc((void**)&d_work, lworkInBytes));
    CHECK(cudaDeviceSynchronize());
    CHECK_CUSPARSE(cusparseScsrsv2_analysis(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    M,
                                    totalNnz,
                                    descr,
                                    dCsrValA,
                                    dCsrRowPtrA,
                                    dCsrColIndA,
                                    info_A,
                                    policy,
                                    d_work));
    CHECK(cudaDeviceSynchronize());

    cusparseStatus_t status = cusparseXcsrsv2_zeroPivot(handle, info_A, &structural_zero);
if (CUSPARSE_STATUS_ZERO_PIVOT == status){
   printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
}
    CHECK_CUSPARSE(cusparseScsrsv2_solve(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    M,
                                    totalNnz,
                                    &alpha,
                                    descr,
                                    dCsrValA,
                                    dCsrRowPtrA,
                                    dCsrColIndA,
                                    info_A,
                                    dY,
                                    dX,
                                    policy,
                                    d_work));
    CHECK(cudaDeviceSynchronize());
    status = cusparseXcsrsv2_zeroPivot(handle, info_A, &numerical_zero);
if (CUSPARSE_STATUS_ZERO_PIVOT == status){
   printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
}
    
    // Copy the result vector back to the host
    CHECK(cudaMemcpy(Y, dY, sizeof(float) * M, cudaMemcpyDeviceToHost));

    for (row = 0; row < 5; row++)
    {
        printf("%2.2f ", Y[row]);
    }

    printf("...\n");

    CHECK(cudaMemcpy(X, dX, sizeof(float) * M, cudaMemcpyDeviceToHost));

    for (row = 0; row < 5; row++)
    {
        printf("%2.2f ", X[row]);
    }

    printf("...\n\n");

    // Perform matrix-vector multiplication with the CSR-formatted matrix A
  
    
    free(A);
    free(X);
    free(Y);

    CHECK(cudaFree(dX));
    CHECK(cudaFree(dY));
    CHECK(cudaFree(dA));
    CHECK(cudaFree(d_work));
    CHECK(cudaFree(dNnzPerRow));
    CHECK(cudaFree(dCsrValA));
    CHECK(cudaFree(dCsrRowPtrA));
    CHECK(cudaFree(dCsrColIndA));
    cusparseDestroyCsrsv2Info(info_A);
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
    CHECK_CUSPARSE(cusparseDestroy(handle));


    return 0;
}
/* 
 ovo nece iz ne znam ni ja kog razloga mozda je i neki bug
 a mozda i ja nesto nisam dobro uradio al u sustini po dokumentaciji
 bi trebalo ovako da prodje

*/
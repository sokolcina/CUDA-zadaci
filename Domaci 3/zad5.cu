#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>
#define _USE_MATH_DEFINES
#include <math.h>

/*
 * An example usage of the cuFFT library. This example performs a 1D forward
 * FFT.
 */

int nprints = 10;

/*
 * Create N fake samplings along the function cos(x). These samplings will be
 * stored as single-precision floating-point values.
 */
void generate_fake_samples(int N, float **out)
{
    
    float *result = (float *)malloc(sizeof(float) * N);
    double delta = M_PI / 20.0;
    double ndelta = -delta;
    int half = N / 2;
    int end = half - 1;
    
    for (int i=0; i < half; i++)
        result[end - i] = sin(i * ndelta) + cos(i * ndelta);

    for (int i=0; i < half; i++)
        result[i + half] = sin(i * delta) + cos(i * delta);

    *out = result;
}

/*
 * Convert a real-valued vector r of length Nto a complex-valued vector.
 */
void real_to_complex(float *r, cufftComplex **complx, int N)
{
    int i;
    (*complx) = (cufftComplex *)malloc(sizeof(cufftComplex) * N);

    for (i = 0; i < N; i++)
    {
        (*complx)[i].x = r[i];
        (*complx)[i].y = 0;
    }
}

int main(int argc, char **argv)
{
    int i;
    int N = 2048;
    float *samples;
    cufftHandle plan = 0;
    cufftComplex *dComplexSamples, *complexSamples, *complexFreq, 
    *complexFreqInver;

    // Input Generation
    generate_fake_samples(N, &samples);
    real_to_complex(samples, &complexSamples, N);
    complexFreq = (cufftComplex *)malloc(sizeof(cufftComplex) * N);
    complexFreqInver = (cufftComplex *)malloc(sizeof(cufftComplex) * N); 
    printf("Initial Samples:\n");

    for (i = 0; i < nprints; i++)
    {
        printf("  %2.4f\n", samples[i]);
    }

    printf("  ...\n");

    // Setup the cuFFT plan
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    // Allocate device memory
    CHECK(cudaMalloc((void **)&dComplexSamples,
            sizeof(cufftComplex) * N));
   
    // Transfer inputs into device memory
    CHECK(cudaMemcpy(dComplexSamples, complexSamples,
            sizeof(cufftComplex) * N, cudaMemcpyHostToDevice));

    // Execute a complex-to-complex 1D FFT
    CHECK_CUFFT(cufftExecC2C(plan, dComplexSamples, dComplexSamples,
                             CUFFT_FORWARD));

    // Retrieve the results into host memory
    CHECK(cudaMemcpy(complexFreq, dComplexSamples,
            sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost));
            
    CHECK_CUFFT(cufftExecC2C(plan, dComplexSamples, dComplexSamples,
                CUFFT_INVERSE));

    CHECK(cudaMemcpy(complexFreqInver, dComplexSamples,
            sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost));

    printf("Fourier Coefficients:\n");

    for (i = 0; i < nprints; i++)
    {
        printf("  %d: (%2.4f, %2.4f)\n", i + 1, complexFreq[i].x,
               complexFreq[i].y);
    }

    printf("  ...\n");

    printf("Fourier Coefficients Inverse:\n");
    for (i = 0; i < nprints; i++)
    {
        printf("  %d: (%2.4f, %2.4f)\n", i + 1, (complexFreqInver[i].x/N),
            (complexFreqInver[i].y/N));
    
    }

    printf("  ...\n");

    printf("Samples:\n");
    for (i = 0; i < nprints; i++)
    {
        printf("  %d: (%2.4f, %2.4f)\n", i + 1, complexSamples[i].x,
            complexSamples[i].y);
    }

    printf("  ...\n");

    free(samples);
    free(complexSamples);
    free(complexFreq);
    free(complexFreqInver);
    CHECK(cudaFree(dComplexSamples));
    CHECK_CUFFT(cufftDestroy(plan));

    return 0;
}

/*

Initial Samples:
  -0.4370
  -0.6420
  -0.8313
  -1.0000
  -1.1441
  -1.2601
  -1.3450
  -1.3968
  -1.4142
  -1.3968
  ...
Fourier Coefficients:
  1: (-5.6595, 0.0000)
  2: (-7.6626, 0.1376)
  3: (-5.6688, 0.2036)
  4: (-7.6879, 0.4140)
  5: (-5.6968, 0.4090)
  6: (-7.7387, 0.6943)
  7: (-5.7442, 0.6182)
  8: (-7.8162, 0.9811)
  9: (-5.8117, 0.8333)
  10: (-7.9219, 1.2773)
  ...
Fourier Coefficients Inverse:
  1: (-0.4370, 0.0000)
  2: (-0.6420, -0.0000)
  3: (-0.8313, -0.0000)
  4: (-1.0000, -0.0000)
  5: (-1.1441, 0.0000)
  6: (-1.2601, -0.0000)
  7: (-1.3450, -0.0000)
  8: (-1.3968, 0.0000)
  9: (-1.4142, -0.0000)
  10: (-1.3968, 0.0000)
  ...
Samples:
  1: (-0.4370, 0.0000)
  2: (-0.6420, 0.0000)
  3: (-0.8313, 0.0000)
  4: (-1.0000, 0.0000)
  5: (-1.1441, 0.0000)
  6: (-1.2601, 0.0000)
  7: (-1.3450, 0.0000)
  8: (-1.3968, 0.0000)
  9: (-1.4142, 0.0000)
  10: (-1.3968, 0.0000)
  ...


*/

#include <iostream>

#include "multiply.h"

using namespace std;

namespace matrix
{
// -----------------------------------------------------------------------------

float* multiplyGPU(float *A, int A_nrows, int A_ncols,
                 float *B, int B_nrows, int B_ncols,
                 float *C, int C_nrows, int C_ncols)
{
    float elapsed_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate device memory 
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, A_nrows * A_ncols * sizeof(float));
    cudaMalloc((void**)&d_B, B_nrows * B_ncols * sizeof(float));
    cudaMemcpy(d_A, A, A_nrows * A_ncols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, B_nrows * B_ncols, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_C, C_nrows * C_ncols * sizeof(float));

    // setup the execution configuration
    int k = 100;
    int l = 100;
    int width = 100;
    dim3 dimBlock((k-1)/width+1, (l-1)/width+1);
    dim3 dimGrid(width, width);
    
    // start timing
    cudaEventRecord(start, 0);
    // launch kernel
    multiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
    cudaMemcpy(C, d_C, width*width*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    // stop timing

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    cout << "(GPU) Elapsed Time: " << elapsed_time_ms << endl;

    // free the device matrices
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}


__global__ void multiplyKernel(float *d_A, float *d_B, float *d_C, int width)
{
    // 2D-thread ID
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;

    float tempValue = 0;
    for (int i = 0; i < width; ++i)
    {
        float d_A_elems = d_A[ty * width + i];
        float d_B_elems = d_B[i * width + tx];
        tempValue += (d_A_elems * d_B_elems);
    }

    d_C[ty * width + tx] = tempValue;
}

// -----------------------------------------------------------------------------

}

#ifndef MULTIPLY_H_
#define MULTIPLY_H_

#include <cuda_runtime.h>

namespace matrix
{

// ----------------------------------------------------------------------------

void multiply(float *A, int A_nrows, int A_ncols,
              float *B, int B_nrows, int B_ncols,
              float *C, int C_nrows, int C_ncols);

// CPU
float* multiplyCPU(float *A, int A_nrows, int A_ncols,
                   float *B, int B_nrows, int B_ncols,
                   float *C, int C_nrows, int C_ncols);



// GPU
float* multiplyGPU(float *A, int A_nrows, int A_ncols,
                   float *B, int B_nrows, int B_ncols,
                   float *C, int C_nrows, int C_ncols);

__global__ void multiplyKernel(float *d_A, float *d_B, float *d_C, int width);




// ----------------------------------------------------------------------------
}

#endif // MULTIPLY_H_

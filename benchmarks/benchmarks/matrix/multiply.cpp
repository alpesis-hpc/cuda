#include <iostream>

#include "multiply.h"

using namespace std;


namespace matrix
{

// ----------------------------------------------------------------------------

void multiply(float *A, int A_nrows, int A_ncols,
              float *B, int B_nrows, int B_ncols,
              float *C, int C_nrows, int C_ncols)
{
    cout << "Matrix Multiplication (GPU):" << endl;
    float *C_gpu = multiplyGPU(A, A_nrows, A_ncols,
                               B, B_nrows, B_ncols,
                               C, C_nrows, C_ncols);

    cout << "Matrix Multiplication (CPU):" << endl;
    float *C_cpu = multiplyCPU(A, A_nrows, A_ncols,
                               B, B_nrows, B_ncols,
                               C, C_nrows, C_ncols);
}


// CPU

float* multiplyCPU(float *A, int A_nrows, int A_ncols,
                   float *B, int B_nrows, int B_ncols,
                   float *C, int C_nrows, int C_ncols)
{

    clock_t start, end;

    start = clock();
    for (int i = 0; i < A_nrows; ++i)
    {
        for (int j = 0; j < B_ncols; ++j)
        {
            for (int k = 0; k < A_ncols; ++k)
            {
                C[i*j + j] += A[i*k + k] * B[k * j + j];
            }
        }
    }
    end = clock();

    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    cout << "(CPU) Elapsed Time: " << cpu_time << endl;

    return C;
}

// ----------------------------------------------------------------------------

}


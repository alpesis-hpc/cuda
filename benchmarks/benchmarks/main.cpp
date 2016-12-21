#include "matrix/multiply.h" 


int main(void)
{

    const int nrows = 100;
    const int ncols = 100;
    float A[nrows * ncols];
    float B[nrows * ncols];
    float C[nrows * ncols];

    for (int i = 0; i < (nrows * ncols); ++i)
    {
        A[i] = 5;
        B[i] = 5;
        C[i] = 0;
    }

    matrix::multiply(A, nrows, ncols,
                     B, nrows, ncols,
                     C, nrows, ncols);


    return 0;
}

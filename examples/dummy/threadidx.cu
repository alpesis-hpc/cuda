#include <stdio.h>
#include "cuda_runtime.h"

__global__ void threadIdxPrinter (size_t n)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    printf ("index: %d, blockIdx: %d, threadIdx: %d, blockDim: %d, gridDim: %d\n", i, blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
  }
}


int main (void)
{
  threadIdxPrinter<<<1, 1024>>>(2048);
  cudaDeviceSynchronize();
}

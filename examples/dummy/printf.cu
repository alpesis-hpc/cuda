#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel (void)
{
  printf("Hello World from GPU.\n");
}


int main ()
{
  printf ("Hello World from CPU.\n");
  helloKernel<<<1, 10>>>();
  cudaDeviceReset();

  return 0;
}

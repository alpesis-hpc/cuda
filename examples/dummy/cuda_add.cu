#include <stdio.h>

__global__ void add(int a, int b, int * c)
{
  *c = a + b;
}


int main (void)
{
  int c;
  int * d_c;
  cudaMalloc((void**)&d_c, sizeof(int));
  add<<<1, 1>>>(2, 7, d_c);
  cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
  printf ("2+7=%d\n", c);
  cudaFree (d_c);

  return 0;
}

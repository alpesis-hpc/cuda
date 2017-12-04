#include <stdlib.h>
#include <stdio.h>


typedef struct
{
  float a;
  float b;
} point;


__global__ void structkernel (point * p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  p[i].a = 1.1;
  p[i].b = 2.2;
}


void eval (point * point_cpu, const int N)
{
  for (int i = 0; i < N; ++i)
  {
    //TEST_ASSERT_EQUAL_FLOAT (point_cpu[i].a, 1.1);
    //TEST_ASSERT_EQUAL_FLOAT (point_cpu[i].b, 2.2);
    printf("point_cpu[%d].a: %f\n", i, point_cpu[i].a);
  }
}


int main (void)
{
  int n_points = 16;
  int n_bytes = n_points * sizeof(point);

  int blocksize = 4;
  int gridsize = n_points / blocksize; 

  point * point_cpu;
  point * point_gpu;
  point_cpu = (point*)malloc(n_bytes);
  cudaMalloc((void**)&point_gpu, n_bytes);

  structkernel<<<gridsize, blocksize>>>(point_gpu);
  cudaMemcpy (point_cpu, point_gpu, n_bytes, cudaMemcpyDeviceToHost);
 
  eval (point_cpu, n_points);
 
  cudaFree (point_gpu);
  free (point_cpu);

  return 0;
}

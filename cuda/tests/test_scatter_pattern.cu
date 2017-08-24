#include <stdio.h>
#include "stdlib.h"
#include <unity.h>

int add_cpu (float * a, unsigned int n_a, float * b, unsigned int n_b)
{
  unsigned index = 0;
  for (int i = 0; i < n_a; ++i)
  {
    b[index] = a[i];
    b[index+1] = a[i+1]; 
    index += 2;
  }

  if (index-2 != n_b)
  {
    printf ("index (%d) != n_b (%d)\n", index, n_b);
    exit (EXIT_FAILURE);
  }

  return 0;
}


void eval (float * b, float * b_gpu, unsigned int n_b)
{
  for (int i = 0; i < n_b; ++i)
    TEST_ASSERT_EQUAL_FLOAT (b[i], b_gpu[i]); 
}


void data_printer (float * a, unsigned n)
{
  for (int i = 0; i < n; ++i)
    printf ("%f\n", a[i]);
}


__global__ void map (float * d_a, unsigned int n_a, float * d_b, unsigned int n_b)
{
  // for (int i = threadIdx.x; i < n_b; i += blockDim.x * gridDim.x)
  // {
  int i = threadIdx.x;
  d_b[i] = d_a[0] + 1; 
  //}
}



int main (void)
{
  unsigned int n_a = 4;
  unsigned int n_b = 6;
  float * a = (float*)malloc(sizeof(float)*n_a);
  float * b = (float*)malloc(sizeof(float)*n_b);
  float * b_gpu = (float*)malloc(sizeof(float)*n_b);
  for (int i = 0; i < n_a; ++i)
    a[i] = i;
  int status = add_cpu (a, n_a, b, n_b);
  data_printer (b, n_b);

  float * d_a;
  float * d_b;
  cudaMalloc((void**)&d_a, sizeof(float)*n_a);
  cudaMalloc((void**)&d_b, sizeof(float)*n_b);
  cudaMemcpy(d_a, a, sizeof(float)*n_a, cudaMemcpyHostToDevice);
  map<<<1,512>>>(d_a, n_a, d_b, n_b);
  cudaMemcpy(b_gpu, d_b, sizeof(float)*n_b, cudaMemcpyDeviceToHost);

  eval (b, b_gpu, n_b);
  data_printer (b_gpu, n_b);

  cudaFree (d_a);
  cudaFree (d_b);

  // free (a); 
  // free (b);
  // free (b_gpu);
}

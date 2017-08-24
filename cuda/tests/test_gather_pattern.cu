#include <stdio.h>
#include <unity.h>

void add_cpu (float * a, unsigned int n_a, float * b, unsigned int n_b)
{
  for (int i = 0; i < n_a; ++i)
    b[i] = a[i] + a[i+1] + a[i+2];
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


__global__ void add (float * d_a, unsigned int n_a, float * d_b, unsigned int n_b)
{
  for (int i = threadIdx.x; i < n_a; i += blockDim.x * gridDim.x) 
    d_b[i] = d_a[i] + d_a[i+1] + d_a[i+2];
}



int main (void)
{
  unsigned int n_a = 9;
  unsigned int n_b = 7;
  float * a = (float*)malloc(sizeof(float)*n_a);
  float * b = (float*)malloc(sizeof(float)*n_b);
  float * b_gpu = (float*)malloc(sizeof(float)*n_b);
  for (int i = 0; i < n_a; ++i)
    a[i] = i;

  add_cpu (a, n_a, b, n_b);

  float * d_a;
  float * d_b;
  cudaMalloc((void**)&d_a, sizeof(float)*n_a);
  cudaMalloc((void**)&d_b, sizeof(float)*n_b);

  cudaMemcpy(d_a, a, sizeof(float)*n_a, cudaMemcpyHostToDevice);
  add<<<7,1>>>(d_a, n_a, d_b, n_b);
  cudaMemcpy(b_gpu, d_b, sizeof(float)*n_b, cudaMemcpyDeviceToHost);

  eval (b, b_gpu, n_b);
  data_printer (b_gpu, n_b);

  cudaFree (d_a);
  cudaFree (d_b);
  free (a); 
  free (b);
  free (b_gpu);
}

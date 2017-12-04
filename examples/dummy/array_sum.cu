#include <stdio.h>
#include <stdlib.h>

#include "timer.h"
#include "cu_engine.h"


__global__ void array_sum_kernel (float * A, float * B, float * C, const int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) C[i] = A[i] + B[i];
}


void array_sum (float * A, float * B, float * C, const int N)
{
  for (int i = 0; i < N; ++i)
  {
    C[i] = A[i] + B[i];
  }
}


void init_data (float * A, int size)
{
  // generate different seed for random number
  time_t t;
  srand((unsigned) time(&t));
  for (int i = 0; i < size; ++i)
  {
    // rand() & 11111111 = rand()
    A[i] = (float)(rand() & 0xFF) / 10.0f;
  }

  return;
}


void result_eval (float * C_cpu, float * C_gpu, const int N)
{
  double epsilon = 1.0E-8;
  bool match = 1;

  for (int i = 0; i < N; ++i)
  {
    if (abs(C_cpu[i] - C_gpu[i]) > epsilon)
    {
      match = 0;
      printf ("Arrays do not match!\n");
      printf ("host %5.2f gpu %5.2f at currrent %d\n", C_cpu[i], C_gpu[i], i);
      break;
    }
  }

  if (match) printf ("Arrays match.\n\n");

  return;
}


int main (void)
{
  int dev = 0;
  cudaDeviceProp device_prop;
  CHECK(cudaGetDeviceProperties(&device_prop, dev));
  printf ("Using Device %d: %s\n", dev, device_prop.name);
  CHECK(cudaSetDevice(dev));

  // set up data size of vectors
  int n_elements = 1 << 24;
  printf("Vector size: %d\n", n_elements);

  size_t n_bytes = n_elements * sizeof(float);
  printf ("Vector bytes: %d\n", n_bytes);

  float * h_A = (float*)malloc(n_bytes);
  float * h_B = (float*)malloc(n_bytes);
  float * h_ref_cpu = (float*)malloc(n_bytes);
  float * h_ref_gpu = (float*)malloc(n_bytes);

  double tic, elapsed;
  // init data
  tic = seconds();
  init_data (h_A, n_elements);
  init_data (h_B, n_elements);
  elapsed = seconds() - tic;
  printf ("init_data time elapsed: %f sec\n.", elapsed);
  memset (h_ref_cpu, 0, n_bytes);
  memset (h_ref_gpu, 0, n_bytes);

  // malloc device global memory
  float * d_A, * d_B, * d_C;
  CHECK(cudaMalloc((float**)&d_A, n_bytes));
  CHECK(cudaMalloc((float**)&d_B, n_bytes));
  CHECK(cudaMalloc((float**)&d_C, n_bytes));
  // transfer data from host to device
  CHECK(cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_C, h_ref_gpu, n_bytes, cudaMemcpyHostToDevice));


  // invoke kernel at host side
  dim3 block (512);
  dim3 grid ((n_elements + block.x - 1) / block.x);
  tic = seconds();
  array_sum_kernel<<<grid, block>>>(d_A, d_B, d_C, n_elements);
  CHECK(cudaDeviceSynchronize());
  elapsed = seconds() - tic;
  printf ("Execution configure <<<%d, %d>>>\n", grid.x, block.x);
  printf ("array_sum gpu time elapsed: %f sec\n.", elapsed);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(h_ref_gpu, d_C, n_bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_C));
  CHECK(cudaDeviceReset());

  tic = seconds();
  array_sum (h_A, h_B, h_ref_cpu, n_elements);
  elapsed = seconds() - tic;
  printf ("array_sum cpu time elapsed: %f sec\n.", elapsed);
  result_eval (h_ref_cpu, h_ref_gpu, n_elements);

  free (h_A);
  free (h_B);
  free (h_ref_cpu);
  free (h_ref_gpu);

  return 0;
}

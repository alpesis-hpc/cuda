#include <stdio.h>

#include "timer.h"
#include "cu_engine.h"


__global__ void warmingup (float * d_C)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;

  if ((tid / warpSize) % 2 == 0)
  {
    ia = 100.0f;
  }
  else
  {
    ib = 200.0f;
  }
  d_C[tid] = ia + ib;
}


__global__ void math1 (float * d_C)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;

  if (tid % 2 == 0)
  {
    ia = 100.0f;
  }
  else
  {
    ib = 200.0f;
  }
  d_C[tid] = ia + ib;
}


__global__ void math2 (float * d_C)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;

  if ((tid/warpSize) % 2 == 0)
  {
    ia = 100.0f;
  }
  else
  {
    ib = 200.0f;
  }
  d_C[tid] = ia + ib;
}


__global__ void math3 (float * d_C)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;

  bool ipred = (tid % 2 == 0);

  if (ipred)
  {
    ia = 100.0f;
  }
 
  if (!ipred)
  {
    ib = 200.0f;
  }

  d_C[tid] = ia + ib;
}


__global__ void math4 (float * d_C)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;

  int itid = tid >> 5;

  if (itid & 0x01 == 0)
  {
    ia = 100.0f;
  }
  else
  {
    ib = 200.0f;
  }
  d_C[tid] = ia + ib;
}


int main (void)
{
  int dev = 0;
  cudaDeviceProp device_prop;
  CHECK(cudaGetDeviceProperties(&device_prop, dev));
  printf("using Device %d: %s\n", dev, device_prop.name);

  int size = 64;
  int blocksize = 64;
  size_t n_bytes = size * sizeof(float);
  printf("Data size: %d\n", size);

  dim3 block(blocksize, 1);
  dim3 grid((size + block.x - 1)/block.x, 1);
  printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

  float * d_C;
  CHECK(cudaMalloc((float**)&d_C, n_bytes));

  size_t tic, elapsed;
  CHECK(cudaDeviceSynchronize());
  tic = seconds();
  warmingup<<<grid, block>>>(d_C);
  CHECK(cudaDeviceSynchronize());
  elapsed = seconds() - tic;
  printf("warmup<<<%d, %d>>> elapsed %d sec\n", grid.x, block.x, elapsed);
  CHECK(cudaGetLastError());

  tic = seconds();
  math1<<<grid, block>>>(d_C);
  CHECK(cudaDeviceSynchronize());
  elapsed = seconds() - tic;
  printf("math1<<<%d, %d>>> elapsed %d sec\n", grid.x, block.x, elapsed);
  CHECK(cudaGetLastError());

  tic = seconds();
  math2<<<grid, block>>>(d_C);
  CHECK(cudaDeviceSynchronize());
  elapsed = seconds() - tic;
  printf("math2<<<%d, %d>>> elapsed %d sec\n", grid.x, block.x, elapsed);
  CHECK(cudaGetLastError());

  tic = seconds();
  math3<<<grid, block>>>(d_C);
  CHECK(cudaDeviceSynchronize());
  elapsed = seconds() - tic;
  printf("math3<<<%d, %d>>> elapsed %d sec\n", grid.x, block.x, elapsed);
  CHECK(cudaGetLastError());

  tic = seconds();
  math4<<<grid, block>>>(d_C);
  CHECK(cudaDeviceSynchronize());
  elapsed = seconds() - tic;
  printf("math4<<<%d, %d>>> elapsed %d sec\n", grid.x, block.x, elapsed);
  CHECK(cudaGetLastError());

  CHECK(cudaFree(d_C));
  CHECK(cudaDeviceReset());

  return 0;
}

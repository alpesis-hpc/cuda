#include <stdlib.h>

#include "timer.h"
#include "cu_engine.h"


int recursiveReduce (int * data, int const size)
{
  if (size == 1) return data[0];

  int const stride = size / 2;
  for (int i = 0; i < stride; ++i)
  {
    data[i] += data[i + stride];
  }

  return recursiveReduce(data, stride);
}


__global__ void reduce_neighbored (int * g_idata, int * g_odata, unsigned int n)
{
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int * idata = g_idata + blockIdx.x * blockDim.x;
  if (idx >= n) return;
}


int main (void)
{
  int dev = 0;
  cudaDeviceProp device_prop;
  CHECK(cudaGetDeviceProperties(&device_prop, dev));
  printf("device %d: %s \n", dev, device_prop.name);
  CHECK(cudaSetDevice(dev));

  bool is_result = false;

  int size = 1 << 24;
  int blocksize = 512;
  printf("array size: %d\n", size);

  dim3 block (blocksize, 1);
  dim3 grid ((size + block.x - 1)/block.x, 1);
  printf("grid %d block %d\n", grid.x, block.x);

  size_t bytes = size * sizeof(int);
  int * h_idata = (int*)malloc(bytes);
  int * h_odata = (int*)malloc(grid.x * sizeof(int));
  int * tmp = (int*)malloc(bytes);

  for (int i = 0; i < size; ++i)
  {
    // mask off high 2 bytes to force max number to 255
    h_idata[i] = (int)(rand() & 0xFF);
  }

  memcpy (tmp, h_idata, bytes);

  double tic, elapsed;
  int gpu_sum = 0;

  // allocate device memory
  int * d_idata = NULL;
  int * d_odata = NULL;
  CHECK(cudaMalloc((void**)&d_idata, bytes));
  CHECK(cudaMalloc((void**)&d_odata, grid.x*sizeof(int)));

  tic = seconds();
  int cpu_sum = recursiveReduce(tmp, size);
  elapsed = seconds() - tic;
  printf("cpu reduce elapsed %f sec cpu_sum: %d\n", elapsed, cpu_sum);

  return 0;
}

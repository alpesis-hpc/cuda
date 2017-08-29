/* 
 * block_idx (0, 2)
 * thread_idx (3, 1)
 * ix = 3 + 0 * 4 = 3
 * iy = 1 + 2 * 2 = 5
 * global_idx = 5 * 8 + 3 = 43 
 */

#include <stdio.h>
#include "cu_engine.h"


__global__ void print_thread_index_kernel (int * A, const int nx, const int ny)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * nx + ix;

  printf("thread_id (%d, %d), block_id (%d, %d), coordinate (%d, %d) global index %2d ival %2d\n",
         threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}


void print_matrix (int * C, const int nx, const int ny)
{
  int * ic = C;
  printf ("\nMatrix: (%d.%d)\n", nx, ny);

  for (int iy = 0; iy < ny; iy++)
  {
    for (int ix = 0; ix < nx; ix++)
    {
      printf ("%3d", ic[ix]);
    }
    ic += nx;
    printf("\n");
  }

  printf ("\n");
  return;
}


int main (void)
{
  int dev = 0;
  cudaDeviceProp device_prop;
  CHECK(cudaGetDeviceProperties(&device_prop, dev));
  printf("Using device %d: %s\n", dev, device_prop.name);
  CHECK(cudaSetDevice(dev));

  int nx = 8;
  int ny = 6;
  int nxy = nx * ny;
  int n_bytes = nxy * sizeof(float);

  int * h_A = (int*)malloc(n_bytes);
  for (int i = 0; i < nxy; ++i)
  {
    h_A[i] = i;
  }
  print_matrix (h_A, nx, ny);

  int * d_A;
  CHECK(cudaMalloc((void**)&d_A, n_bytes));
  CHECK(cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice));

  dim3 block(4, 2);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  print_thread_index_kernel<<<grid, block>>>(d_A, nx, ny);
  CHECK(cudaGetLastError());
  CHECK(cudaFree(d_A));
  free(h_A);

  CHECK(cudaDeviceReset());

  return 0;
}

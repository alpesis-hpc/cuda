/* 
 *            gridDim
 *             (2, 1, 1)
 * blockIdx              / blockDim
 *   (0, 0, 0) (1, 0, 0) (3, 1, 1)
 * threadIdx 
 *   (0, 0, 0) (1, 0, 0) (2, 0, 0)
 *   (0, 0, 0) (1, 0, 0) (2, 0, 0)
 *
 */


#include <stdio.h>


__global__ void checkIndex (void)
{
  printf("threadIdx: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
  printf("blockIdx: (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

  printf("blockDim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
  printf("gridDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
}


int main (void)
{
  int n_elements = 12;

  dim3 block(3);
  dim3 grid((n_elements + block.x - 1) / block.x);

  printf("grid.x %d gird.y %d grid.z %d\n", grid.x, grid.y, grid.z);
  printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

  checkIndex<<<grid, block>>>();
  cudaDeviceReset();

  return 0;
}

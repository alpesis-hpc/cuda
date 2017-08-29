#include <stdio.h>

#include "cu_engine.h"

int main (void)
{
  int n_elements = 1024;

  dim3 block (1024);
  dim3 grid ((n_elements + block.x - 1) / block.x);
  printf("grid.x %d block.x %d\n", grid.x, block.x);

  // reset block
  block.x = 512;
  grid.x = (n_elements + block.x - 1) / block.x;
  printf("grid.x %d block.x %d\n", grid.x, block.x);

  block.x = 256;
  grid.x = (n_elements + block.x - 1) / block.x;
  printf("grid.x %d block.x %d\n", grid.x, block.x);

  block.x = 128;
  grid.x = (n_elements + block.x - 1) / block.x;
  printf("grid.x %d block.x %d\n", grid.x, block.x);

  CHECK(cudaDeviceReset());

  return 0;
}

#include <stdio.h>

#include "cu_engine.h"
#include "timer.h"

void result_eval(float *hostRef, float *gpuRef, const int N)
{
  double epsilon = 1.0E-8;
  bool match = 1;
  
  for (int i = 0; i < N; i++)
  {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon)
    {
       match = 0;
       printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
       break;
    }
   }
 
   if (match)
     printf("Arrays match.\n\n");
   else
     printf("Arrays do not match.\n\n");
}


void init_data (float * A, const int N)
{
  for (int i = 0; i < N; ++i)
  {
    A[i] = (float)(rand() & 0xFF) / 100.0f;
  }
}


void matrix_sum (float * A, float * B, float * C, const int nx, const int ny)
{
  float * ia = A;
  float * ib = B;
  float * ic = C;

  for (int iy = 0; iy < ny; iy++)
  {
    for (int ix = 0; ix < nx; ix++)
    {
      ic[ix] = ia[ix] + ib[ix];
    }
    ia += nx;
    ib += nx;
    ic += nx;
  }

  return;
} 


__global__ void matrix_sum_2D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * nx + ix;
  
  if (ix < nx && iy < ny)
     MatC[idx] = MatA[idx] + MatB[idx];
}


__global__ void matrix_sum_1D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if(ix < nx)
  {
    for (int iy = 0; iy < ny; iy++)
    {
      int idx = iy * nx + ix;
      MatC[idx] = MatA[idx] + MatB[idx];
    }
  }
}


__global__ void matrix_sum_mix(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = blockIdx.y;
  unsigned int idx = iy * nx + ix;

  if (ix < nx && iy < ny)
      MatC[idx] = MatA[idx] + MatB[idx];
}


void compute_gpu_2D (dim3 block, dim3 grid, float * d_A, float * d_B, float * d_C, int nx, int ny)
{
  double tic = seconds();
  matrix_sum_2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
  CHECK(cudaDeviceSynchronize());
  double elapsed = seconds() - tic;
  printf ("matrix sum gpu <<<(%d,%d), (%d,%d)>>> time elapsed: %f sec\n", grid.x, grid.y, block.x, block.y, elapsed); 
}


void compute_gpu_1D (dim3 block, dim3 grid, float * d_A, float * d_B, float * d_C, int nx, int ny)
{
  double tic = seconds();
  matrix_sum_1D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
  CHECK(cudaDeviceSynchronize());
  double elapsed = seconds() - tic;
  printf ("matrix sum gpu <<<(%d,%d), (%d,%d)>>> time elapsed: %f sec\n", grid.x, grid.y, block.x, block.y, elapsed); 
}


void compute_gpu_mix (dim3 block, dim3 grid, float * d_A, float * d_B, float * d_C, int nx, int ny)
{
  double tic = seconds();
  matrix_sum_mix<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
  CHECK(cudaDeviceSynchronize());
  double elapsed = seconds() - tic;
  printf ("matrix sum gpu <<<(%d,%d), (%d,%d)>>> time elapsed: %f sec\n", grid.x, grid.y, block.x, block.y, elapsed); 
}


int main (void)
{
  int dev = 0;
  cudaDeviceProp device_prop;
  CHECK(cudaGetDeviceProperties(&device_prop, dev));
  printf("Using Device %d: %s\n", dev, device_prop.name);
  CHECK(cudaSetDevice(dev));

  // set up data size
  int nx = 1 << 14;
  int ny = 1 << 14;
  int nxy = nx * ny;
  int n_bytes = nxy * sizeof(float);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  float * h_A = (float*)malloc(n_bytes);
  float * h_B = (float*)malloc(n_bytes);
  float * h_C_cpu = (float*)malloc(n_bytes);
  float * h_C_gpu = (float*)malloc(n_bytes);

  double tic = seconds();
  init_data (h_A, nxy);
  init_data (h_B, nxy);
  double elapsed = seconds() - tic;
  printf ("matrix init time elapsed: %f sec\n", elapsed);  

  memset(h_C_cpu, 0, n_bytes);
  memset(h_C_gpu, 0, n_bytes);

  tic = seconds();
  matrix_sum (h_A, h_B, h_C_cpu, nx, ny);
  elapsed = seconds() - tic;
  printf ("matrix sum cpu time elapsed: %f sec\n", elapsed);  

  float * d_A, * d_B, * d_C;
  CHECK(cudaMalloc((void**)&d_A, n_bytes));
  CHECK(cudaMalloc((void**)&d_B, n_bytes));
  CHECK(cudaMalloc((void**)&d_C, n_bytes));
  CHECK(cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice));

  int dimx = 32;
  int dimy = 32;
  dim3 block(dimx, dimy);
  dim3 grid((nx + block.x - 1)/ block.x, (ny+block.y-1)/block.y);
  compute_gpu_2D (block, grid, d_A, d_B, d_C, nx, ny);

  block.x = 16;
  grid.x = (nx + block.x - 1) / block.x;
  grid.y = (ny + block.y - 1) / block.y;
  compute_gpu_2D (block, grid, d_A, d_B, d_C, nx, ny);

  block.x = 16;
  block.y = 32;
  grid.x = (nx + block.x - 1) / block.x;
  grid.y = (ny + block.y - 1) / block.y;
  compute_gpu_2D (block, grid, d_A, d_B, d_C, nx, ny);

  block.x = 16;
  block.y = 16;
  grid.x = (nx + block.x - 1) / block.x;
  grid.y = (ny + block.y - 1) / block.y;
  compute_gpu_2D (block, grid, d_A, d_B, d_C, nx, ny);

  block.x = 16;
  block.y = 64;
  grid.x = (nx + block.x - 1) / block.x;
  grid.y = (ny + block.y - 1) / block.y;
  compute_gpu_2D (block, grid, d_A, d_B, d_C, nx, ny);

  block.x = 64;
  block.y = 16;
  grid.x = (nx + block.x - 1) / block.x;
  grid.y = (ny + block.y - 1) / block.y;
  compute_gpu_2D (block, grid, d_A, d_B, d_C, nx, ny);


  block.x = 32;
  grid.x = (nx + block.x - 1) / block.x;
  block.y = 1;
  grid.y = 1;
  compute_gpu_1D (block, grid, d_A, d_B, d_C, nx, ny);

  block.x = 64;
  grid.x = (nx + block.x - 1) / block.x;
  block.y = 1;
  grid.y = 1;
  compute_gpu_1D (block, grid, d_A, d_B, d_C, nx, ny);

  block.x = 128;
  grid.x = (nx + block.x - 1) / block.x;
  block.y = 1;
  grid.y = 1;
  compute_gpu_1D (block, grid, d_A, d_B, d_C, nx, ny);

  
  block.x = 32;
  grid.x = (nx + block.x - 1) / block.x;
  block.y = 1;
  grid.y = ny;
  compute_gpu_mix (block, grid, d_A, d_B, d_C, nx, ny);

  block.x = 64;
  grid.x = (nx + block.x - 1) / block.x;
  block.y = 1;
  grid.y = ny;
  compute_gpu_mix (block, grid, d_A, d_B, d_C, nx, ny);

  block.x = 128;
  grid.x = (nx + block.x - 1) / block.x;
  block.y = 1;
  grid.y = ny;
  compute_gpu_mix (block, grid, d_A, d_B, d_C, nx, ny);

  block.x = 256;
  grid.x = (nx + block.x - 1) / block.x;
  block.y = 1;
  grid.y = ny;
  compute_gpu_mix (block, grid, d_A, d_B, d_C, nx, ny);

  block.x = 512;
  grid.x = (nx + block.x - 1) / block.x;
  block.y = 1;
  grid.y = ny;
  compute_gpu_mix (block, grid, d_A, d_B, d_C, nx, ny);

  CHECK(cudaMemcpy(h_C_gpu, d_C, n_bytes, cudaMemcpyDeviceToHost));
  result_eval (h_C_cpu, h_C_gpu, nxy);  

  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_C));

  free (h_A);
  free (h_B);
  free (h_C_cpu);
  free (h_C_gpu);

  return 0;
}

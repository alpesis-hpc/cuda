#include <stdio.h>
#include "cuda_runtime.h"

__global__ void threadIdxPrinter (size_t n, size_t width_col, size_t height_col,
                                  size_t kernel_h, size_t kernel_w,
                                  size_t stride_h, size_t stride_w,
                                  size_t pad_h, size_t pad_w,
                                  size_t width,
                                  size_t height)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    const int h_index = i / width_col;
    const int h_col = h_index % height_col;
    const int w_col = i % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    int data_col_offset = (c_col * height_col + h_col) * width_col + w_col;
    int data_im_offset = (c_im * height + h_offset) * width + w_offset;
    printf ("index %d h_index %d h_col %d w_col %d c_im %d c_col %d h_offset %d w_offset %d data_col_offset %d data_im_offset %d\n", i, h_index, h_col, w_col, c_im, c_col, h_offset, w_offset, data_col_offset, data_im_offset);

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        int h_im = h_offset + i;
        int w_im = w_offset + j;
        printf ("data_im_offset: %d, data_col_offset:%d\n", data_im_offset, data_col_offset);
        data_im_offset += i * width + j;
        data_col_offset += height_col * width_col;
      }
    }
  }
}


int main (void)
{
  size_t channels = 3;
  size_t width_col = 4;
  size_t height_col = 4;
  size_t n = channels * width_col * height_col;
  size_t kernel_h = 2;
  size_t kernel_w = 2;
  size_t stride = 1;
  size_t pad = 0;
  size_t width = 3;
  size_t height = 3;
  threadIdxPrinter<<<1, 48>>>(n, width_col, height_col,
                              kernel_h, kernel_w, stride, stride, pad, pad,
                              width, height);
  cudaDeviceSynchronize();
}

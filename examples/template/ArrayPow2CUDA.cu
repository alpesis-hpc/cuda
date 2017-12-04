#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Array2D.h"
#include "Array2DCUDA.h"

#define BLOCK_SIZE 1024

template <class T>
__global__ void ArrayPow2Kernel(T * in, T * out, size_t N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) out[idx] = in[idx] * in[idx];
}


template <class T>
void ArrayPow2CUDA(Array2D<T>& in, Array2D<T>& result)
{
    std::cout << "Using the GPU version\n";
    Array2D<TypeCUDA<T>> d_in(in);
    std::cout << "in[0] = " << *in.begin() << std::endl;

    size_t N = in.size();
    std::cout << "N = " << N << std::endl;
    ArrayPow2Kernel<<<(N - 1) /BLOCK_SIZE + 1, BLOCK_SIZE>>>(d_in.begin(), d_in.begin(), in.size());
    cudaDeviceSynchronize();
    cudaMemcpy(result.begin(), d_in.begin(), sizeof(T)*N, cudaMemcpyDeviceToHost);
}


template void ArrayPow2CUDA(Array2D<float>&, Array2D<float>&);
template __global__ void ArrayPow2Kernel(float*, float*, size_t);

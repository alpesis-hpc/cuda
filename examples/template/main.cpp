#include <iostream>
#include "Array2D.h"
#include "ArrayPow2.h"

#ifdef ENABLE_GPU
    #include "Array2DCUDA.h"
    #include "ArrayPow2CUDA.cuh"
#endif

template <class T>
using ArrayPow2_F = void(*)(Array2D<T>&, Array2D<T>&);

ArrayPow2_F<float> ArrayPow2;


int main(int argc, char ** argv)
{
#ifdef ENABLE_GPU
    ArrayPow2 = ArrayPow2CUDA;
#else
    ArrayPow2 = ArrayPow2CPU;
#endif

    Array2D<float> data(new float[120], 60, 2);
    int a = 2;
    for (auto& i:data) i = ++a;
    Array2D<float> result(data);
    ArrayPow2(data, result);

    std::cout << "data[0] = " << *data.begin() << std::endl;
    std::cout << "data[0]^2 = " << *result.begin() << std::endl;

    return 0;
}

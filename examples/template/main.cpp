#include <iostream>
#include "Array2D.h"
#include "ArrayPow2.h"


int main(int argc, char ** argv)
{
    Array2D<float> data(new float[120], 60, 2);
    int a = 2;
    for (auto& i:data) i = ++a;
    Array2D<float> result(data);
    ArrayPow2(data, result);

    std::cout << "data[0] = " << *data.begin() << std::endl;
    std::cout << "data[0]^2 = " << *result.begin() << std::endl;

    return 0;
}

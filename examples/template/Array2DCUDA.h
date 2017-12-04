#ifndef ARRAY2DCUDA_H
#define ARRAY2DCUDA_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Array2D.h"

template <class T>
struct TypeCUDA
{
    T val;
};


template <class U>
class Array2D<TypeCUDA<U> >
{
    public:
        Array2D(U * _data,
                const size_t & _nrows,
                const size_t & _ncols);
        Array2D(const Array2D<U>&);
        Array2D<TypeCUDA<U> >& operator = (const Array2D<U>& other);
        ~Array2D();
        size_t get_nrows() const { return * this->nrows; }
        size_t get_ncols() const { return * this->ncols; }
        size_t size() const { return * this->N; }
        U * begin() { return data; }
        U * begin() const { return data; }
        U * end() { return data + this->size(); }
        U * end() const { return data + this->size(); }

    private:
        U * data;
        size_t * nrows;
        size_t * ncols;
        size_t * N;
};


template <class U>
Array2D<TypeCUDA<U> >::Array2D(U * _data,
                              const size_t & _nrows,
                              const size_t & _ncols):data(_data)
{
    size_t N_tmp = _nrows * _ncols;
    
    cudaMalloc((void**)&nrows, sizeof(size_t));
    cudaMalloc((void**)&ncols, sizeof(size_t));
    cudaMalloc((void**)&N, sizeof(size_t));
    cudaMalloc((void**)&data, sizeof(U)*N_tmp);

    cudaMemcpy(nrows, &_nrows, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ncols, &_ncols, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(N, &N_tmp, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(data, _data, sizeof(U)*N_tmp, cudaMemcpyHostToDevice);
};


template <class U>
Array2D<TypeCUDA<U> >::Array2D(const Array2D<U>& other)
{
    size_t N_tmp = other.size();

    cudaMalloc((void**)&nrows, sizeof(size_t));
    cudaMalloc((void**)&ncols, sizeof(size_t));
    cudaMalloc((void**)&N, sizeof(size_t));
    cudaMalloc((void**)&data, sizeof(U)*N_tmp);

    const size_t other_nrows = other.get_nrows();
    const size_t other_ncols = other.get_ncols();
    const size_t other_N = other.size();
    U * other_data = other.begin();

    cudaMemcpy(nrows, &other_nrows, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ncols, &other_ncols, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(N, &other_N, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(data, other_data, sizeof(U)*N_tmp, cudaMemcpyHostToDevice);
};


template <class U>
Array2D<TypeCUDA<U> >& Array2D<TypeCUDA<U> >::operator = (const Array2D<U>& other)
{
    size_t N_tmp = other.size();

    cudaMalloc((void**)&nrows, sizeof(size_t));
    cudaMalloc((void**)&ncols, sizeof(size_t));
    cudaMalloc((void**)&N, sizeof(size_t));
    cudaMalloc((void**)&data, sizeof(U)*N_tmp);

    const size_t other_nrows = other.get_nrows();
    const size_t other_ncols = other.get_ncols();
    const size_t other_N = other.size();
    U * other_data = other.begin();

    cudaMemcpy(nrows, &other_nrows, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ncols, &other_ncols, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(N, &other_N, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(data, other_data, sizeof(U)*N_tmp, cudaMemcpyHostToDevice);

    return * this;
};


template <class U>
Array2D<TypeCUDA<U> >::~Array2D()
{
    cudaFree(nrows);
    cudaFree(ncols);
    cudaFree(N);
    cudaFree(data);
};

#endif

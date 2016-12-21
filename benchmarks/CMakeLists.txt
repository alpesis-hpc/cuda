CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

# -------------------------------------------------------------------
# Project settings

PROJECT ( GPUBENCHMARKS )
# SET ( GPUBENCHMARKS_VERSION_MAJOR 1 )
# SET ( GPUBENCHMARKS_VERSION_MINOR 0 )

# -------------------------------------------------------------------
# Libraries

# cuda
INCLUDE ( cmake/FindCUDA.cmake )

# -------------------------------------------------------------------
# Compile settings

# c++
SET ( CMAKE_CXX_STANDARD 11 )

# nvcc
LIST ( APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE" )

# -------------------------------------------------------------------
# Include settings

# include all files
FILE ( GLOB SOURCES
    "${PROJECT_SOURCE_DIR}/benchmarks/*.h"
    "${PROJECT_SOURCE_DIR}/benchmarks/*.cpp"
    "${PROJECT_SOURCE_DIR}/benchmarks/*.cu"
)

# -------------------------------------------------------------------
# Executable settings

CUDA_ADD_EXECUTABLE ( main ./benchmarks/main.cpp
    ./benchmarks/matrix/multiply.cpp
    ./benchmarks/matrix/multiply.cu
)

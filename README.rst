##############################################################################
CUDA
##############################################################################

Example List:

- template_arraypow2

::

    $ cd examples/<example>
    $ make all

==============================================================================
Getting Started
==============================================================================

::

    $ make all
    TEST _build/tests/test_dummy
    TEST _build/tests/checkDeviceInfor
    TEST _build/tests/simpleDeviceQuery
    TEST _build/tests/sumMatrix
    TEST _build/tests/test_array_sum
    TEST _build/tests/test_cuda_add
    TEST _build/tests/test_cuda_dummy
    TEST _build/tests/test_define_grid_block
    TEST _build/tests/test_device_query
    TEST _build/tests/test_matrix_sum
    TEST _build/tests/test_printf
    TEST _build/tests/test_reduce_integer
    TEST _build/tests/test_simple_divergence
    TEST _build/tests/test_struct
    TEST _build/tests/test_thread_dimensions
    TEST _build/tests/test_threadidx_communications
    TEST _build/tests/test_threadidx
    TEST _build/tests/test_thread_index


Profiling

::

    # nvidia profiling: nvprof ./_build/tests/test_array_sum
    $ nvprof <path/to/exec>
    $ nvprof --metrics branch_efficiency ./_build/tests/test_simple_divergence
    $ nvprof --metrics achieved_occupancy ./sumMatrix 32 32
    $ nvprof --metrics gld_throughput./sumMatrix 32 32
    $ nvprof --metrics gld_efficiency ./sumMatrix 32 32

    # nvidia-smi
    $ nvidia-smi -L
    $ nvidia-smi -q -i 0
    $ nvidia-smi -q -i 0 -d MEMORY | tail -n 5
    $ nvidia-smi -q -i 0 -d UTILIZATION | tail -n 4

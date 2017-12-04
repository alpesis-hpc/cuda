##############################################################################
CUDA
##############################################################################

==============================================================================
Example List
==============================================================================


CUDA syntax

- template_arraypow2

CUDA Advanced

- global_memory
- shared_constant_memory
- streams_and_concurrency
- tuning_primitives
- cuda_libs
- multi_gpu
- optimization_and_debugging

CUDA dummy (for testing)

::

    $ cd examples/<example>
    $ make all


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

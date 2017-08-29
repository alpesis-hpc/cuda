##############################################################################
CUDA
##############################################################################

==============================================================================
Getting Started
==============================================================================

::

    $ make all

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

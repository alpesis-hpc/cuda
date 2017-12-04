CU_APPS=array_sum checkDeviceInfor cuda_add cuda_dummy \
		define_grid_block device_query matrix_sum printf \
		reduce_integer simpleDeviceQuery simple_divergence struct sumMatrix \
		thread_dimensions threadidx_communications threadidx thread_index
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}


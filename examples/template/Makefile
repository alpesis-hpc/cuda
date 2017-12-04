BUILD_DIR = _build

CC = g++
CC_OPTS = -std=c++11
CC_FLAGS = $(CC_OPTS)

NVCC = nvcc
NVCC_OPTS = -std=c++11

all: cpu gpu

cpu:
	$(CC) $(CC_FLAGS) main.cpp -o $(BUILD_DIR)/demo_cpu	
	$(BUILD_DIR)/demo_cpu
	
gpu:
	$(NVCC) $(NVCC_FLAGS) main.cpp -D ENABLE_GPU $(BUILD_DIR)/demo_gpu

build:
	mkdir $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)/*

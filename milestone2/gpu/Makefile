NVCC      = nvcc
# sm_80 = A100 (Midway3 GPU partition). Change to sm_70 for V100 if needed.
NVCCFLAGS = -O3 -arch=sm_80

# Mixed C/CUDA build: compile mnist.c with gcc, link everything with nvcc
CC        = gcc
CFLAGS    = -O3

SRC_CU  = src/main.cu src/gpu_network.cu src/gpu_gemm.cu
SRC_C   = src/mnist.c
OBJ_C   = mnist.o

TARGET  = nn_gpu

all: $(TARGET)

# Compile the pure-C file separately
$(OBJ_C): $(SRC_C)
	$(CC) $(CFLAGS) -c $(SRC_C) -o $(OBJ_C)

# Compile + link CUDA files, then link in the C object
$(TARGET): $(SRC_CU) $(OBJ_C)
	$(NVCC) $(NVCCFLAGS) $(SRC_CU) $(OBJ_C) -lm -o $(TARGET)

clean:
	rm -f $(TARGET) $(OBJ_C)

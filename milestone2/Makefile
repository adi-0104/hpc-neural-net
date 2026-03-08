CC     = gcc
CFLAGS =  -O3 -Wall -fopenmp -march=native
LDFLAGS = -lm

ifdef BLAS
    CFLAGS  += -DUSE_BLAS
    LDFLAGS += -lopenblas
endif

# Toggle: make TILE=32 / TILE=64 / TILE=128  (default 64 in gemm.h)
ifdef TILE
    CFLAGS += -DTILE=$(TILE)
endif

SRC    = src/main.c src/network.c src/gemm.c src/mnist.c
TARGET = nn

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

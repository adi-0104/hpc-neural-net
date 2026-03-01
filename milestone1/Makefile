CC = gcc
CFLAGS = -O3 -Wall -lm
SRC = src/main.c src/network.c src/mnist.c
TARGET = nn

all: $(TARGET)
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)
clean:
	rm -f $(TARGET)

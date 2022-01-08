#build main.c

#compiler
CC = nvcc
#target file name
LIB_NAME = cuda-neural-network
all:
	$(CC) ./src/$(LIB_NAME).cu -c -g -o ./bin/$(LIB_NAME).o -I ./includes
	$(CC) ./tests/main.cu ./bin/* -o ./tests/main.exe -I ./includes
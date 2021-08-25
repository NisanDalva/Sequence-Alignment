CUFLAGS = --compiler-options -Wall
LIBS = -lm
O_FILES = main.o general.o cudaFunctions.o

default: build

build:
	mpicxx -fopenmp -c $(LIBS) main.c -o main.o
	mpicxx -fopenmp -c $(LIBS) general.c -o general.o
	nvcc -I./inc -c $(LIBS) $(CUFLAGS) cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o prog $(O_FILES) /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./prog

run:
	mpiexec -np 2 ./prog $(file_name)

runOn2:
	mpiexec -np 2 -machinefile mf -map-by node ./prog $(file_name)
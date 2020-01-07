all: nbody nbodycu nbodyacc

nbody: nbody.c
	gcc nbody.c -o nbody -Wall -Wextra -lm -fopenmp -O3

nbodyacc: nbodyacc.c
	gcc nbodyacc.c -o nbodyacc -Wall -Wextra -lm -acc -O3

nbodycu: nbody.cu
	nvcc nbody.cu -o nbodycu -g -lm -O3

clean:
	rm -f nbody
	rm -f nbodycu
	rm -f nbodyacc


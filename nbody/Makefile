all: nbody nbodycu nbodyomp nbodysoa nbodysoacu nbodyacc
run: runnbody runnbodycu runnbodyomp runnbodysoa runnbodysoacu runnbodyacc

# Compilation
nbody: nbody.c
	gcc $< -o $@ -Wall -Wextra -fopenmp -O3 -lm 

nbodyacc: nbodyacc.c
	pgcc $< -o $@ -g -acc -ta=nvidia -Minfo=accel -O3 -lm

nbodyomp: nbodyomp.c
	gcc $< -o $@ -Wall -Wextra -fopenmp -O3 -lm

nbodycu: nbody.cu
	nvcc $< -o $@ -g -O3 -Xcompiler -fopenmp -lm

nbodysoa: nbodysoa.c
	gcc $< -o $@ -Wall -Wextra -fopenmp -O3 -lm 

nbodysoacu: nbodysoa.cu
	nvcc $< -o $@ -O3 -Xcompiler -fopenmp -lm 

# Execution
runnbody: nbody
	./$<

runnbodyacc: nbodyacc
	./$<

runnbodyomp: nbodyomp
	./$<

runnbodycu: nbodycu
	./$<

runnbodysoa: nbodysoa
	./$<

runnbodysoacu: nbodysoacu
	./$<

# Benchmarking
#TODO

clean:
	rm -f nbody
	rm -f nbodycu
	rm -f nbodyacc
	rm -f nbodyomp
	rm -f nbodysoa
	rm -f nbodysoacu
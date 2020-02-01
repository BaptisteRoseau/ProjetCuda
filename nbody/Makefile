help:
	@echo "\
	Compile sources:\n\
		- make all:           Compiles all sources\n\
		- make nbody:         Compiles nbody\n\
		- make nbodyacc:      Compiles nbodyacc\n\
		- make nbodyomp:      Compiles nbodyomp\n\
		- make nbodycu:       Compiles nbodycu\n\
		- make nbodysoa:      Compiles nbodysoa\n\
		- make nbodysoacu:    Compiles nbodysoacu\n\
	Run binaries:\n\
		- make run:           Runs all binaries\n\
		- make runnbody:      Runs nbody\n\
		- make runnbodyacc:   Runs nbodyacc\n\
		- make runnbodyomp:   Runs nbodyomp\n\
		- make runnbodycu:    Runs nbodycu\n\
		- make runnbodysoa:   Runs nbodysoa\n\
		- make runnbodysoacu: Runs nbodysoacu\
	"

# Alises for all binaries
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

clean:
	rm -f nbody nbodycu nbodyacc nbodyomp nbodysoa nbodysoacu

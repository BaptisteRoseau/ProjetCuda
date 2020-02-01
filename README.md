Author: Baptiste Roseau

## Nbodies CUDA college projet

### Requirement

- GCC
- CUDA 9.2 or higher
- PGI 19.10 or higher
- Python3 (Numpy and Matplotlib), used for benchmark only.


### Description

This project provides multiple implementations of a N-bodies interraction simulation.

The implementations have been made in single-thread, OpenMP, OpenACC and CUDA:

- nbody.c:     First implementation made by the professor.
- nbody.cu:    Naive CUDA portage of nbody.c
- nbodyomp.c:  OpenMP version of nbody.c
- nbodyacc.c:  OpenACC version of nbody.c
- nbodysoa.c:  Structure of Array version of nbody.c
- nbodysoa.cu: CUDA portage of nbodysoa.c


### How to use

These verions can be compiled and run with the Makefile provided in `nbody`.

Please run `cd nbody && make all` to compile all the sources. You can also use `make help` to see all the available commands.

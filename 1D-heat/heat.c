#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void spmv(int N, // number of matrix rows
          int k_max, // max. number of nonzero columns per row
          double *a, // array of nonzero column entries
          int *j, // array of nonzero column indices
          double *x, // input vector x
          double *y // result vector y
	  )
{
    int i, k;

    for(i = 0; i < N; i++)
    {
        y[i] = 0;
        for(k = 0; k < k_max; k++)
        {
            y[i] += a[k_max*i+k] * x[j[k_max*i+k]];
        }
    }
}

int main(int argc, char **argv)
{
    int N = 10000;    
    int nsteps = 1000;
    double eps = 0.0001;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) nsteps = atoi(argv[2]);
    if (argc > 3) eps = atof(argv[3]);

    double dx = 1.0;
    double c = 2.0;

    double dt = 0.8*(dx*dx/(2.0*c));

    int k_max = 3;
    int Annz = k_max * N;

    double *Avalues = (double *)malloc(k_max*N*sizeof(double));
    int *Acols = (int *)malloc(k_max*N*sizeof(int));

    Avalues[k_max*0 + 0] =  2.0;
    Avalues[k_max*0 + 1] = -1.0;
    Avalues[k_max*0 + 2] =  0.0; // dummy

    Acols[k_max*0 + 0] = 0;
    Acols[k_max*0 + 1] = 1;
    Acols[k_max*0 + 2] = 2; // dummy
 
    int row;
    for (row = 1; row < N-1; row++)
    {
        Avalues[k_max*row + 0] = -1.0;
        Avalues[k_max*row + 1] =  2.0;
        Avalues[k_max*row + 2] = -1.0;

        Acols[k_max*row + 0] = row - 1;
        Acols[k_max*row + 1] = row;
        Acols[k_max*row + 2] = row + 1;
    }

    Avalues[k_max*(N-1) + 0] = -1.0;
    Avalues[k_max*(N-1) + 1] =  2.0;
    Avalues[k_max*(N-1) + 2] =  0.0; // dummy

    Acols[k_max*0 + 0] = N-2;
    Acols[k_max*0 + 1] = N-1;
    Acols[k_max*0 + 2] = N-3; // dummy

    int step = 0;
    double *x = (double *)calloc(N, sizeof(double));
    double *y = (double *)calloc(N, sizeof(double));
 
    int ii;
    for (ii = 0; ii < N; ii++)
    {
        x[ii] = (ii < (N/2))?1.0:0.0;
    }

    FILE *f;
    f = fopen("input.txt", "w+");
    for (ii = 0; ii < N; ii++)
    {
        fprintf(f, "%e	%e\n", (1.0*ii)/N, x[ii]);
    }
    fclose(f);

    double err = 2.0*eps;
    while (step < nsteps && err > eps)
    {
        // y <- Ax
	spmv(N, k_max, Avalues, Acols, x, y); 

	// ||y||_2
	double norm2 = 0.0;
	for (ii = 1; ii < N-1; ii++)
	{
            norm2 += y[ii]*y[ii]; 
	}
	err = sqrt(norm2);

	// x <- x - c * dt/(dx)**2 y
        for (ii = 0; ii < N; ii++)
	{
            x[ii] -= c*dt/(dx*dx)*y[ii];
	}

	x[0] = 1.0;
	x[N-1] = 0.0;
	
	step++;
    }

    fprintf(stderr, "step : %d\n", step);
    fprintf(stderr, "Err : %e\n", err);

    f = fopen("output.txt", "w+");
    for (ii = 0; ii < N; ii++)
    {
        fprintf(f, "%e	%e\n", (1.0*ii)/N, x[ii]);
    }
    fclose(f);

    free(x);
    free(y);
    free(Avalues);
    free(Acols);

    return 0;
}

#include <stdio.h>
#include <math.h>
#include <stdlib.h> // drand48
#include <sys/time.h>
#include <cuda.h>
#include <omp.h>

//#define DUMP

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
    fprintf(stderr,"GPUassert: %s in file %s: %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct ParticleType { 
  float x, y, z;
  float vx, vy, vz; 
};

__global__
void cuMoveParticles(const int nParticles, struct ParticleType *particle, const float dt) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int i = tid;
  while (i < nParticles){
    // Components of the gravity force on particle i
    float Fx = 0, Fy = 0, Fz = 0; 
      
    // Loop over particles that exert force
    int j;
    for (j = 0; j < nParticles; j++) { 
      // No self interaction
      if (i != j) {
          // Avoid singularity and interaction with self
          const float softening = 1e-20;

          // Newton's law of universal gravity
          const float dx = particle[j].x - particle[i].x;
          const float dy = particle[j].y - particle[i].y;
          const float dz = particle[j].z - particle[i].z;
          const float drSquared  = dx*dx + dy*dy + dz*dz + softening;
          const float drPower32  = pow(drSquared, 3.0/2.0);
            
          // Calculate the net force
          Fx += dx / drPower32;  
          Fy += dy / drPower32;  
          Fz += dz / drPower32;
      }
    }

    // Accelerate particles in response to the gravitational force
    particle[i].vx += dt*Fx;
    particle[i].vy += dt*Fy;
    particle[i].vz += dt*Fz;

    // Getting next particle for current thread
    i += blockDim.x*gridDim.x;
  }

  // Waiting for all the particles to be moved
  __syncthreads();

  // Updating particles position
  i = tid;
  while (i < nParticles){
    particle[i].x += particle[i].vx*dt;
    particle[i].y += particle[i].vy*dt;
    particle[i].z += particle[i].vz*dt;
    i += blockDim.x*gridDim.x;
  }
}

void dump(int iter, int nParticles, struct ParticleType* particle)
{
    char filename[64];
    snprintf(filename, 64, "output_%d.txt", iter);

    FILE *f;
    f = fopen(filename, "w+");

    int i;
    for (i = 0; i < nParticles; i++)
    {
        fprintf(f, "%e %e %e %e %e %e\n",
                   particle[i].x, particle[i].y, particle[i].z,
		   particle[i].vx, particle[i].vy, particle[i].vz);
    }

    fclose(f);
}

int main(const int argc, const char** argv)
{

  // Problem size and other parameters
  const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
  // Duration of test
  const int nSteps = (argc > 2)?atoi(argv[2]):10;
  // Particle propagation time step
  const float dt = 0.0005f;

  struct ParticleType* particle = (struct ParticleType*) malloc(nParticles*sizeof(struct ParticleType));

  // Initialize random number generator and particles
  srand48(0x2020);

  int i;
  for (i = 0; i < nParticles; i++)
  {
     particle[i].x =  2.0*drand48() - 1.0;
     particle[i].y =  2.0*drand48() - 1.0;
     particle[i].z =  2.0*drand48() - 1.0;
     particle[i].vx = 2.0*drand48() - 1.0;
     particle[i].vy = 2.0*drand48() - 1.0;
     particle[i].vz = 2.0*drand48() - 1.0;
  }
  
  // Copying particles into the GPU
  struct ParticleType* cuParticles;
  gpuErrchk( cudaMalloc((void**)&cuParticles, nParticles*sizeof(ParticleType)) );
  gpuErrchk( cudaMemcpy(cuParticles, particle, nParticles*sizeof(ParticleType), cudaMemcpyHostToDevice) );


  // Getting max occupancy
  int blockSize, minGridSize, gridSize;
  gpuErrchk( cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, cuMoveParticles, 0, nParticles) ); 
  gridSize = (nParticles + blockSize - 1) / blockSize; 

  // Perform benchmark
  printf("\nPropagating %d particles using %d grids of %d thread...\n\n", 
	 nParticles, gridSize, blockSize
	 );
  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration (warm-up)
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  int step;
  for (step = 1; step <= nSteps; step++) {

    const double tStart = omp_get_wtime(); // Start timing
    cuMoveParticles<<< gridSize, blockSize >>>(nParticles, cuParticles, dt);
    cudaDeviceSynchronize();
    const double tEnd = omp_get_wtime(); // End timing
    const double tElapsed = (tEnd - tStart); // seconds

    const float HztoInts   = ((float)nParticles)*((float)(nParticles-1)) ;
    const float HztoGFLOPs = 20.0*1e-9*((float)(nParticles))*((float)(nParticles-1));

    if (step > skipSteps) { // Collect statistics
      rate  += HztoGFLOPs/tElapsed; 
      dRate += HztoGFLOPs*HztoGFLOPs/(tElapsed*tElapsed); 
    }

    printf("%5d %10.3e %10.3e %8.1f %s\n", 
	   step, tElapsed, HztoInts/tElapsed, HztoGFLOPs/tElapsed, (step<=skipSteps?"*":""));
    fflush(stdout);

#ifdef DUMP
    gpuErrchk( cudaMemcpy(particle, cuParticles, nParticles*sizeof(ParticleType), cudaMemcpyDeviceToHost) );
    dump(step, nParticles, particle);
#endif
  }
  rate/=(double)(nSteps-skipSteps); 
  dRate=sqrt(dRate/(double)(nSteps-skipSteps)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");
  free(particle);
  gpuErrchk( cudaFree(cuParticles) );

  return 0;
}



#include <stdio.h>
#include <math.h>
#include <stdlib.h> // drand48
#include <sys/time.h>
#include <omp.h>

//#define DUMP

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s in file %s: %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct ParticleType { 
  float *x, *y, *z;
  float *vx, *vy, *vz; 
  size_t nParticles;
};

void mallocParticleType(ParticleType *p, size_t amount){
  p->x =  (float*) malloc(amount*sizeof(float));
  p->y =  (float*) malloc(amount*sizeof(float));
  p->z =  (float*) malloc(amount*sizeof(float));
  p->vx = (float*) malloc(amount*sizeof(float));
  p->vy = (float*) malloc(amount*sizeof(float));
  p->vz = (float*) malloc(amount*sizeof(float));
}

void freeParticleType(ParticleType *p){
  free(p->x);
  free(p->y);
  free(p->z);
  free(p->vx);
  free(p->vy);
  free(p->vz);
  free(p);
}

__global__
void cuMoveParticles(struct ParticleType const *particles, const float dt) {
  const int nParticles = particles->nParticles;
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int i = tid;
  while (i < nParticles){
    // Components of the gravity force on particle i
    float Fx = 0, Fy = 0, Fz = 0; 
      
    // Loop over particles that exert force
    for (int j = 0; j < nParticles; j++) {  //We could use reduction here
      // No self interaction
      if (i != j) {
          // Avoid singularity and interaction with self
          const float softening = 1e-20;

          // Newton's law of universal gravity
          const float dx = particles->x[j] - particles->x[i];
          const float dy = particles->y[j] - particles->y[i];
          const float dz = particles->z[j] - particles->z[i];
          const float drSquared  = dx*dx + dy*dy + dz*dz + softening;
          const float drPower32  = pow(drSquared, 3.0/2.0);
            
          // Calculate the net force
          Fx += dx / drPower32;  
          Fy += dy / drPower32;  
          Fz += dz / drPower32;
      }
    }

    // Accelerate particles in response to the gravitational force
    particles->vx[i] += dt*Fx; 
    particles->vy[i] += dt*Fy; 
    particles->vz[i] += dt*Fz;

    // Getting next particle for current thread
    i += blockDim.x*gridDim.x;
  }

  // Waiting for all the particles to be moved
  __syncthreads();

  // Updating particles position
  i = tid;
  while (i < nParticles){
    particles->x[i] += particles->vx[i]*dt;
    particles->y[i] += particles->vy[i]*dt;
    particles->z[i] += particles->vz[i]*dt;
    i += blockDim.x*gridDim.x;
  }
}

void dump(int iter, struct ParticleType *particles)
{
    const int nParticles = particles->nParticles;

    char filename[64];
    snprintf(filename, 64, "output_%d.txt", iter);

    FILE *f;
    f = fopen(filename, "w+");

    int i;
    for (i = 0; i < nParticles; i++)
    {
        fprintf(f, "%e %e %e %e %e %e\n",
                   particles->x[i], particles->y[i], particles->z[i],
		   particles->vx[i], particles->vy[i], particles->vz[i]);
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

  struct ParticleType *particles = (struct ParticleType *) malloc(sizeof(ParticleType));
  particles->nParticles = nParticles;
  mallocParticleType(particles, nParticles);
  
  // Initialize random number generator and particles
  srand48(0x2020);
  
  int i;
  for (i = 0; i < nParticles; i++)
  {
    particles->x[i] =  2.0*drand48() - 1.0;
    particles->y[i] =  2.0*drand48() - 1.0;
    particles->z[i] =  2.0*drand48() - 1.0;
    particles->vx[i] = 2.0*drand48() - 1.0;
    particles->vy[i] = 2.0*drand48() - 1.0;
    particles->vz[i] = 2.0*drand48() - 1.0;
  }
  
  // Allocating data onto GPU (structure with pointers is on host)
  struct ParticleType *deviceParticlesPointers = (struct ParticleType*) malloc(sizeof(struct ParticleType));
  deviceParticlesPointers->nParticles = nParticles;

  gpuErrchk( cudaMalloc((void**)&deviceParticlesPointers->x, nParticles*sizeof(float)) );
  gpuErrchk( cudaMalloc((void**)&deviceParticlesPointers->y, nParticles*sizeof(float)) );
  gpuErrchk( cudaMalloc((void**)&deviceParticlesPointers->z, nParticles*sizeof(float)) );
  gpuErrchk( cudaMalloc((void**)&deviceParticlesPointers->vx, nParticles*sizeof(float)) );
  gpuErrchk( cudaMalloc((void**)&deviceParticlesPointers->vy, nParticles*sizeof(float)) );
  gpuErrchk( cudaMalloc((void**)&deviceParticlesPointers->vz, nParticles*sizeof(float)) );

  gpuErrchk( cudaMemcpy(deviceParticlesPointers->x, particles->x, nParticles*sizeof(float), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(deviceParticlesPointers->y, particles->y, nParticles*sizeof(float), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(deviceParticlesPointers->z, particles->z, nParticles*sizeof(float), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(deviceParticlesPointers->vx, particles->vx, nParticles*sizeof(float), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(deviceParticlesPointers->vy, particles->vy, nParticles*sizeof(float), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(deviceParticlesPointers->vz, particles->vz, nParticles*sizeof(float), cudaMemcpyHostToDevice) );
  
  // Copying host structure with device pointers onto device
  struct ParticleType *cuParticles;
  gpuErrchk( cudaMalloc((void**)&cuParticles, sizeof(ParticleType)) );
  gpuErrchk( cudaMemcpy(cuParticles, deviceParticlesPointers, sizeof(ParticleType), cudaMemcpyHostToDevice) );
  
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
  for (int step = 1; step <= nSteps; step++) {

    const double tStart = omp_get_wtime(); // Start timing
    cuMoveParticles<<< gridSize, blockSize >>>(cuParticles, dt);
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
    gpuErrchk( cudaMemcpy(particles->x, deviceParticlesPointers->x, nParticles*sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(particles->y, deviceParticlesPointers->y, nParticles*sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(particles->z, deviceParticlesPointers->z, nParticles*sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(particles->vx, deviceParticlesPointers->vx, nParticles*sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(particles->vy, deviceParticlesPointers->vy, nParticles*sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(particles->vz, deviceParticlesPointers->vz, nParticles*sizeof(float), cudaMemcpyDeviceToHost) );
    dump(step, particles);
#endif
  }


  rate/=(double)(nSteps-skipSteps); 
  dRate=sqrt(dRate/(double)(nSteps-skipSteps)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");

  // Releasing memory
  freeParticleType(particles);
  gpuErrchk( cudaFree(deviceParticlesPointers->x) );
  gpuErrchk( cudaFree(deviceParticlesPointers->y) );
  gpuErrchk( cudaFree(deviceParticlesPointers->z) );
  gpuErrchk( cudaFree(deviceParticlesPointers->vx) );
  gpuErrchk( cudaFree(deviceParticlesPointers->vy) );
  gpuErrchk( cudaFree(deviceParticlesPointers->vz) );
  gpuErrchk( cudaFree(cuParticles) );
  free(deviceParticlesPointers);

  return 0;
}



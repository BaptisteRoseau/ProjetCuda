#include <stdio.h>
#include <math.h>
#include <stdlib.h> // drand48
#include <sys/time.h>
#include <cuda.h>

//#define DUMP

//TODO: Checker les retours d'erreur
//TODO: Suivre le sujet --"

struct ParticleType { 
  float *x, *y, *z;
  float *vx, *vy, *vz; 
  size_t nParticles;
};

float get_time(){
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_usec;
}

void allocParticleType(ParticleType *p, size_t amount){
  p->nParticles = amount;
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
}

void cudaMallocParticleType(ParticleType *pDevice, size_t amount){
  cudaMalloc((void**)&pDevice, sizeof(ParticleType));
  //size_t tmp = amount;
  //cudaMemcpy(&pDevice->nParticles, &tmp, sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&pDevice->x, amount*sizeof(float));
  cudaMalloc((void**)&pDevice->y, amount*sizeof(float));
  cudaMalloc((void**)&pDevice->z, amount*sizeof(float));
  cudaMalloc((void**)&pDevice->vx, amount*sizeof(float));
  cudaMalloc((void**)&pDevice->vy, amount*sizeof(float));
  cudaMalloc((void**)&pDevice->vz, amount*sizeof(float));
}

void cudaFreeParticleType(ParticleType *pDevice){
  cudaFree(pDevice->x);
  cudaFree(pDevice->y);
  cudaFree(pDevice->z);
  cudaFree(pDevice->vx);
  cudaFree(pDevice->vy);
  cudaFree(pDevice->vz);
  cudaFree(pDevice);
}

void cudaMemcpyParticleType(ParticleType *pDevice, const ParticleType *pHost, cudaMemcpyKind kind){
  cudaMemcpy(pDevice->x, pHost->x, pHost->nParticles*sizeof(float), kind);
  cudaMemcpy(pDevice->y, pHost->y, pHost->nParticles*sizeof(float), kind);
  cudaMemcpy(pDevice->z, pHost->z, pHost->nParticles*sizeof(float), kind);
  cudaMemcpy(pDevice->vx, pHost->vx, pHost->nParticles*sizeof(float), kind);
  cudaMemcpy(pDevice->vy, pHost->vy, pHost->nParticles*sizeof(float), kind);
  cudaMemcpy(pDevice->vz, pHost->vz, pHost->nParticles*sizeof(float), kind);
}

//TODO: Improve this kernel
__global__
void cuMoveParticles(struct ParticleType const *particles, const float dt) {
  const int nParticles = particles->nParticles;
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int i = tid;
  while (i < nParticles){
    // Components of the gravity force on particle i
    float Fx = 0, Fy = 0, Fz = 0; 
      
    // Loop over particles that exert force
    for (int j = 0; j < nParticles; j++) { 
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
}

__global__
void cuUpdateParticles(struct ParticleType const *particles, const float dt){
  // Move particles according to their velocities
  // O(N) work, so using a serial loop
  const int nParticles = particles->nParticles;
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int i = tid;
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
  allocParticleType(particles, nParticles);
  
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
  
  // Allocating data onto GPU
  struct ParticleType *cuParticles;
  //cudaMallocParticleType(cuParticles, nParticles);
  
  //--- CUDA MALLOC BEGIN
  cudaMalloc(&cuParticles, sizeof(ParticleType));
  //size_t *tmp;
  //*tmp = nParticles;
  //cudaMemcpy(&cuParticles->nParticles, &tmp, sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMalloc(&cuParticles->x, nParticles*sizeof(float));
  cudaMalloc(&cuParticles->y, nParticles*sizeof(float));
  cudaMalloc(&cuParticles->z, nParticles*sizeof(float));
  cudaMalloc(&cuParticles->vx, nParticles*sizeof(float));
  cudaMalloc(&cuParticles->vy, nParticles*sizeof(float));
  cudaMalloc(&cuParticles->vz, nParticles*sizeof(float));
  //--- CUDA MALLOC END


  cudaMemcpyParticleType(cuParticles, particles, cudaMemcpyHostToDevice);

  // Perform benchmark
  printf("\nPropagating %d particles using 1 thread...\n\n", 
	 nParticles
	 );
  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration (warm-up)
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);


  // Getting max occupancy
  int blockSize, minGridSize, gridSize;
  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, cuMoveParticles, 0, nParticles); 
  gridSize = (nParticles + blockSize - 1) / blockSize; 

  // Main loop
  for (int step = 1; step <= nSteps; step++) {

    const double tStart = get_time(); // Start timing
    cuMoveParticles<<< gridSize, blockSize >>>(cuParticles, dt);
    cuUpdateParticles<<< gridSize, blockSize >>>(cuParticles, dt);
    const double tEnd = get_time(); // End timing

    const float HztoInts   = ((float)nParticles)*((float)(nParticles-1)) ;
    const float HztoGFLOPs = 20.0*1e-9*((float)(nParticles))*((float)(nParticles-1));

    if (step > skipSteps) { // Collect statistics
      rate  += HztoGFLOPs/(tEnd - tStart); 
      dRate += HztoGFLOPs*HztoGFLOPs/((tEnd - tStart)*(tEnd-tStart)); 
    }

    printf("%5d %10.3e %10.3e %8.1f %s\n", 
	   step, (tEnd-tStart), HztoInts/(tEnd-tStart), HztoGFLOPs/(tEnd-tStart), (step<=skipSteps?"*":""));
    fflush(stdout);

#ifdef DUMP
    cudaMemcpyParticleType(cuParticles, particles, cudaMemcpyDeviceToHost);
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
  freeParticleType(particles);
  cudaFreeParticleType(cuParticles);
}



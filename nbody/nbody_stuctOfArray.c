#include <stdio.h>
#include <math.h>
#include <stdlib.h> // drand48
#include <sys/time.h>
//#include <cuda.h>

//#define DUMP

struct ParticleType { 
  float *x, *y, *z;
  float *vx, *vy, *vz; 
  unsigned int nParticles;
};

float get_time(){
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec;
}

void cuMoveParticles(struct ParticleType const particles, const float dt) {
  const int nParticles = particles.nParticles;
  
  // Loop over particles that experience force
  for (int i = 0; i < nParticles; i++) { 

    // Components of the gravity force on particle i
    float Fx = 0, Fy = 0, Fz = 0; 
      
    // Loop over particles that exert force
    for (int j = 0; j < nParticles; j++) { 
      // No self interaction
      if (i != j) {
          // Avoid singularity and interaction with self
          const float softening = 1e-20;

          // Newton's law of universal gravity
          const float dx = particles.x[j] - particles.x[i];
          const float dy = particles.y[j] - particles.y[i];
          const float dz = particles.z[j] - particles.z[i];
          const float drSquared  = dx*dx + dy*dy + dz*dz + softening;
          const float drPower32  = pow(drSquared, 3.0/2.0);
            
          // Calculate the net force
          Fx += dx / drPower32;  
          Fy += dy / drPower32;  
          Fz += dz / drPower32;
      }

    }

    // Accelerate particles in response to the gravitational force
    particles.vx[i] += dt*Fx; 
    particles.vy[i] += dt*Fy; 
    particles.vz[i] += dt*Fz;
  }

  // Move particles according to their velocities
  // O(N) work, so using a serial loop
  for (int i = 0 ; i < nParticles; i++) { 
    particles.x[i]  += particles.vx[i]*dt;
    particles.y[i]  += particles.vy[i]*dt;
    particles.z[i]  += particles.vz[i]*dt;
  }
}

void dump(int iter, struct ParticleType particles)
{
    const int nParticles = particles.nParticles;

    char filename[64];
    snprintf(filename, 64, "output_%d.txt", iter);

    FILE *f;
    f = fopen(filename, "w+");

    int i;
    for (i = 0; i < nParticles; i++)
    {
        fprintf(f, "%e %e %e %e %e %e\n",
                   particles.x[i], particles.y[i], particles.z[i],
		   particles.vx[i], particles.vy[i], particles.vz[i]);
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

  struct ParticleType particles;
  particles.nParticles = nParticles;
  particles.x =  (float*) malloc(nParticles*sizeof(float));
  particles.y =  (float*) malloc(nParticles*sizeof(float));
  particles.z =  (float*) malloc(nParticles*sizeof(float));
  particles.vx = (float*) malloc(nParticles*sizeof(float));
  particles.vy = (float*) malloc(nParticles*sizeof(float));
  particles.vz = (float*) malloc(nParticles*sizeof(float));

  // Initialize random number generator and particles
  srand48(0x2020);
  
  int i;
  for (i = 0; i < nParticles; i++)
  {
     particles.x[i] =  2.0*drand48() - 1.0;
     particles.y[i] =  2.0*drand48() - 1.0;
     particles.z[i] =  2.0*drand48() - 1.0;
     particles.vx[i] = 2.0*drand48() - 1.0;
     particles.vy[i] = 2.0*drand48() - 1.0;
     particles.vz[i] = 2.0*drand48() - 1.0;
  }
  
  // Perform benchmark
  printf("\nPropagating %d particles using 1 thread...\n\n", 
	 nParticles
	 );
  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration (warm-up)
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  for (int step = 1; step <= nSteps; step++) {

    const double tStart = get_time(); // Start timing
    cuMoveParticles(particles, dt);
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
  free(particles.x);
  free(particles.y);
  free(particles.z);
  free(particles.vx);
  free(particles.vy);
  free(particles.vz);
}



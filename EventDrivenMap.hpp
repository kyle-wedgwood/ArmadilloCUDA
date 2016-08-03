#ifndef EVENTDRIVEMAPHEADERDEF
#define EVENTDRIVEMAPHEADERDEF

#include <cmath>
#include <armadillo>
#include <curand.h>
#include <cassert>
#include "AbstractNonlinearProblem.hpp"
#include "AbstractNonlinearProblemJacobian.hpp"

class EventDrivenMap:
  public AbstractNonlinearProblem
{

  public:

    // Specialised constructor
    EventDrivenMap( const arma::vec* pParameters, unsigned int noReal);

    // Destructor
    ~EventDrivenMap();

    // Right-hand side
    void ComputeF( const arma::vec& u, arma::vec& f);

    // Equation-free stuff
    void SetTimeHorizon( const float T);

    // CUDA stuff
    // Change number of realisations
    void SetNoRealisations( const int noReal);

    // Set variance
    void SetParameterStdDev( const float sigam);

    // Set parameter
    void SetParameters( const unsigned int parId, const float parVal);

    // Set storage for spikes
    void SetStorageCapacity( const unsigned int storageCapacity);

    // Reset seed
    void ResetSeed();

    // Structure to store firing times and indices */
    struct __align__(8) firing
    {
      float time;
      unsigned int index;
    };

    // Structure to do simultaneous reductions */
    struct averaging
    {
      float t;
      float x;
      float tSq;
      float xt;
      unsigned int count;
    };

  private:

    // Hiding default constructor
    EventDrivenMap();

    // Parameter values
    arma::fvec* mpHost_p;

    // Float vector for temporary storage
    arma::fvec* mpU;
    arma::fvec* mpF;

    // For testing porpoises
    float* mpHostData;

    // Integration time
    float mFinalTime;

    // threads & blocks
    unsigned int mNoReal;
    unsigned int mNoThreads;
    unsigned int mNoBlocks;
    unsigned int mNoSpikes;

    // Storagae options
    unsigned int mNoStoredSpikes;

    // CPU variables
    unsigned short *mpHost_lastSpikeInd;

    // GPU variables
    float *mpDev_p;
    float *mpDev_beta;
    float *mpDev_v;
    float *mpDev_s;
    float *mpDev_w;
    float *mpDev_U;
    float *mpDev_Z;
    float *mpDev_firingTime;
    float *mpDev_averages;
    unsigned short *mpDev_firingInd;
    unsigned short *mpDev_spikeInd;
    unsigned short *mpDev_spikeCount;
    unsigned short *mpDev_lastSpikeInd;

    // For parameter heterogeneity
    float mParStdDev;
    curandGenerator_t mGen; // random number generator

    // Functions to do lifting
    void initialSpikeInd( const arma::vec& U);

    void ZtoU( const arma::vec& Z, arma::vec& U);

    void UtoZ( const arma::vec *U, arma::vec *Z);

    void BuildCouplingKernel();
};

__global__ void LiftKernel( float *s, float *v, const float *par, const float *U,
    const unsigned int noReal);

// Functions to find spike time
__device__ float fun( float t, float v, float s, float beta);

__device__ float dfun( float t, float v, float s, float beta);

__device__ float eventTime( float v0, float s0, float beta);

// evolution
__global__ void EvolveKernel( float *v, float *s, const float *beta,
    const float *w, const float finalTime, unsigned short *firingInd,
    float *firingTime, unsigned short *spikeInd, unsigned short *spikeCount,
    unsigned short *lastSpikeInd, const unsigned int noReal,
    const unsigned int noStoredSpikes);

// restriction
__global__ void Restrict( float *U, const float *averages, const unsigned int noReal);

// averaging functions
__global__ void averagesSimultaneousBlocksKernel( float *averages, const unsigned short *dev_firingInd, const float *dev_firingTime, const unsigned short *dev_spikeInd, const unsigned short *dev_spikeCount, unsigned int noReal, const unsigned int noStoredSpikes);
__global__ void realisationReductionKernelBlocks( float *dev_Z, const float* dev_U, const unsigned int noReal);

// helper functions
void circshift( float *w, int shift);
__device__ struct EventDrivenMap::firing warpReduceMin( struct EventDrivenMap::firing val);
__device__ struct EventDrivenMap::firing blockReduceMin( struct EventDrivenMap::firing val);
__device__ float warpReduceSum ( float val);
__device__ float blockReduceSum( float val);
__device__ struct EventDrivenMap::averaging warpReduceSumSimultaneous( struct averaging val);
__device__ struct EventDrivenMap::averaging blockReduceSumSimultaneous( struct EventDrivenMap::averaging val);

void SaveData( int npts, float *x, char *filename);

#endif

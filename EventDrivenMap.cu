#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <armadillo>
#include <curand.h>
#include <assert.h>
#include "EventDrivenMap.hpp"
#include "parameters.hpp"

// Specialised constructor
EventDrivenMap::EventDrivenMap(const arma::vec* pParameters, unsigned int noReal)
{

  // Read in parameters
  mpHost_p = new arma::fvec((*pParameters).n_elem);
  *mpHost_p = arma::conv_to<arma::fvec>::from(*pParameters);

  // Declare memory for temporary storage of U and F
  mpU = new arma::fvec(noSpikes);
  mpF = new arma::fvec(noSpikes);

  // CUDA stuff
  mNoReal    = noReal;
  mNoThreads = noThreads;
  mNoBlocks  = (mNoReal+mNoThreads-1)/mNoThreads;

  // Spike storage
  mNoStoredSpikes = 900; // default storage for spike list

  // Set initial time horizon
  mFinalTime = timeHorizon;

  // allocate memory on CPU
  mpHost_lastSpikeInd = (unsigned short*) malloc( noSpikes*sizeof(short));

  // allocate memory on GPU
  cudaMalloc( &mpDev_p, mpHost_p->n_elem*sizeof(float) );
  cudaMalloc( &mpDev_beta, mNoReal*mNoThreads*sizeof(float) );
  cudaMalloc( &mpDev_v, mNoReal*mNoThreads*sizeof(float) );
  cudaMalloc( &mpDev_s, mNoReal*mNoThreads*sizeof(float) );
  cudaMalloc( &mpDev_w, mNoThreads*sizeof(float) );
  cudaMalloc( &mpDev_U, mNoReal*noSpikes*sizeof(float) );
  cudaMalloc( &mpDev_Z, noSpikes*sizeof(float) );
  cudaMalloc( &mpDev_firingTime, mNoReal*mNoStoredSpikes*sizeof(float) );
  cudaMalloc( &mpDev_firingInd, mNoReal*mNoStoredSpikes*sizeof(unsigned short) );
  cudaMalloc( &mpDev_spikeInd, mNoReal*mNoStoredSpikes*sizeof(unsigned short) );
  cudaMalloc( &mpDev_spikeCount, mNoReal*sizeof(unsigned short) );
  cudaMalloc( &mpDev_lastSpikeInd, noSpikes*sizeof(unsigned short) );
  cudaMalloc( &mpDev_averages, 4*noSpikes*mNoReal*sizeof(float) );

  // Set up coupling kernel
  BuildCouplingKernel();

  // Copy parameters over
  cudaMemcpy(mpDev_p,mpHost_p->begin(),mpHost_p->n_elem*sizeof(float),cudaMemcpyHostToDevice);

  // initialise random number generators
  curandCreateGenerator( &mGen, CURAND_RNG_PSEUDO_DEFAULT);
  ResetSeed();
  mParStdDev = 0.0f;

  // For testing
  mpHostData = (float*) malloc( noReal*mNoStoredSpikes*sizeof(float));
}

void EventDrivenMap::BuildCouplingKernel()
{
  float *w;
  w = (float*) malloc( mNoThreads*sizeof(float));
  for (int i=0;i<mNoThreads;++i)
  {
    float x = -L + (float)(2*L/mNoThreads)*i;
    w[i] = (a1*exp(-b1*abs(x))-a2*exp(-b2*abs(x)))*2*L/mNoThreads;
  }
  circshift(w,mNoThreads/2);
  cudaMemcpy(mpDev_w,w,mNoThreads*sizeof(float),cudaMemcpyHostToDevice);
  FILE *fp = fopen("test.dat","w");
  for (int i=0;i<mNoThreads;++i)
  {
    fprintf(fp,"%f\n",w[i]);
  }
  fclose(fp);
  free(w);
}

EventDrivenMap::~EventDrivenMap()
{
  delete mpU;
  delete mpF;
  delete mpHost_p;

  free(mpHost_lastSpikeInd);

  cudaFree(mpDev_p);
  cudaFree(mpDev_beta);
  cudaFree(mpDev_v);
  cudaFree(mpDev_s);
  cudaFree(mpDev_w);
  cudaFree(mpDev_U);
  cudaFree(mpDev_Z);
  cudaFree(mpDev_firingTime);
  cudaFree(mpDev_firingInd);
  cudaFree(mpDev_spikeInd);
  cudaFree(mpDev_spikeCount);
  cudaFree(mpDev_lastSpikeInd);
  cudaFree(mpDev_averages);

  curandDestroyGenerator(mGen);

  free(mpHostData);
}

void EventDrivenMap::ComputeF(const arma::vec& Z, arma::vec& f)
{

  arma::vec U(noSpikes+1);
  arma::fvec fU(noSpikes+1);

  // Find initial spike indices
  initialSpikeInd( Z);
  cudaMemcpy( mpDev_lastSpikeInd, mpHost_lastSpikeInd, noSpikes*sizeof(unsigned short), cudaMemcpyHostToDevice );

  // Then, put vector in correct form
  ZtoU(Z,U);

  // Then, typecast data as floats
  fU = arma::conv_to<arma::fvec>::from(U);

  // Assuming that weight kernel does not change
  cudaMemcpy(mpDev_U,fU.begin(),(noSpikes+1)*sizeof(float),cudaMemcpyHostToDevice);

  // Introduce parameters heterogeneity
  curandGenerateNormal( mGen, mpDev_beta, mNoReal*mNoThreads, (*mpHost_p)[0], mParStdDev);

  // Lift - working
  LiftKernel<<<mNoReal,mNoThreads>>>(mpDev_s,mpDev_v,mpDev_p,mpDev_U,mNoReal);

  // Copy data to GPU
  //cudaMemcpy( dev_w, w, mNoThreads*sizeof(float), cudaMemcpyHostToDevice );
  // Save spike indices
  cudaMemset( mpDev_firingTime, 0.0f, mNoReal*mNoStoredSpikes*sizeof(float) );
  cudaMemset( mpDev_firingInd, 0, mNoReal*mNoStoredSpikes*sizeof(unsigned short) );
  cudaMemset( mpDev_spikeInd, 0, mNoReal*mNoStoredSpikes*sizeof(unsigned short) );
  cudaMemset( mpDev_spikeCount, 0, mNoReal*sizeof(unsigned short) );

  // Evolve - working
  EvolveKernel<<<mNoReal,mNoThreads>>>(mpDev_v,mpDev_s,mpDev_beta,mpDev_w,mFinalTime,mpDev_firingInd,mpDev_firingTime,
      mpDev_spikeInd,mpDev_spikeCount,mpDev_lastSpikeInd,mNoReal,mNoStoredSpikes);

  // Restrict
  averagesSimultaneousBlocksKernel<<<noSpikes*mNoReal,mNoThreads>>>( mpDev_averages, mpDev_firingInd,
      mpDev_firingTime, mpDev_spikeInd, mpDev_spikeCount, mNoReal, mNoStoredSpikes);

  Restrict<<<(mNoReal+mNoThreads-1)/mNoThreads,mNoThreads>>>( mpDev_U, mpDev_averages, mNoReal);
  realisationReductionKernelBlocks<<<noSpikes,mNoThreads>>>( mpDev_Z, mpDev_U, mNoReal);

  fU.resize(noSpikes);
  U.resize(noSpikes);
  // for (int i=0;i<noSpikes;++i)
  // {
  //   fU[i] = 0.0f;
  // }

  // Copy data back to CPU
  cudaMemcpy( fU.begin(), mpDev_Z, noSpikes*sizeof(float), cudaMemcpyDeviceToHost );

  U = arma::conv_to<arma::vec>::from(fU);

  // Compute F
  f = U-Z;
}

void EventDrivenMap::SetTimeHorizon( const float T)
{
  assert(T>0);
  mFinalTime = T;
}

void EventDrivenMap::SetNoRealisations( const int noReal)
{
  assert(noReal>0);
  mNoReal = noReal;
}

void EventDrivenMap::SetParameterStdDev( const float sigma)
{
  assert(sigma>=0);
  mParStdDev = sigma;
}

void EventDrivenMap::SetParameters( const unsigned int parId, const float parVal)
{
  // Need to check that I don't need to create a temporary variable here
  mpHost_p[parId] = parVal;
  cudaMemcpy(mpDev_p+parId,&parVal,sizeof(float),cudaMemcpyHostToDevice);
}

void EventDrivenMap::SetStorageCapacity( const unsigned int storageCapacity)
{
  assert(storageCapacity>0);
  mNoStoredSpikes = storageCapacity;
  cudaFree( mpDev_firingInd);
  cudaFree( mpDev_firingTime);
  cudaFree( mpDev_spikeInd);
  cudaMalloc( &mpDev_firingInd, mNoReal*mNoStoredSpikes*sizeof(short));
  cudaMalloc( &mpDev_firingTime, mNoReal*mNoStoredSpikes*sizeof(float));
  cudaMalloc( &mpDev_spikeInd, mNoReal*mNoStoredSpikes*sizeof(short));
}

void EventDrivenMap::ResetSeed()
{
  curandSetPseudoRandomGeneratorSeed( mGen, (unsigned long long) clock() );
}

void EventDrivenMap::initialSpikeInd( const arma::vec& U)
{
  unsigned int i,m;
  mpHost_lastSpikeInd[0] = mNoThreads/2;
  for (m=1;m<noSpikes;m++) {
    for (i=mpHost_lastSpikeInd[m-1];i>0;i--) {
      if (-L+(float)(2*i*L/mNoThreads)<-U[0]*U[m]) {
        mpHost_lastSpikeInd[m] = i;
        break;
      }
    }
  }
}

void EventDrivenMap::ZtoU( const arma::vec& Z, arma::vec& U) {
  assert(U.n_elem==Z.n_elem+1);
  U[0] = Z[0];
  U[1] = 0.0;
  for (int i=2;i<=noSpikes;i++) {
    U[i] = Z[i-1];
  }
}

void EventDrivenMap::UtoZ( const arma::vec *U, arma::vec *Z) {
  Z[0] = U[0];
  for (int i=1;i<noSpikes;i++) {
    Z[i] = U[i+1];
  }
}

__global__ void LiftKernel( float *S, float *v, const float *par, const float *U,
    const unsigned int noReal)
{
  int k = threadIdx.x + blockIdx.x*blockDim.x;
  int m;
  if(k<noThreads*noReal){

    //Define x-array
    float x = L - (float)(2*L/noThreads)*threadIdx.x;
    float s = 0.0f;
    float c = U[0];
    float beta = par[0];
    float dummyV, dummyS = 0.0f;

    // Lift Voltage
    # pragma unroll
    for(m=1; m<=noSpikes;m++){
      dummyV = ((x-c*U[m]>0.0f)*(((a1*beta*c)/((beta+c*b1)*(1.0f+c*b1)))* exp(c*U[m]*((1.0f+c*b1)/c))*exp(-b1*c*U[m])
              - ((a2*beta*c)/((beta+c*b2)*(1.0f+c*b2)))* exp(c*U[m]*((1.0f+c*b2)/c))*exp(-b2*c*U[m])+(a1*beta*c/(1.0f-beta))*exp(beta*U[m])*(1.0f/(beta+c*b1)+ 1.0f/(c*b1 - beta))*(exp((x/c)*(1.0f-beta))-exp(((c*U[m])/c)*(1.0f-beta)))-(a1*beta*c/((-beta+c*b1)*(1.0f-c*b1)))*exp(b1*c*U[m])*(exp(x*((1.0f-c*b1)/c))-exp(c*U[m]*((1.0f-c*b1)/c)))
               -(a2*beta*c/(1.0f-beta))*exp(beta*U[m])*(1.0f/(beta+c*b2) + 1.0f/(c*b2 - beta))*(exp((x/c)*(1.0f-beta))-exp((U[m])*(1.0f-beta)))
              +(a2*beta*c/((-beta+c*b2)*(1.0f-c*b2)))*exp(b2*c*U[m])*(exp(x*((1.0f-c*b2)/c))-exp(c*U[m]*((1.0f-c*b2)/c))))
               +
            (x-c*U[m]<=0.0f)*(((a1*beta*c)/((beta +c*b1)*(1.0f+c*b1)))*(exp(x*((1.0f+c*b1)/c)))*exp(-b1*c*U[m])
               - ((a2*beta*c)/((beta +c*b2)*(1.0f+c*b2)))*(exp(x*((1.0f+c*b2)/c)))*exp(-b2*c*U[m])))*exp(-x/c);

      s += dummyV - ((x - c*U[m])>0.0f)*exp(-(x-c*U[m])/c) + ((x-c*U[m])<=0.0f)*0.0f;

      dummyS += ((c*U[m]-x)>0.0f)*(beta*a1*(c/(beta +c*b1))*exp(b1*(x- c*U[m])) - beta*a2*(c/(beta+c*b2))*exp(b2*(x- c*U[m])))
        +((c*U[m]-x)<= 0.0f)*((2.0f*a1/b1)*(beta/(1.0f - ((beta*beta)/(c*c*b1*b1))))*exp(-(beta/c)*(x-c*U[m])) -beta*a1*(c/(-beta +c*b1))*(exp(b1*(c*U[m] - x)))
        - (2.0f*a2/b2)*(beta/(1.0f - ((beta*beta)/(c*c*b2*b2))))*exp(-(beta/c)*(x-c*U[m])) + beta*a2*(c/(-beta +c*b2))*(exp(b2*(c*U[m] - x))));
    }

    v[k] = I + s;
    v[k] *= (v[k]<1.0f);
    S[k] = dummyS;
  }

}

__device__ float fun( float t, float v, float s, float beta)
{
  return v*exp(-t)+I*(1.0f-exp(-t))+s*exp(-t)/(1.0f-beta)*(exp((1.0f-beta)*t)-1.0f)-vth;
}

__device__ float dfun( float t, float v, float s, float beta)
{
  return I*exp(-t)-v*exp(-t)+s*exp(-t)*exp(-t*(beta-1))+(s*exp(-t)*(exp(-t*(beta-1))-1.0f))/(beta-1);
}

__device__ float eventTime( float v0, float s0, float beta)
{
  int decision;
  float f, df, estimatedTime = 0.0f;
  decision = (int) (v0>vth*pow(s0/(vth-I),1.0f/beta)+I*(1.0f-pow(s0/(vth-I),1.0f/beta))-(vth-I)/(beta-1.0f)*(s0/(vth-I)-pow(s0/(vth-I),1.0f/beta)));

  f  = fun( estimatedTime, v0, s0, beta)*decision;
  df = dfun( estimatedTime, v0, s0, beta);

  while (abs(f)>tol) {
    estimatedTime -= f/df;
    f  = fun( estimatedTime, v0, s0, beta);
    df = dfun( estimatedTime, v0, s0, beta);
  }

  return estimatedTime+100.0f*(1.0f-decision);

}

__global__ void EvolveKernel( float *v, float *s, const float *beta,
    const float *w, const float finalTime, unsigned short *firingInd,
    float *firingTime, unsigned short *spikeInd, unsigned short *spikeCount,
    unsigned short *lastSpikeInd, const unsigned int noReal,
    const unsigned noStoredSpikes)
{
  __shared__ unsigned int local_lastSpikeInd[noSpikes];
  unsigned int m = 0;
  float currentTime = 0.0f;
  float local_v, local_s, local_beta;
  unsigned short minIndex;
  struct EventDrivenMap::firing val;

  // load values from global memory
  local_v = v[threadIdx.x+blockIdx.x*blockDim.x];
  local_s = s[threadIdx.x+blockIdx.x*blockDim.x];
  local_beta = beta[threadIdx.x+blockIdx.x*blockDim.x];

  if (threadIdx.x<noSpikes) {
    local_lastSpikeInd[threadIdx.x] = lastSpikeInd[threadIdx.x];
  }
  while ((currentTime<finalTime)&&(m<noStoredSpikes)) {
    // find next firing times
    val.time  = eventTime(local_v,local_s,local_beta);
    val.index = threadIdx.x;

    // Find minimum firing time
    val = blockReduceMin( val);

    // update values to spike time
    local_v *= exp(-val.time);
    local_v +=
      I*(1.0f-exp(-val.time))+local_s*exp(-val.time)/(1.0f-local_beta)*(exp((1.0f-local_beta)*val.time)-1.0f);
    local_v *= (threadIdx.x!=val.index);
    local_s *= exp(-local_beta*val.time);
    local_s += local_beta*w[(threadIdx.x-val.index)*(threadIdx.x>=val.index)+(val.index-threadIdx.x)*(threadIdx.x<val.index)];

    // store values
    minIndex = 0;
    if (threadIdx.x==0) {
      for (int i=1;i<noSpikes;i++) {
        minIndex += ((std::abs((int)(val.index-local_lastSpikeInd[i])))<(std::abs((int)(val.index-local_lastSpikeInd[minIndex]))));
      }
      spikeInd[noStoredSpikes*blockIdx.x+m]   = minIndex;
      firingInd[noStoredSpikes*blockIdx.x+m]  = val.index;
      firingTime[noStoredSpikes*blockIdx.x+m] = currentTime+val.time;
      local_lastSpikeInd[minIndex] = val.index;
    }
    currentTime += val.time;
    m++;
  }

  if (threadIdx.x==0) {
    spikeCount[blockIdx.x] = m;
  }
}

/* Do minimisation without using __shfl_down
//__global__ void EvolveKernel( float *v, float *s, const float *beta,
//    const float *w, const float finalTime, unsigned short *firingInd,
//    float *firingTime, unsigned short *spikeInd, unsigned short *spikeCount,
//    unsigned short *lastSpikeInd, const unsigned int noReal,
//    const unsigned noStoredSpikes)
//{
//  __shared__ float spikeTime[noThreads];
//  __shared__ unsigned short index[noThreads];
//  __shared__ unsigned short local_lastSpikeInd[noSpikes];
//  unsigned int m = 0;
//  float currentTime = 0.0f;
//  unsigned int thread2, halfpoint, nTotalThreads;
//  float local_v, local_s, local_beta;
//  unsigned short minIndex;
//  float temp;
//
//  // load values from global memory
//  local_v = v[threadIdx.x+blockIdx.x*blockDim.x];
//  local_s = s[threadIdx.x+blockIdx.x*blockDim.x];
//  local_beta = *beta;
//
//  if (threadIdx.x<noSpikes) {
//    local_lastSpikeInd[threadIdx.x] = lastSpikeInd[threadIdx.x];
//  }
//  while ((currentTime<finalTime)&&(m<noStoredSpikes)) {
//    // find next firing times
//    spikeTime[threadIdx.x] = eventTime(local_v,local_s,local_beta);
//    index[threadIdx.x] = threadIdx.x;
//    __syncthreads();
//
//    // perform reduction to find minimum spike time
//    nTotalThreads = blockDim.x;
//    while (nTotalThreads>1) {
//      halfpoint = (nTotalThreads>>1);
//      if (threadIdx.x<halfpoint) {
//        thread2 = threadIdx.x + halfpoint;
//
//        temp = spikeTime[thread2];
//        if (temp<spikeTime[threadIdx.x]) {
//          spikeTime[threadIdx.x] = temp;
//          index[threadIdx.x] = index[thread2];
//        }
//      }
//      __syncthreads();
//      nTotalThreads = halfpoint;
//    }
//
//    // update values to spike time
//    local_v *= exp(-spikeTime[0]);
//    local_v +=
//      I*(1.0f-exp(-spikeTime[0]))+local_s*exp(-spikeTime[0])/(1.0f-local_beta)*(exp((1.0f-local_beta)*spikeTime[0])-1.0f);
//    local_v *= (threadIdx.x!=index[0]);
//    local_s *= exp(-local_beta*spikeTime[0]);
//    local_s += local_beta*w[(threadIdx.x-index[0])*(threadIdx.x>=index[0])+(index[0]-threadIdx.x)*(threadIdx.x<index[0])];
//
//    // store values
//    minIndex = 0;
//    if (threadIdx.x==0) {
//      for (int i=1;i<noSpikes;i++) {
//        minIndex += ((abs(index[0]-local_lastSpikeInd[i]))<(abs(index[0]-local_lastSpikeInd[minIndex])));
//      }
//      spikeInd[noStoredSpikes*blockIdx.x+m]   = minIndex;
//      firingInd[noStoredSpikes*blockIdx.x+m]  = index[0];
//      firingTime[noStoredSpikes*blockIdx.x+m] = currentTime+spikeTime[0];
//      local_lastSpikeInd[minIndex] = index[0];
//    }
//    currentTime += spikeTime[0];
//    m++;
//  }
//
//  if (threadIdx.x==0) {
//    spikeCount[blockIdx.x] = m;
//  }
//}

/* Restrict functions */
__global__ void Restrict( float *U, const float *averages,
    const unsigned int noReal)
{
  float xBar, tBar;
  unsigned int i;
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  float baseOffset, speedNum = 0.0f, speedDenom = 0.0f, speed;

  if (index<noReal) {
    # pragma unroll
    for (i=0;i<noSpikes;i++) {
      speedNum   += averages[2*noSpikes*noReal+i*noReal+threadIdx.x];  // Multiply by noCross to undo scalings from earlier
      speedDenom += averages[3*noSpikes*noReal+i*noReal+threadIdx.x];  // Multiply by noCross to undo scalings from earlier
    }
    speed = speedNum/speedDenom;
    U[index] = speed*(2.0f*L/noThreads);

    xBar = averages[0*noSpikes*noReal+0*noReal+threadIdx.x];
    tBar = averages[1*noSpikes*noReal+0*noReal+threadIdx.x];
    baseOffset = xBar-speed*tBar;

    # pragma unroll
    for (i=1;i<noSpikes;i++)
    {
      xBar = averages[0*noSpikes*noReal+i*noReal+threadIdx.x];
      tBar = averages[1*noSpikes*noReal+i*noReal+threadIdx.x];
      U[index+i*noReal] = (baseOffset-xBar)/speed+tBar;
    }
  }
}

__global__ void averagesSimultaneousBlocksKernel( float *averages, const
    unsigned short *firingInd, const float *firingTime, const unsigned short
    *spikeInd, const unsigned short *spikeCount, unsigned int noReal, const unsigned int noStoredSpikes)
{
  unsigned int i, spikeNo = blockIdx.x % noSpikes, realNo = blockIdx.x/noSpikes;
  unsigned short count = spikeCount[realNo];
  unsigned int index;
  unsigned int noLoad = (count+blockDim.x-1)/blockDim.x;
  unsigned int indSpike;
  float x, t;
  struct EventDrivenMap::averaging val = { 0.0, 0.0, 0.0, 0.0, 0};

  for (i=0;i<noLoad;i++) {
    index = threadIdx.x+i*blockDim.x+noStoredSpikes*realNo;
    indSpike = (threadIdx.x+i*blockDim.x < count) ? spikeInd[index]==spikeNo : 0;
    t        = (threadIdx.x+i*blockDim.x < count) ? (float) (firingTime[index]*indSpike) : 0.0f;
    x        = (threadIdx.x+i*blockDim.x < count) ? (float) (firingInd[index]*indSpike) : 0.0f;
    val.t     += t;
    val.x     += x;
    val.count += indSpike;
    val.tSq   += t*t;
    val.xt    += t*x;
  }

  val = blockReduceSumSimultaneous( val);

  if (threadIdx.x==0) {
    averages[0*noSpikes*noReal+spikeNo*noReal+realNo] = val.x/val.count;
    averages[1*noSpikes*noReal+spikeNo*noReal+realNo] = val.t/val.count;
    averages[2*noSpikes*noReal+spikeNo*noReal+realNo] = val.xt-val.x*val.t/val.count; // SpeedNum
    averages[3*noSpikes*noReal+spikeNo*noReal+realNo] = val.tSq-val.t*val.t/val.count; // SpeedDenom
  }
}

//__global__ void averagesSimultaneousBlocksKernel( float *averages, const
//    unsigned short *firingInd, const float *firingTime, const unsigned short
//    *spikeInd, const unsigned short *spikeCount, unsigned int noReal, const unsigned int noStoredSpikes)
//{
//  unsigned int i, spikeNo = blockIdx.x % noSpikes, realNo = blockIdx.x/noSpikes;
//  unsigned short count = spikeCount[realNo];
//  unsigned int index;
//  unsigned int noLoad = (count+blockDim.x-1)/blockDim.x;
//  unsigned int indSpike;
//  float x, t;
//  struct EventDrivenMap::averaging val = { 0.0, 0.0, 0.0, 0.0, 0};
//
//  for (i=0;i<noLoad;i++) {
//    index = threadIdx.x+i*blockDim.x+noStoredSpikes*realNo;
//    indSpike = (threadIdx.x+i*blockDim.x < count) ? spikeInd[index]==spikeNo : 0;
//    t        = (threadIdx.x+i*blockDim.x < count) ? (float) (firingTime[index]*indSpike) : 0.0f;
//    x        = (threadIdx.x+i*blockDim.x < count) ? (float) (firingInd[index]*indSpike) : 0.0f;
//    val.t     += t;
//    val.x     += x;
//    val.count += indSpike;
//    val.tSq   += t*t;
//    val.xt    += t*x;
//  }
//
//  val = blockReduceSumSimultaneous( val);
//
//  if (threadIdx.x==0) {
//    averages[0*noSpikes*noReal+spikeNo*noReal+realNo] = val.x;
//    averages[1*noSpikes*noReal+spikeNo*noReal+realNo] = val.t;
//    averages[2*noSpikes*noReal+spikeNo*noReal+realNo] = val.tSq;
//    averages[3*noSpikes*noReal+spikeNo*noReal+realNo] = val.xt;
//    averages[4*noSpikes*noReal+spikeNo*noReal+realNo] = val.count;
//  }
//}

__global__ void realisationReductionKernelBlocks( float *Z, const float *U, const unsigned int noReal)
{
  unsigned int i, spikeNo = blockIdx.x;
  unsigned int index;
  unsigned int noLoad = (noReal+blockDim.x-1)/blockDim.x;
  float average = 0.0f;

  for (i=0;i<noLoad;i++)
  {
    index = threadIdx.x+i*blockDim.x;
    average += (index < noReal) ? U[index+spikeNo*noReal] : 0.0f;
  }
  average = blockReduceSum( average);
  if (threadIdx.x==0)
  {
    Z[spikeNo] = average/noReal;
  }
}

void circshift( float *w, int shift) {
  int i;
  float dummy[noThreads];
  # pragma unroll
  for (i=0;i<noThreads-shift;i++) {
    dummy[i] = w[shift+i];
  }
  # pragma unroll
  for (i=0;i<shift;i++) {
    dummy[noThreads-shift+i] = w[i];
  }
  # pragma unroll
  for (i=0;i<noThreads;i++) {
    w[i] = dummy[i];
  }
}

__device__ struct EventDrivenMap::firing warpReduceMin( struct EventDrivenMap::firing val)
{
  float dummyTime;
  unsigned int dummyIndex;
  for (int offset = warpSize/2; offset>0; offset/=2) {
    dummyTime  = __shfl_down( val.time, offset);
    dummyIndex = __shfl_down( val.index, offset);
    val.time   = (val.time < dummyTime) ? val.time : dummyTime;
    val.index  = (val.time < dummyTime) ? val.index : dummyIndex;
  }
  return val;
}

__device__ struct EventDrivenMap::firing blockReduceMin( struct EventDrivenMap::firing val)
{
  __shared__ struct EventDrivenMap::firing shared[32];
  int lane = threadIdx.x % warpSize;
  int wid  = threadIdx.x / warpSize;

  val = warpReduceMin( val);

  if (lane==0) {
    shared[wid] = val;
  }
  __syncthreads();

  val.time  = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].time  : 0.0f;
  val.index = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].index : 0;

  if (wid==0) {
    val = warpReduceMin( val);
  }

  if (threadIdx.x==0) {
    shared[0] = val;
  }
  __syncthreads();
  val = shared[0];

  return val;
}

__device__ float warpReduceSum( float val) {
  for (int offset = warpSize/2; offset>0; offset/=2) {
    val += __shfl_down( val, offset);
  }
  return val;
}

__device__ float blockReduceSum( float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int wid  = threadIdx.x / warpSize;

  val = warpReduceSum( val);

  if (lane==0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x<blockDim.x/warpSize) ? shared[lane] : 0.0f;

  if (wid==0) {
    val = warpReduceSum( val);
  }

  return val;
}

/* These functions are to help with doing reductions */
__device__ struct EventDrivenMap::averaging warpReduceSumSimultaneous( struct
    EventDrivenMap::averaging val)
{
  for (int offset = warpSize/2; offset>0; offset/=2) {
    val.t   += __shfl_down( val.t, offset);
    val.x   += __shfl_down( val.x, offset);
    val.tSq += __shfl_down( val.tSq, offset);
    val.xt  += __shfl_down( val.xt, offset);
    val.count += __shfl_down( val.count, offset);
  }
  return val;
}

__device__ struct EventDrivenMap::averaging blockReduceSumSimultaneous( struct
    EventDrivenMap::averaging val)
{
  __shared__ struct EventDrivenMap::averaging shared[32];
  int lane = threadIdx.x % warpSize;
  int wid  = threadIdx.x / warpSize;

  val = warpReduceSumSimultaneous( val);

  if (lane==0) {
    shared[wid] = val;
  }
  __syncthreads();

  val.t     = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].t : 0.0f;
  val.x     = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].x : 0.0f;
  val.tSq   = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].tSq : 0.0f;
  val.xt    = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].xt : 0.0f;
  val.count = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].count : 0;

  if (wid==0) {
    val = warpReduceSumSimultaneous( val);
  }

  return val;
}

void SaveData( int npts, float *x, char *filename) {
  FILE *fp = fopen(filename,"w");
  for (int i=0;i<npts;i++) {
    fprintf(fp,"%f\n",x[i]);
  }
  fclose(fp);
}

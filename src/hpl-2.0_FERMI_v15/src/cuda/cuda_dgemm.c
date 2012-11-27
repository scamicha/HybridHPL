/*
# Copyright (c) 2008-2011,  NVIDIA CORPORATION
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are 
# met: Redistributions of source code must retain the above copyright 
# notice, this list of conditions and the following disclaimer. 
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution. Neither the name 
# of NVIDIA nor the names of its contributors may be used to endorse or 
# promote products derived from this software without specific prior written
# permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND 
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


/*
  Initial implementation of host library to intercept DGEMM and DTRSM
  and distribute the work between GPU and CPU cores.
  This implementation assumes that each MPI process is going to use a
  single GPU.


  @2008-2009 Massimiliano Fatica	
  @2010-2011 Everett Phillips and Massimiliano Fatica	
  
  History:
  04/09/2009	Fixed bug in the sorting of the host names
  12/28/2008	Initial external release

*/
#ifdef MPI
#include <mpi.h>
#endif

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <dlfcn.h>
#include <ctype.h>
#include <math.h>

#include "cuda_runtime.h"
#include "cublas.h"
#include "fermi_dgemm.h"

//#define DTRSM_ACCUM_TIMER
//#define DGEMM_ACCUM_TIMER

#define NSTREAMS (4)

//automatically adjust DGEMM split (big DGEMMs - update)
#define AUTOSPLIT

//automatically adjust DGEMM split (small DGEMMs - panel factorization)
#define AUTOSPLIT2

//interleave streams 
//enabled : copy1 copy2 copy3 ... kernel1 kernel2 kernel3 ...
//disabled: copy1 kernel1 copy2 kernel2 copy3 kernel3 ...
//#define INTERLEAVE

#define NK (64)
#define NN (64)
#define NM (128)
//#define cudaThreadSynchronize() fermiSyncStream(0)

#define imin(a,b) (((a)<(b))?(a):(b))
#define imax(a,b) (((a)<(b))?(b):(a))

#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>

/* 
  Name of the symbols in the library. Depending on the mangling on the host compiler 
*/
#define CUBLAS_DGEMM_MF dgemm_
#define CUBLAS_DTRSM_MF dtrsm_

/*define scratch arrays on GPU */
  static double *dev_scratch[NSTREAMS+1];
  static size_t scratch_size;
  static int stream[NSTREAMS];
/*

   Timer function: returns wallclock time in seconds
*/

double wallclock(void)
{
  struct timeval tv;
  struct timezone tz;
  double t;

  gettimeofday(&tv, &tz);

  t = (double)tv.tv_sec;
  t += ((double)tv.tv_usec)/1000000.0;

  return t;
}


/* 
   The first time DGEMM or DTRSM are called, the library needs to map a GPU to the MPI process.
   This variable checks if this step has already been performed
*/
    static int first_time=1;
    static int myrank=0;
    static int gpu_per_node=0;
    static int SM_COUNT=1;
    static int mydev;

int stringCmp( const void *a, const void *b)
{
     return strcmp(a,b);

}

#ifdef MPI
       static char     host_name[MPI_MAX_PROCESSOR_NAME];
#else
       static char     host_name[20];
#endif

/* 
   This function finds out how many MPI processes are running on the same node 
   and assigns a local rank that can be used to map a process to a device.
   This function needs to be called by all the MPI processes.
*/


void  assignDeviceToProcess()
{
#ifdef MPI
       char (*host_names)[MPI_MAX_PROCESSOR_NAME];
       MPI_Comm nodeComm;
#endif

       int i, n, namelen, color, rank, nprocs;
       size_t bytes;
       int dev, err1;
       struct cudaDeviceProp deviceProp;

       /* Check if the device has been alreasy assigned */
       if(first_time)
        {
         first_time=0;

#ifdef MPI
       MPI_Comm_rank(MPI_COMM_WORLD, &rank);
       MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
       MPI_Get_processor_name(host_name,&namelen);

       bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
       host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);

       strcpy(host_names[rank], host_name);

       for (n=0; n<nprocs; n++)
       {
        MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD); 
       }


       qsort(host_names, nprocs,  sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

       color = 0;

       for (n=0; n<nprocs; n++)
       {
         if(n>0&&strcmp(host_names[n-1], host_names[n])) color++;
         if(strcmp(host_name, host_names[n]) == 0) break;
       }

       MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);

       MPI_Comm_rank(nodeComm, &myrank);
       MPI_Comm_size(nodeComm, &gpu_per_node);

#else
      myrank = 0;
#endif
        /* Find out how many DP capable GPUs are in the system and their device number */
       int deviceCount,slot=0;
       int *devloc;
       cudaGetDeviceCount(&deviceCount);
       devloc=(int *)malloc(deviceCount*sizeof(int));
       devloc[0]=999;
       for (dev = 0; dev < deviceCount; ++dev)
        {
        cudaGetDeviceProperties(&deviceProp, dev);
#ifdef USE_FERMI_DGEMM
        if(deviceProp.major==2)
#endif
          {
           devloc[slot]=dev;
           slot++;
          };
        }

        int gpu_count_err=0, global_gpu_count_err=0;
        if(slot<gpu_per_node)
        {
          if(myrank==0) printf ("!!! ERROR: Not enough GPUs on node %s, %d GPUs found, %d GPUs required !!!\n",host_name,slot,gpu_per_node);
          gpu_count_err = 1;
        }
#ifdef MPI
        MPI_Allreduce( &gpu_count_err, &global_gpu_count_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
#else
        global_gpu_count_err = gpu_count_err;
#endif
        if(global_gpu_count_err>0)
        {
#ifdef MPI
          MPI_Finalize();
#endif
          exit(1);
          return;
        }

#ifdef VERBOSE_PRINT         
       printf ("rank %d Assigning device %d to process on node %s \n", rank, devloc[myrank],  host_name );
#endif
       /* Assign device to MPI process, initialize BLAS and probe device properties */
       cudaSetDevice(devloc[myrank]);
       mydev = devloc[myrank];
       cublasInit();
       cudaGetDevice(&dev);
       cudaGetDeviceProperties(&deviceProp, dev);
       SM_COUNT = deviceProp.multiProcessorCount;
       //printf("SM_COUNT = %d\n",SM_COUNT);
       size_t free_bytes, total_bytes;
       size_t scratch_size_A = 1024*1024*1024;
       cudaMemGetInfo(&free_bytes, &total_bytes);
       //printf("free: %lld MB, Total: %lld MB\n",free_bytes>>20,total_bytes>>20);
       scratch_size = (size_t)((imin( ((free_bytes>>20)-352-(scratch_size_A>>20)), 1024)/NSTREAMS)*1024.0*1024.0);
       while(scratch_size < 64*1024*1024) 
       {
         scratch_size_A -= 64*1024*1024;
         scratch_size = (size_t)((imin( ((free_bytes>>20)-352-(scratch_size_A>>20)), 1024)/NSTREAMS)*1024.0*1024.0);
       }
       //scratch_size = (size_t)((1024.0/NSTREAMS)*1024.0*1024.0);
#ifdef VERBOSE_PRINT
       printf("rank %d Allocating main buffer: %lld MB\n",rank,(size_t)((double)scratch_size*(double)NSTREAMS+(double)scratch_size_A)>>20);
#endif

       err1=cudaMalloc((void**)&dev_scratch[0], (size_t)((double)scratch_size*(double)NSTREAMS+(double)scratch_size_A) );

        int gpu_malloc_err=0, global_gpu_malloc_err=0;
        if(err1)
        {
          printf ("Error allocating scratch space %lld MB on node %s rank %d device %d\n",(size_t)((double)scratch_size*(double)NSTREAMS+(double)scratch_size_A)>>20, host_name, rank, devloc[myrank]);
          gpu_malloc_err = 1;
        }
#ifdef MPI
        MPI_Allreduce( &gpu_malloc_err, &global_gpu_malloc_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
#else
        global_gpu_malloc_err = gpu_malloc_err;
#endif
        if(global_gpu_malloc_err>0)
        {
#ifdef MPI
          MPI_Finalize();
#endif
          exit(1);
          return;
        }

       dev_scratch[1] = dev_scratch[0] + scratch_size_A/8;

       for(i=0; i<NSTREAMS; i++)
       {
         dev_scratch[i+1] = dev_scratch[1] + i*scratch_size/8;
         stream[i] = i+1;
       }

       err1 = fermiCreateStreams(NSTREAMS);

        if(err1)
        {
          printf ("Error in fermiCreateStreams on node %s rank %d device %d\n",host_name, rank, devloc[myrank]);
          gpu_malloc_err = 1;
        }
#ifdef MPI
        MPI_Allreduce( &gpu_malloc_err, &global_gpu_malloc_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
#else
        global_gpu_malloc_err = gpu_malloc_err;
#endif
        if(global_gpu_malloc_err>0)
        {
#ifdef MPI
          MPI_Finalize();
#endif
          exit(1);
          return;
        }

//        if(rank==0) printf("GPU initializations successfull\n");

       free(devloc);
       myrank = rank;

       }else{
          cudaSetDevice(mydev);
       }
}

void CUBLAS_DTRSM_MF (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   const double *alpha, const double *A, const int *lda,
                   double *B, const int *ldb)
{
    double *devPtrA, *devPtrB;
    int k;

    char *error;
    static void *handle2;
    static void (*mkl_dtrsm)();
    static int  first_time_dtrsm=1;

    double  time_dtrsm, time_dtrsm_mkl, time_io, time_tmp, time_cuda;
#ifdef DTRSM_ACCUM_TIMER
    static double time_dtrsm_total=0.0;
//    static double time_dtrsm_total_gpu=0.0;
//    static double time_dtrsm_total_cpu=0.0;
    double tmp_time;
#endif
    static cudaEvent_t start, copy1, copy2, stop;
    float eventTimer;
    static int last_m=0;
    size_t n_gpu, n_cpu, b_offset;
    static float split=0.7, used_split;
    float ratio;
    cublasStatus status;
    int err1, err2, err3, i;

    if (first_time)
    {
      assignDeviceToProcess();
      first_time=0;
    }

    if (first_time_dtrsm)
    {
          cudaSetDevice(mydev);
#ifdef GOTO
      handle2 = dlopen ("libgoto2.so", RTLD_LAZY);
#endif
#ifdef ACML
      handle2 = dlopen ("libacml_mp.so", RTLD_LAZY);
#else
      handle2 = dlopen ("libmkl_intel_lp64.so", RTLD_LAZY);
#endif

      if (!handle2) 
      {
        fprintf (stderr, "%s\n", dlerror());
        exit(1);
      }
      dlerror();    /* Clear any existing error */

#ifdef GOTO
      mkl_dtrsm = (void(*)())dlsym(handle2, "dtrsm_");
#endif
#ifdef ACML
      mkl_dtrsm = (void(*)())dlsym(handle2, "dtrsm_");
#else
      mkl_dtrsm = (void(*)())dlsym(handle2, "dtrsm");
#endif

      if ((error = dlerror()) != NULL)  {
          fprintf (stderr, "%s\n", error);
          exit(1);
       }

       cudaEventCreate(&start);
       cudaEventCreate(&stop);
       cudaEventCreate(&copy1);

      /*Check if environment variable CUDA_DTRSM_SPLIT is set, otherwise use standard split */
      const char *name= "CUDA_DTRSM_SPLIT";
      char *value;
      value = getenv(name);
      if (value != NULL ){
          split = atof(value);
#ifdef VERBOSE_PRINT
          printf ("%d DTRSM split from environment variable %f \n",myrank,split);
#endif
          }
     used_split=split;
     first_time_dtrsm=0;

    }

//     mkl_dtrsm (side, uplo, transa, diag, m, n, alpha, A,
//                  lda, B, ldb);
//     return;

if(transa[0] == 'N' || transa[0] == 'n')
{
    //if ( (*m) < 960 || (*n) <960 || (*n) >24480 ) {
    if ( (*m) < 512 || (*n) < 2*(*m) || split < 0.01 ) {
#ifdef DTRSM_ACCUM_TIMER
    tmp_time = -wallclock();
#endif
    mkl_dtrsm (side, uplo, transa, diag, m, n, alpha, A,
                 lda, B, ldb);
#ifdef DTRSM_ACCUM_TIMER
    tmp_time += wallclock();
    time_dtrsm_total += tmp_time;
//    time_dtrsm_total_cpu += tmp_time;
#endif

    return;
    }
          cudaSetDevice(mydev);
    if(*n>last_m+4096 && *n!=*m)
    {
      used_split=split;
#ifdef VERBOSE_PRINT
      printf("%d resetting DTRSM SPLIT = %f \n",myrank,used_split);
#endif
#ifdef DTRSM_ACCUM_TIMER
      time_dtrsm_total = 0.0;
#endif
    }
    last_m = *n;
/* No lda, just the data */

    n_gpu = imin(ceil(*n/NN*used_split)*NN ,(*n)-(*n)%NN);
    n_cpu = *n - n_gpu;
    b_offset = n_gpu * (*ldb);

    time_dtrsm=wallclock();
    cudaEventRecord(start, 0);

    devPtrA=dev_scratch[0];
    status = fermiSetMatrix (*m, *m, sizeof(A[0]), A, *lda, devPtrA, *m, 0);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) 
            printf ( "!!!! device access error on node %s rank %d (write A at %s:%d) %d %s\n",
                host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
         }
    cudaEventRecord(copy1, 0);

    size_t nmax=SM_COUNT*256;
    int iters=n_gpu/nmax;
    while(n_gpu>iters*nmax) iters++;
    size_t b_offset_gpu=0;

    for(i=0; i<iters; i++)
    {
      nmax=imin(nmax,n_gpu-i*nmax);
      //printf("iter = %d, nmax = %d \n",i,nmax);
      devPtrB=dev_scratch[i%4+1];
      status = fermiSetMatrix (*m, nmax, sizeof(B[0]), B+b_offset_gpu, *ldb, devPtrB, *m, stream[i%4]);
      if (status != CUBLAS_STATUS_SUCCESS) {
          cudaError_t err = cudaGetLastError();
          if (cudaSuccess != err) 
              printf ( "!!!! device access error on node %s rank %d (write B at %s:%d) %d %s\n", 
                  host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
      }

      dtrsm_gpu (side[0], uplo[0], transa[0], diag[0], *m, nmax, *alpha, devPtrA, *m, devPtrB, *m, stream[i%4]);

      status = fermiGetMatrix (*m, nmax, sizeof(B[0]), devPtrB, *m, B+b_offset_gpu, *ldb, stream[i%4]);
      if (status != CUBLAS_STATUS_SUCCESS) {
          cudaError_t err = cudaGetLastError();
          if (cudaSuccess != err) 
              printf ("!!!! device access error on node %s rank %d (read C at %s:%d) %d %s\n", 
                  host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
      }

      b_offset_gpu += (*ldb)*nmax;

    }
    cudaEventRecord(stop, 0);

//    time_dtrsm_mkl = wallclock();
    mkl_dtrsm (side, uplo, transa, diag, m, &n_cpu, alpha, A, lda, B+(size_t)(*ldb)*n_gpu, ldb);
}
else
{
    //if ( (*m) < 960 || (*n) <960 || (*n) >24480 ) {
    if ( (*n) < 512 || (*m) < 2*(*n) || split < 0.01 ) {
#ifdef DTRSM_ACCUM_TIMER
    tmp_time = -wallclock();
#endif
    mkl_dtrsm (side, uplo, transa, diag, m, n, alpha, A,
                 lda, B, ldb);
#ifdef DTRSM_ACCUM_TIMER
    tmp_time += wallclock();
    time_dtrsm_total += tmp_time;
//    time_dtrsm_total_cpu += tmp_time;
#endif

    return;
    }
          cudaSetDevice(mydev);
    if(*m>last_m+4096 && *m!=*n)
    {
      used_split=split;
#ifdef VERBOSE_PRINT
      printf("%d resetting DTRSM SPLIT = %f \n",myrank,used_split);
#endif
    }
    last_m = *m;
/* No lda, just the data */

    n_gpu = imin(ceil(*m/NN*used_split)*NN ,(*m)-(*m)%NN);
    n_cpu = *m - n_gpu;
    b_offset = n_gpu;

    time_dtrsm=wallclock();
    cudaEventRecord(start, 0);

    devPtrA=dev_scratch[0];
    status = fermiSetMatrix (*n, *n, sizeof(A[0]), A, *lda, devPtrA, *n, 0);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) 
            printf ( "!!!! device access error on node %s rank %d (write A at %s:%d) %d %s\n",
                host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
         }
    cudaEventRecord(copy1, 0);

    size_t nmax=SM_COUNT*256;
    int iters=n_gpu/nmax;
    while(n_gpu>iters*nmax) iters++;
    size_t b_offset_gpu=0;

    for(i=0; i<iters; i++)
    {
      nmax=imin(nmax,n_gpu-i*nmax);
      //printf("iter = %d, nmax = %d \n",i,nmax);
      devPtrB=dev_scratch[i%4+1];
      status = fermiSetMatrix (nmax, *n, sizeof(B[0]), B+b_offset_gpu, *ldb, devPtrB, nmax, stream[i%4]);
      if (status != CUBLAS_STATUS_SUCCESS) {
          cudaError_t err = cudaGetLastError();
          if (cudaSuccess != err) 
              printf ( "!!!! device access error on node %s rank %d (write B at %s:%d) %d %s\n", 
                  host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
      }

      dtrsm_gpu (side[0], uplo[0], transa[0], diag[0], nmax, *n, *alpha, devPtrA, *n, devPtrB, nmax, stream[i%4]);

      status = fermiGetMatrix (nmax, *n, sizeof(B[0]), devPtrB, nmax, B+b_offset_gpu, *ldb, stream[i%4]);
      if (status != CUBLAS_STATUS_SUCCESS) {
          cudaError_t err = cudaGetLastError();
          if (cudaSuccess != err) 
              printf ("!!!! device access error on node %s rank %d (read C at %s:%d) %d %s\n", 
                  host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
      }

      b_offset_gpu += nmax;

    }
    cudaEventRecord(stop, 0);

    mkl_dtrsm (side, uplo, transa, diag, &n_cpu, n, alpha, A, lda, B+n_gpu, ldb);
}

    time_dtrsm_mkl=wallclock()-time_dtrsm;

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&eventTimer, start, stop); 
    time_cuda=(double)eventTimer/1000.0;
    cudaEventElapsedTime(&eventTimer, start, copy1);
    time_io = (double)eventTimer/1000.0;
    ratio = time_cuda/time_dtrsm_mkl;
    time_dtrsm=wallclock()-time_dtrsm;
    used_split = 0.5*(used_split + 1.0/(1.0 + (ratio)*(1.0-used_split)/used_split ));
#ifdef VERBOSE_PRINT
    printf (" Proc %d DTRSM CUDA (%d) in %f alpha=%f ", myrank,  n_gpu,  time_cuda, *alpha );
    printf("Ratio=%f , Split=%f\n",ratio, used_split );
    printf (" Proc %d DTRSM (M,N, N_GPU, lda, ldb) %d %d  %d %d %d %c %c %c %c in %f (MKL %f I/O %f)\n", myrank, *m, *n, n_gpu, *lda, *ldb,  side[0], uplo[0], transa[0], diag[0], time_dtrsm, time_dtrsm_mkl, time_io ); 
#endif

#ifdef DTRSM_ACCUM_TIMER
    time_dtrsm_total += time_dtrsm;
//    time_dtrsm_total_gpu += time_cuda;
//    time_dtrsm_total_cpu += time_dtrsm_mkl;
//    printf(" proc %d, DTRSM ACCUM TIMER: CPU: %f GPU: %f TOTAL: %f \n",myrank, time_dtrsm_total_cpu, time_dtrsm_total_gpu, time_dtrsm_total);
     printf(" proc %d, DTRSM TIMER: %f TOTAL: %f \n",myrank, time_dtrsm, time_dtrsm_total);
#endif
}

void CUBLAS_DGEMM_MF (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const double *alpha,
                   const double *A, const int *lda, const double *B,
                   const int *ldb, const double *beta, double *C, const int *ldc)
{
    size_t m_gpu, n_gpu, k_gpu, m_cpu2, n_cpu2, opt_M;
    size_t m_cpu, n_cpu, k_cpu;
    size_t nmax, mmax, nmax1, mmax1, nlocal, iter, i, j, outer_iter, last_iter, iter_m;
    size_t a_offset, b_offset, c_offset;
    size_t a_offset_gpu, b_offset_gpu, c_offset_gpu;
    size_t a_offset_gpu1, b_offset_gpu1, c_offset_gpu1;
    size_t a_offset_gpu2, b_offset_gpu2, c_offset_gpu2;
    size_t a_offset_gpu3, b_offset_gpu3, c_offset_gpu3;
    double *devPtrA, *devPtrB, *devPtrC;
    double CPU_GFLOPS,GPU_GFLOPS,GFLOPS;
    static float split=0.7;
    static float * splits;
    static int last_m=0;
    cublasStatus status;
    float used_split=0.7, opt_split, ratio;
    char *error;
    static void *handle;
    static void (*dgemm_mkl)();

    double  time_dgemm_io, time_dgemm_io2, time_dgemm, time_tmp, time_cuda;
    double  time_dgemm_mkl,time_cudamalloc,time_cudamalloc_tot;
#ifdef DGEMM_ACCUM_TIMER
    static double time_dgemm_total=0.0;
//    static double time_dgemm_total_gpu=0.0;
//    static double time_dgemm_total_cpu=0.0;
    double tmp_time;
#endif
    static cudaEvent_t start, stop;
    float eventTimer;

    int err1, err2,err3;
    static int first_time_dgemm=1;
    static int first_split;
    static int counter=0;

    if( (*m==0) || (*n==0) || (*k==0) ) return;

    if (first_time)
    {
      assignDeviceToProcess();
      first_time=0;
    }

    if (first_time_dgemm)
    {
          cudaSetDevice(mydev);
#ifdef GOTO
      handle = dlopen ("libgoto2.so", RTLD_LAZY);
#endif
#ifdef ACML
      handle = dlopen ("libacml_mp.so", RTLD_LAZY);
#else
      handle = dlopen ("libmkl_intel_lp64.so", RTLD_LAZY);
#endif

      if (!handle) {
        fprintf (stderr, "%s\n", dlerror());
        exit(1);
      }
      dlerror();    /* Clear any existing error */

#ifdef GOTO
      dgemm_mkl = (void(*)())dlsym(handle, "dgemm_");
#endif
#ifdef ACML
      dgemm_mkl = (void(*)())dlsym(handle, "dgemm_");
#else
      dgemm_mkl = (void(*)())dlsym(handle, "dgemm");
#endif


       if ((error = dlerror()) != NULL)  {
        fprintf (stderr, "%s\n", error);
        exit(1);
      }

      cudaEventCreate(&start);
      cudaEventCreate(&stop);

     /*Check if environment variable CUDA_DGEMM_SPLIT is set, otherwise use standard split */
      const char *name= "CUDA_DGEMM_SPLIT"; 
      char *value;
      value = getenv(name);
      if (value != NULL ){
          split = atof(value);
#ifdef VERBOSE_PRINT
          printf ("%d DGEMM split from environment variable %f \n",myrank,split);
#endif
          }
      splits = 0;
      splits = (float*)malloc(100*gpu_per_node*sizeof(float));
      if(!splits) printf("SPLITS ALLOCATION FAILED on node %s rank %d \n",host_name,myrank);
      for(i=0; i<100*gpu_per_node; i++) splits[i] = split;
      first_split = 1;
      first_time_dgemm=0;
    }

    //if ( (*n) < 256 || (*m) <256 || (*k) <256 ) {   
    //if ( (*n) < NN || (*m) < NM || (*k) < NK || split < 0.01f ||  (*n==*k)&&((*n%NN)||(*k%NK)) ) { 
    if ( (*n) < NN || (*m) < NM || (*k) < 128 || split < 0.01f || (*k%NK) ){ //||  (*n==*k)&&(transb[0]=='N'||transb[0]=='n')&&((*n%NN)||(*k%NK)) || (*k%16) ) { 
#ifdef DGEMM_ACCUM_TIMER
    tmp_time = -wallclock();
#endif
    dgemm_mkl(transa, transb, m, n, k, alpha, A, lda, 
              B, ldb, beta,C, ldc);
#ifdef DGEMM_ACCUM_TIMER
    tmp_time += wallclock();
    time_dgemm_total += tmp_time;
//    time_dgemm_total_cpu += tmp_time;
#endif
     return;
    }
          cudaSetDevice(mydev);
    if(*m > last_m+4096)
    {
#ifdef VERBOSE_PRINT
      printf("RESETTING SPLITS TO %f \n",split);
#endif
      for(i=0; i<100*gpu_per_node; i++) splits[i] = split;
      first_split = 1;
#ifdef DGEMM_ACCUM_TIMER
      time_dgemm_total = 0.0;
#endif
    }
    last_m = *m;
    // New split
    k_gpu = k_cpu = *k ;

    if(*n == *k)
      {
       time_dgemm=wallclock();
       cudaEventRecord(start,0);

       n_gpu = (*n/NN)*NN;
       n_cpu = *n;
       n_cpu2 = n_gpu-n_gpu;

#ifdef AUTOSPLIT2
       used_split = splits[*k/NK];
#else
       used_split = split;
#endif

       if(transb[0]=='N'||transb[0]=='n')  opt_M = NM;
       else opt_M = 64;
       m_gpu = imin(ceil(*m /opt_M*(used_split))*opt_M,(*m)-(*m)%opt_M);

       m_cpu = (*m)-m_gpu ;
       a_offset = m_gpu;
       b_offset = 0;
       c_offset = m_gpu;
       a_offset_gpu = 0;
       b_offset_gpu = 0;
       c_offset_gpu = 0;
       nmax = *n;

       mmax = ((16*1024*1024)/8/(*k)/opt_M)*opt_M;
       mmax = imin( m_gpu, mmax);
       iter_m = m_gpu/mmax;
       while(m_gpu>iter_m*mmax) iter_m++;
       mmax1 = m_gpu-(iter_m-1)*mmax;


    devPtrB= dev_scratch[1];
    status = fermiSetMatrix (k_gpu, nmax, sizeof(B[0]), B+b_offset_gpu, *ldb, devPtrB, k_gpu, (int)0);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) 
          printf ("!!!! device access error *n=*k on node %s rank %d (write B at %s:%d) %d %s\n", 
              host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
    }
    devPtrC= devPtrB + k_gpu*nmax;
    devPtrA= dev_scratch[0];

for(i=0; i<iter_m; i++)
{
   mmax = imin(mmax,m_gpu-mmax*i);

    status = fermiSetMatrix (mmax, nmax, sizeof(C[0]), C+c_offset_gpu, *ldc, devPtrC, m_gpu, (int)stream[i%NSTREAMS]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) 
          printf ("!!!! device access error on node %s rank %d (write C at %s:%d) %d %s\n", 
              host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
    }

    status = fermiSetMatrix (mmax, k_gpu, sizeof(A[0]), A+a_offset_gpu, *lda, devPtrA, m_gpu, (int)stream[i%NSTREAMS]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) 
          printf ("!!!! device access error on node %s rank %d (write A at %s:%d) %d %s\n",
              host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
    }

      status = fermiDgemm_stream(transa[0], transb[0], mmax, nmax, k_gpu, *alpha, devPtrA, m_gpu,
                devPtrB, k_gpu, *beta, devPtrC, m_gpu, (int)stream[i%NSTREAMS]);

    if(status) printf("FERMI DGEMM ERROR: %u in loop %d \n",status,i);

    status = fermiGetMatrix (mmax, nmax, sizeof(C[0]), devPtrC, m_gpu, C+c_offset_gpu, *ldc, (int)stream[i%NSTREAMS]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) 
          printf ("!!!! device access error on node %s rank %d (read C at %s:%d) %d %s\n", 
              host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
    }

    a_offset_gpu += mmax;
    c_offset_gpu += mmax;
    devPtrA += mmax;
    devPtrC += mmax;
}
    cudaEventRecord(stop,0);

    dgemm_mkl(transa, transb, &m_cpu, &n_cpu, &k_cpu, alpha, A+a_offset, lda,
              B+b_offset, ldb, beta,C+c_offset, ldc);
    if(n_cpu2){
    a_offset = 0;
       if(transb[0]=='N'||transb[0]=='n')  b_offset = n_gpu*(*ldb);
       else b_offset = n_gpu;
    
    c_offset = n_gpu*(*ldc);
        dgemm_mkl(transa, transb, &m_gpu, &n_cpu2, &k_cpu, alpha, A+a_offset, lda,
              B+b_offset, ldb, beta,C+c_offset, ldc);
    }

    time_dgemm_mkl=wallclock()-time_dgemm;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&eventTimer, start, stop); 
    time_cuda=(double)eventTimer/1000.0; 
    time_dgemm=wallclock()-time_dgemm;

      }
    else
      { 
       time_dgemm=wallclock();
       cudaEventRecord(start,0);
#ifdef AUTOSPLIT
       used_split = splits[0+100*(counter%gpu_per_node)];
#else
       used_split = split;
#endif
       n_gpu = imin(ceil(*n/NN*used_split)*NN+NN ,(*n)-(*n)%NN);
       n_cpu = *n - n_gpu;
       a_offset = 0;
       if(transb[0]=='N'||transb[0]=='n')  b_offset = (size_t)((double)(*ldb)*(double)(n_gpu));
       else b_offset = (size_t)(n_gpu);
       c_offset = (size_t)((double)(*ldc)*(double)(n_gpu));
       a_offset_gpu = 0;
       b_offset_gpu = 0;
       c_offset_gpu = 0;

       //nmax=((scratch_size)/8/(*k + *m)/NN)*NN + NN;
       //if(*m<23040) nmax=((scratch_size/2)/8/(*k + *m)/NN)*NN + NN;
#ifdef USE_FERMI_DGEMM
       nmax = (SM_COUNT*32/NN)*NN;
#else
       nmax = 512;
#endif
       while(8*(nmax*(*k + *m))>=scratch_size) nmax-=NN;
       nmax = imin (n_gpu, nmax);
       if(nmax==*k) nmax-=NN;
       iter = n_gpu/nmax;
       while(n_gpu>iter*nmax) iter++;
       m_gpu = m_cpu = (*m/NM)*NM;

       mmax = ((8*1024*1024)/8/(*k)/NM)*NM;
       int iter_m = m_gpu/mmax;
       while(m_gpu>iter_m*mmax) iter_m++;
       mmax1 = m_gpu-(iter_m-1)*mmax;

#ifdef VERBOSE_PRINT
    printf("%d m= %3dx%4d,n= %3dx%4d\n",myrank,iter_m,mmax, iter,nmax);
#endif
    devPtrB= dev_scratch[1];
if(transb[0]=='N'||transb[0]=='n')
    status = fermiSetMatrix (k_gpu, nmax, sizeof(B[0]), B+b_offset_gpu, *ldb, devPtrB, k_gpu, (int)0);
else
    status = fermiSetMatrix (nmax, k_gpu, sizeof(B[0]), B+b_offset_gpu, *ldb, devPtrB,  nmax, (int)0);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) 
          printf ("!!!! device access error on node %s rank %d (write B at %s:%d) %d %s\n", 
              host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
    }

    devPtrC= devPtrB + k_gpu*nmax;
    devPtrA= dev_scratch[0];

for(i=0; i<iter_m; i++)
{
   mmax = imin(mmax,m_gpu-mmax*i);

    status = fermiSetMatrix (mmax, nmax, sizeof(C[0]), C+c_offset_gpu, *ldc, devPtrC, m_gpu, (int)stream[i%3]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) 
          printf ("!!!! device access error on node %s rank %d (write C at %s:%d) %d %s\n", 
              host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
    }

    status = fermiSetMatrix (mmax, k_gpu, sizeof(A[0]), A+a_offset_gpu, *lda, devPtrA, m_gpu, (int)stream[i%3]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) 
          printf ( "!!!! device access error on node %s rank %d (write A at %s:%d) %d %s\n", 
              host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
    }

if(transb[0]=='N'||transb[0]=='n')
      status = fermiDgemm_stream(transa[0], transb[0], mmax, nmax, k_gpu, *alpha, devPtrA, m_gpu,
                devPtrB, k_gpu, *beta, devPtrC, m_gpu, (int)stream[i%3]);
else
      status = fermiDgemm_stream(transa[0], transb[0], mmax, nmax, k_gpu, *alpha, devPtrA, m_gpu,
                devPtrB,  nmax, *beta, devPtrC, m_gpu, (int)stream[i%3]);

    if(status) printf("FERMI DGEMM ERROR on node %s rank %d : %u in loop %d \n",host_name,myrank,status,i);

    status = fermiGetMatrix (mmax, nmax, sizeof(C[0]), devPtrC, m_gpu, C+c_offset_gpu, *ldc, (int)stream[i%3]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) 
          printf ("!!!! device access error on node %s rank %d (read C at %s:%d) %d %s\n", 
              host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
    }

    a_offset_gpu += mmax;
    c_offset_gpu += mmax;
    devPtrA += mmax;
    devPtrC += mmax;
}

devPtrA= dev_scratch[0];

#ifndef USE_FERMI_DGEMM
    cudaEventRecord(stop,0);
#endif

if(transb[0]=='N'||transb[0]=='n'){
   b_offset_gpu = (*ldb)*nmax;
   c_offset_gpu = (*ldc)*nmax;
   b_offset_gpu1 = (*ldb)*nmax;
   c_offset_gpu1 = (*ldc)*nmax;
   b_offset_gpu2 = (*ldb)*nmax;
   c_offset_gpu2 = (*ldc)*nmax;
   b_offset_gpu3 = (*ldb)*nmax;
   c_offset_gpu3 = (*ldc)*nmax;
   }
else{
   b_offset_gpu = nmax;
   c_offset_gpu = (*ldc)*nmax;
   b_offset_gpu1 = nmax;
   c_offset_gpu1 = (*ldc)*nmax;
   b_offset_gpu2 = nmax;
   c_offset_gpu2 = (*ldc)*nmax;
   b_offset_gpu3 = nmax;
   c_offset_gpu3 = (*ldc)*nmax;
}
   outer_iter = ceil((double)(iter-1)/(double)NSTREAMS);
   last_iter = (iter-1)-(outer_iter-1)*NSTREAMS;
   nmax1 = nmax;

for(j=0; j<outer_iter; j++){

  if(j==outer_iter-1){ iter = last_iter; }
  else { iter = NSTREAMS; }

for(i=0; i<iter; i++)
{
    nmax = imin(nmax1,n_gpu-nmax1*(1+NSTREAMS*j+i));
    devPtrB= dev_scratch[((i+1)%NSTREAMS)+1];
    devPtrC= devPtrB + k_gpu*nmax;

if(transb[0]=='N'||transb[0]=='n')
    status = fermiSetMatrix (k_gpu, nmax, sizeof(B[0]), B+b_offset_gpu1, *ldb, devPtrB, k_gpu, (int)stream[i%NSTREAMS]);
else
    status = fermiSetMatrix (nmax, k_gpu, sizeof(B[0]), B+b_offset_gpu1, *ldb, devPtrB,  nmax, (int)stream[i%NSTREAMS]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) 
          printf ("!!!! device access error on node %s rank %d (write B at %s:%d) %d %s\n", 
              host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
    }

    status = fermiSetMatrix (m_gpu, nmax, sizeof(C[0]), C+c_offset_gpu1, *ldc, devPtrC, m_gpu, (int)stream[i%NSTREAMS]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) 
          printf ("!!!! device access error on node %s rank %d (write C at %s:%d) %d %s\n", 
              host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
    }

if(transb[0]=='N'||transb[0]=='n')    b_offset_gpu1 += (*ldb)*nmax1;
else    b_offset_gpu1 += nmax1;
    c_offset_gpu1 += (*ldc)*nmax1;
#ifdef INTERLEAVE
}//iter copy in
for(i=0; i<iter; i++)
{
#endif
    nmax = imin(nmax1,n_gpu-nmax1*(1+NSTREAMS*j+i));
    devPtrB= dev_scratch[((i+1)%NSTREAMS)+1];
    devPtrC= devPtrB + k_gpu*nmax;
if(transb[0]=='N'||transb[0]=='n')
      status = fermiDgemm_stream(transa[0], transb[0], m_gpu, nmax, k_gpu, *alpha, devPtrA, m_gpu,
                devPtrB, k_gpu, *beta, devPtrC, m_gpu, (int)stream[i%NSTREAMS]);
else
      status = fermiDgemm_stream(transa[0], transb[0], m_gpu, nmax, k_gpu, *alpha, devPtrA, m_gpu,
                devPtrB,  nmax, *beta, devPtrC, m_gpu, (int)stream[i%NSTREAMS]);

    if(status) printf("FERMI DGEMM ERROR on node %s rank %d : %u in loop %d \n",host_name,myrank,status,(1+NSTREAMS*j+i));

if(transb[0]=='N'||transb[0]=='n')    b_offset_gpu2 += (*ldb)*nmax1;
else    b_offset_gpu2 += nmax1;
    c_offset_gpu2 += (*ldc)*nmax1;
#ifdef INTERLEAVE
}//iter kernel

for(i=0; i<iter; i++)
{
#endif
    nmax = imin(nmax1,n_gpu-nmax1*(1+NSTREAMS*j+i));
    devPtrB= dev_scratch[((i+1)%NSTREAMS)+1];
    devPtrC= devPtrB + k_gpu*nmax;

    status = fermiGetMatrix (m_gpu, nmax, sizeof(C[0]), devPtrC, m_gpu, C+c_offset_gpu3, *ldc, (int)stream[i%NSTREAMS]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) 
          printf ("!!!! device access error on node %s rank %d (read C at %s:%d) %d %s\n", 
              host_name,myrank,__FILE__, __LINE__, status, cudaGetErrorString(err));
    }

if(transb[0]=='N'||transb[0]=='n')    b_offset_gpu3 += (*ldb)*nmax1;
else    b_offset_gpu3 += nmax1;
    c_offset_gpu3 += (*ldc)*nmax1;
}//iter copy out

}//outer_iter

    cudaEventRecord(stop,0);

    dgemm_mkl(transa, transb, m, &n_cpu, &k_cpu, alpha, A+a_offset, lda, 
              B+b_offset, ldb, beta,C+c_offset, ldc);
    m_cpu2 = 0;
    if (*m-m_cpu)
    {
      m_cpu2 = *m-m_cpu;
      dgemm_mkl(transa, transb, &m_cpu2, &n_gpu, k, alpha, A+m_cpu, lda, 
                B, ldb, beta,C+m_cpu, ldc);
    }


    time_dgemm_mkl=wallclock()-time_dgemm;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&eventTimer, start, stop); 
    time_cuda=(double)eventTimer/1000.0; 
    time_dgemm=wallclock()-time_dgemm;

    }



    if ( *k == *n )
    {
      ratio = (time_cuda)/time_dgemm_mkl;
      CPU_GFLOPS = 2.e-9*((double)(m_cpu)*(double)(n_cpu)*(double)(*k))/time_dgemm_mkl;
      GPU_GFLOPS = 2.e-9*((double)(m_gpu)*(double)(n_gpu)*(double)(*k))/time_cuda;
      GFLOPS = 2.e-9*(double)(*m) *(double)(*n)*(double)(*k)/time_dgemm;
      opt_split = GPU_GFLOPS/(GPU_GFLOPS+CPU_GFLOPS);
#ifdef VERBOSE_PRINT
      //printf ("%2d %5d (%5d %5d) %5d %5d (%7.3f %7.3f) RATIO: %5.4f (%4.1f CPU + %6.1f GPU = %6.1f GFLOPS)  USED_SPLIT: %5.3f OPT_SPLIT:%5.3f \n", myrank,*m,m_cpu,m_gpu,*k,*n, time_dgemm_mkl,  time_cuda, ratio, CPU_GFLOPS, GPU_GFLOPS, GFLOPS, used_split, opt_split);
#endif
      if(opt_split > 0.98) opt_split = 0.98;
      if(opt_split < 0.10) opt_split = 0.10;
      splits[*k/NK] = opt_split+0.001;
    }
    else
    {
      ratio = (time_cuda)/time_dgemm_mkl;
      CPU_GFLOPS = 2.e-9*((double)(m_cpu)*(double)(n_cpu)*(double)(*k)+(double)(m_cpu2)*(double)(*n)*(double)(*k))/time_dgemm_mkl;
      GPU_GFLOPS = 2.e-9*((double)(m_gpu)*(double)(n_gpu)*(double)(*k))/time_cuda;
      GFLOPS = 2.e-9*(double)(*m) *(double)(*n)*(double)(*k)/time_dgemm;
      opt_split = GPU_GFLOPS/(GPU_GFLOPS+CPU_GFLOPS);
#ifdef VERBOSE_PRINT
      printf ("%2d %5d %5d %5d (%5d %5d) (%7.3f %7.3f) RATIO: %5.4f (%4.1f CPU + %6.1f GPU = %6.1f GFLOPS)  USED_SPLIT: %5.3f OPT_SPLIT:%5.3f \n", myrank,*m,*k,*n, n_cpu,n_gpu, time_dgemm_mkl,  time_cuda, ratio, CPU_GFLOPS, GPU_GFLOPS, GFLOPS, used_split, opt_split); 
#endif

#ifdef AUTOSPLIT
      if(first_split) 
      {
        splits[0+100*(counter%gpu_per_node)] = opt_split+0.001; 
        split = opt_split+0.001;
        first_split=0;
      }

      if(used_split>opt_split) opt_split = 0.5*used_split + 0.5*opt_split;

      if(opt_split > 0.98) opt_split = 0.98;
      if(opt_split < 0.10) opt_split = 0.10;
      splits[0+100*(counter%gpu_per_node)] = opt_split;
      counter++;
#endif 

#ifdef DGEMM_ACCUM_TIMER
    time_dgemm_total += time_dgemm;
    //time_dgemm_total_gpu += time_cuda;
    //time_dgemm_total_cpu += time_dgemm_mkl;
    printf(" proc %d, DGEMM TIMER: %f TOTAL: %f \n",myrank, time_dgemm, time_dgemm_total);
#endif
    }

}


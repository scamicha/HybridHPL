/*
# Copyright (c) 2011,  NVIDIA CORPORATION
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

// hand-coded DGEMM routines for Fermi

#include "fermi_dgemm.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>

#define imin(a,b) (((a)<(b))?(a):(b))



#ifdef USE_FERMI_DGEMM
#define USE_TEXTURE_FOR_GLOBAL_LOADS
// can't use sizeof(struct) because of potential padding issues
#define TRANSPOSE_ARGS_SIZE 36
struct transpose_args {
  long long dst, src;
  int width, height, in_pitch, out_pitch, elem_size;
};

#define DTRSM_GPU_64_MM_KERNEL_ARGS_SIZE 32
struct dtrsm_gpu_64_mm_kernel_args {
  long long A,B;
  int N, texoff, lda, ldb;
};

#define DTRSM_GPU_64_MM_RT_KERNEL_ARGS_SIZE 32
struct dtrsm_gpu_64_mm_RT_kernel_args {
  long long A,B;
  int N, texoff, lda, ldb;
};

static const unsigned char helper_nvcc_cubin_bits[] = {
#include "helper_nvcc.cubin_bits.h"
};

#define DGEMM_NN_E_KERNEL_ARGS_SIZE 96
struct dgemm_nn_e_kernel_args {
  double alpha, beta;
  long long a_pitch, a_start, a_step1;
  long long b_pitch, b_start, b_step1, b_step2;
  long long c_pitch, c_start;
  int k_major, k_minor;
};

#define DGEMM_NT_E_KERNEL_ARGS_SIZE 88
struct dgemm_nt_e_kernel_args {
  double alpha, beta;
  long long a_pitch, a_start, a_step1;
  long long b_pitch, b_start, b_step1;
  long long c_pitch, c_start;
  int k_major, k_minor;
};

#define DGEMM_NT_TEX_KERNEL_ARGS_SIZE 48
struct dgemm_nt_tex_kernel_args {
  double alpha, beta;
  float a_step1_dy;
  float b_step1_dy;
  long long c_pitch, c_start;
  int k_major, k_minor;
};

static const unsigned char dgemm_kernels_sass_cubin_bits[] = {
#include "dgemm_kernels_sass.cubin_bits.h"
};
#endif

static int done_init = 0;
static int use_hand_coded_kernels = 1;  // use the fast ones by default
static int use_native_nn_kernel = 0;    // use the native NN or transpose+NT?
static CUcontext ctx;


#ifdef USE_FERMI_DGEMM
static CUmodule mod_help, mod_kern;
static CUfunction func_transpose, func_dgemm_nn_e, func_dgemm_nt_e, func_dtrsm_gpu_64_mm, func_dtrsm_gpu_64_mm_RT;
static CUtexref texref_Atex;
#ifdef USE_TEXTURE_FOR_GLOBAL_LOADS
static int use_tex_based_kernels = 1;   // use the tex-based kernels by default
static CUfunction func_dgemm_nt_tex, func_dgemm_nt_tex_64;
static CUtexref texref_a, texref_b;
#endif
#endif

#define MAX_STREAMS 4
int num_streams = 0;
static CUstream streams[MAX_STREAMS+1] = { 0 };

// as a temporary hack, we may transpose one of the inputs to map onto a mode we support


#ifdef USE_FERMI_DGEMM
#define MAX_TRANSPOSE  32*1024*1024
static CUdeviceptr buffers[MAX_STREAMS+1] = { 0 };
#endif

#define ERRCODE(e) (-(__LINE__ * 1000 + (e)))

static int fermi_dgemm_init(void)
{
  // this is the standard trick for making sure the runtime has been initialized (so we can
  //   borrow its context)
  cudaError_t cudares = cudaFree(0);
  if(cudares != cudaSuccess) return(-1);

  // use environment variables to override defaults
  if(getenv("FERMI_DGEMM_USE_NVCC_OUTPUT") != 0) use_hand_coded_kernels = 0;
  if(getenv("FERMI_DGEMM_USE_HAND_CODED") != 0) use_hand_coded_kernels = 1;

  if(getenv("FERMI_DGEMM_NATIVE_NN") != 0) use_native_nn_kernel = 1;
  if(getenv("FERMI_DGEMM_NN_VIA_TRANSPOSE") != 0) use_native_nn_kernel = 0;

#ifdef USE_TEXTURE_FOR_GLOBAL_LOADS
  if(getenv("FERMI_DGEMM_USE_TEXTURE") != 0) use_tex_based_kernels = 1;
  if(getenv("FERMI_DGEMM_USE_LOAD") != 0) use_tex_based_kernels = 0;
#endif

  // now grab the context
  CUresult res;
#if (CUDART_VERSION < 4000)
  if((res = cuCtxAttach(&ctx, 0)) != CUDA_SUCCESS) return(ERRCODE(res));
#else
  if((res = cuCtxGetCurrent(&ctx)) != CUDA_SUCCESS) return(ERRCODE(res));
#endif


#ifdef USE_FERMI_DGEMM
  // load modules (embedded in this file)
  if((res = cuModuleLoadData(&mod_help, helper_nvcc_cubin_bits)) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuModuleLoadData(&mod_kern, dgemm_kernels_sass_cubin_bits)) != CUDA_SUCCESS) return(ERRCODE(res));

  // look up functions
  if((res = cuModuleGetFunction(&func_transpose, mod_help, "transpose")) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuModuleGetFunction(&func_dtrsm_gpu_64_mm, mod_help, "dtrsm_gpu_64_mm")) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuModuleGetFunction(&func_dtrsm_gpu_64_mm_RT, mod_help, "dtrsm_gpu_64_mm_RT")) != CUDA_SUCCESS) return(ERRCODE(res));
#ifndef FORCE_NN_VIA_TRANSPOSE
  if((res = cuModuleGetFunction(&func_dgemm_nn_e, mod_kern, "dgemm_nn_e_kernel")) != CUDA_SUCCESS) return(ERRCODE(res));
#endif
  if((res = cuModuleGetFunction(&func_dgemm_nt_e, mod_kern, "dgemm_nt_e_kernel")) != CUDA_SUCCESS) return(ERRCODE(res));

#ifdef USE_TEXTURE_FOR_GLOBAL_LOADS
  if((res = cuModuleGetFunction(&func_dgemm_nt_tex, mod_kern, "dgemm_nt_tex_kernel")) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuModuleGetFunction(&func_dgemm_nt_tex_64, mod_kern, "dgemm_nt_tex_64_kernel")) != CUDA_SUCCESS) return(ERRCODE(res));
#endif

  // only need to set parameter sizes for each function once
  if((res = cuParamSetSize(func_transpose, TRANSPOSE_ARGS_SIZE) ) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuParamSetSize(func_dtrsm_gpu_64_mm, DTRSM_GPU_64_MM_KERNEL_ARGS_SIZE) ) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuParamSetSize(func_dtrsm_gpu_64_mm_RT, DTRSM_GPU_64_MM_RT_KERNEL_ARGS_SIZE) ) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuModuleGetTexRef(&texref_Atex, mod_help, "Atex")) != CUDA_SUCCESS) return(ERRCODE(res));
#ifndef FORCE_NN_VIA_TRANSPOSE
  if((res = cuParamSetSize(func_dgemm_nn_e, DGEMM_NN_E_KERNEL_ARGS_SIZE) ) != CUDA_SUCCESS) return(ERRCODE(res));
#endif
  if((res = cuParamSetSize(func_dgemm_nt_e, DGEMM_NT_E_KERNEL_ARGS_SIZE) ) != CUDA_SUCCESS) return(ERRCODE(res));

#ifdef USE_TEXTURE_FOR_GLOBAL_LOADS
  if((res = cuParamSetSize(func_dgemm_nt_tex, DGEMM_NT_TEX_KERNEL_ARGS_SIZE) ) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuParamSetSize(func_dgemm_nt_tex_64, DGEMM_NT_TEX_KERNEL_ARGS_SIZE) ) != CUDA_SUCCESS) return(ERRCODE(res));

  if((res = cuModuleGetTexRef(&texref_a, mod_kern, "a_texref")) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuModuleGetTexRef(&texref_b, mod_kern, "b_texref")) != CUDA_SUCCESS) return(ERRCODE(res));
#endif

  // get some temp space for a transpose buffer
  if(!use_native_nn_kernel) {
    if((res = cuMemAlloc(&buffers[0], MAX_TRANSPOSE)) != CUDA_SUCCESS) return(ERRCODE(res));
  }
#endif
  done_init = 1;
  return 0;
}

// creates one or more streams for use with other fermiDgemm/fermi{Get,Set}Matrix calls
// streams will be numbered 1 .. 'count'
// returns 0 on success
int fermiCreateStreams(int count)
{
  int res, i;

  if(!done_init) {
    res = fermi_dgemm_init();
    if(res != 0) return res;
  }

  if(count > MAX_STREAMS) return(ERRCODE(0));

#ifdef VERBOSE_PRINT
  //printf("  creating %d streams \n",count);
#endif

  for(i = 1; i <= count; i++) {
    if((res = cuStreamCreate(&streams[i], 0)) != CUDA_SUCCESS) return(ERRCODE(res));


#ifdef USE_FERMI_DGEMM
    if(!use_native_nn_kernel) {
#ifdef VERBOSE_PRINT
      //if(i==1) printf("  Allocating transpose buffers: %lld MB \n",(count)*((MAX_TRANSPOSE)>>20));
#endif
      if((res = cuMemAlloc(&buffers[i], MAX_TRANSPOSE)) != CUDA_SUCCESS){
        printf("  ERROR allocating transpose buffers %lld MB \n",(count)*((MAX_TRANSPOSE)>>20));
        return(ERRCODE(res));
      }
    }
#endif
  }
  num_streams = count;
  return 0;
}
  
int fermiSyncStream(int stream)
{
  CUresult res;

  if(stream == 0) {
    // do full context sync
    if((res = cuCtxSynchronize()) != CUDA_SUCCESS) return(ERRCODE(res));
    return 0;
  }

  if(stream <= num_streams) {
    if((res = cuStreamSynchronize(streams[stream])) != CUDA_SUCCESS) return(ERRCODE(res));
    return 0;
  }

  // illegal stream number
  return(ERRCODE(res));
}

static inline CUdeviceptr rtptr_to_devptr(const void *rtptr)
{
  return(* (CUdeviceptr *) &rtptr);
}

// memcpy's from host2dev and dev2host using a given stream - otherwise identical to 
// cublas{Get,Set}Matrix
int fermiSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, int stream)
{
  CUresult res;
  CUDA_MEMCPY2D cpyinfo;

#ifdef FERMI_DGEMM_DISABLE_STREAMS
  stream = 0;
#endif
#ifdef FERMI_DGEMM_VERBOSE_PRINT
  printf("fermiSetMatrix(%d, %d, %d, %p, %d, %p, %d) stream=%d\n",
         rows, cols, elemSize, A, lda, B, ldb, stream);
#endif

  cpyinfo.srcMemoryType = CU_MEMORYTYPE_HOST;
  cpyinfo.srcHost = A;
  cpyinfo.srcXInBytes = 0;
  cpyinfo.srcY = 0;
  cpyinfo.srcPitch = lda * elemSize;
  cpyinfo.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  cpyinfo.dstDevice = rtptr_to_devptr(B);
  cpyinfo.dstXInBytes = 0;
  cpyinfo.dstY = 0;
  cpyinfo.dstPitch = ldb * elemSize;
  cpyinfo.WidthInBytes = rows * elemSize;
  cpyinfo.Height = cols;

  if(stream) {
    if((res = cuMemcpy2DAsync(&cpyinfo, streams[stream])) != CUDA_SUCCESS) return(ERRCODE(res));
  } else {
    if((res = cuMemcpy2DAsync(&cpyinfo, 0)) != CUDA_SUCCESS) return(ERRCODE(res));
    //if((res = cuMemcpy2D(&cpyinfo)) != CUDA_SUCCESS) return(ERRCODE(res));
  }
  return 0;
}

int fermiGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, int stream)
{
  CUresult res;
  CUDA_MEMCPY2D cpyinfo;

#ifdef FERMI_DGEMM_DISABLE_STREAMS
  stream = 0;
#endif
#ifdef FERMI_DGEMM_VERBOSE_PRINT
  printf("fermiGetMatrix(%d, %d, %d, %p, %d, %p, %d) stream=%d\n",
         rows, cols, elemSize, A, lda, B, ldb, stream);
#endif

  cpyinfo.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  cpyinfo.srcDevice = rtptr_to_devptr(A);
  cpyinfo.srcXInBytes = 0;
  cpyinfo.srcY = 0;
  cpyinfo.srcPitch = lda * elemSize;
  cpyinfo.dstMemoryType = CU_MEMORYTYPE_HOST;
  cpyinfo.dstHost = B;
  cpyinfo.dstXInBytes = 0;
  cpyinfo.dstY = 0;
  cpyinfo.dstPitch = ldb * elemSize;
  cpyinfo.WidthInBytes = rows * elemSize;
  cpyinfo.Height = cols;

  if(stream) {
    if((res = cuMemcpy2DAsync(&cpyinfo, streams[stream])) != CUDA_SUCCESS) return(ERRCODE(res));
  } else {
    if((res = cuMemcpy2DAsync(&cpyinfo, 0)) != CUDA_SUCCESS) return(ERRCODE(res));
    //if((res = cuMemcpy2D(&cpyinfo)) != CUDA_SUCCESS) return(ERRCODE(res));
  }
  return 0;
}

#ifdef USE_FERMI_DGEMM
int fermi_transpose(double *dst, double *src, int width, int height, int in_pitch, int out_pitch, int stream )
{
  struct transpose_args args;
  CUresult res;

  args.dst = *(long long *)&dst;
  args.src = *(long long *)&src;
  args.width = width;
  args.height = height;
  args.in_pitch = in_pitch;
  args.out_pitch = out_pitch;
  args.elem_size = sizeof(double);

  if((res = cuParamSetv(func_transpose, 0, &args, TRANSPOSE_ARGS_SIZE)) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuFuncSetBlockShape(func_transpose, 16, 16, 1)) != CUDA_SUCCESS) return(ERRCODE(res));
  if(stream) {
    if((res = cuLaunchGridAsync(func_transpose, ceil(width / 16), ceil(height / 16), streams[stream])) != CUDA_SUCCESS) return(ERRCODE(res));
  } else {
    if((res = cuLaunchGrid(func_transpose, ceil(width / 16), ceil(height / 16))) != CUDA_SUCCESS) return(ERRCODE(res));
  }
}

static int dtrsm_64_mm(int N, double *A, int texoff,  double *B, int lda, int ldb, int stream)
{
  struct dtrsm_gpu_64_mm_kernel_args args;
  CUresult res;

  args.N = N;
  args.A = *(long long *)&A;
  args.texoff = texoff;
  args.B = *(long long *)&B;
  args.lda = lda;
  args.ldb = ldb;

  if((res = cuParamSetv(func_dtrsm_gpu_64_mm, 0, &args, DTRSM_GPU_64_MM_KERNEL_ARGS_SIZE)) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuFuncSetBlockShape(func_dtrsm_gpu_64_mm, 16, 16, 1)) != CUDA_SUCCESS) return(ERRCODE(res));
  if(stream) {
    if((res = cuLaunchGridAsync(func_dtrsm_gpu_64_mm, (N+15)/16, 1, streams[stream])) != CUDA_SUCCESS) return(ERRCODE(res));
  } else {
    if((res = cuLaunchGrid(func_dtrsm_gpu_64_mm, (N+15)/16, 1)) != CUDA_SUCCESS) return(ERRCODE(res));
  }

  return 0;
}

static int dtrsm_64_mm_RT(int N, double *A, int texoff,  double *B, int lda, int ldb, int stream)
{
  struct dtrsm_gpu_64_mm_RT_kernel_args args;
  CUresult res;

  args.N = N;
  args.A = *(long long *)&A;
  args.texoff = texoff;
  args.B = *(long long *)&B;
  args.lda = lda;
  args.ldb = ldb;

  if((res = cuParamSetv(func_dtrsm_gpu_64_mm_RT, 0, &args, DTRSM_GPU_64_MM_RT_KERNEL_ARGS_SIZE)) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuFuncSetBlockShape(func_dtrsm_gpu_64_mm_RT, 16, 16, 1)) != CUDA_SUCCESS) return(ERRCODE(res));
  if(stream) {
    if((res = cuLaunchGridAsync(func_dtrsm_gpu_64_mm_RT, (N+15)/16, 1, streams[stream])) != CUDA_SUCCESS) return(ERRCODE(res));
  } else {
    if((res = cuLaunchGrid(func_dtrsm_gpu_64_mm_RT, (N+15)/16, 1)) != CUDA_SUCCESS) return(ERRCODE(res));
  }

  return 0;
}
#endif

int dtrsm_gpu( char side, char uplo, char transa, 
               char diag, int M, int N, double alpha,
               double *A, int lda, double *B, int ldb, int stream)
{

#ifndef USE_FERMI_DGEMM
  cublasSetKernelStream(streams[stream]);
  cublasDtrsm(side,uplo,transa,diag,M,N,alpha,A,lda,B,ldb);
  return 0;
#else
  int res;
  size_t tex_offset;
  int j,jb;
  double beta = -alpha;
  long long buffer64 = buffers[stream];
  long long buffer64_2;
  const int nb1=128;
  const int nb2=64;

  if(!done_init) {
    res = fermi_dgemm_init();
    if(res != 0) return res;
  }

  res = cuTexRefSetFormat(texref_Atex, CU_AD_FORMAT_SIGNED_INT32, 2);


  if(transa == 'N' || transa == 'n')
  {
    if((res = cuTexRefSetAddress(&tex_offset, texref_Atex, rtptr_to_devptr(A), 2*(lda)*(M)*sizeof(double))) != CUDA_SUCCESS) return(ERRCODE(res));
    if(tex_offset!=0) printf("ERROR: TEXTURE OFFSET = %d \n",tex_offset);

    if(M%128)
    {
      if(stream) buffer64_2 = buffers[0];
      else       buffer64_2 = buffers[1];
    }


    for (j=0; j<M; j+= nb1)
    {
      jb = imin(nb1, M-j);
      // Update row-wise
      if(j>0)
      {
        if(jb==64)
        {
          res = fermi_transpose(*(void **)&buffer64_2, B, N, j, ldb, N, stream );
          res = fermiDgemm_stream('N','T', jb, N, j, beta, A+j, lda, *(void **)&buffer64_2, N, alpha, B+j, M, stream);
        }
        else
        {
          res = fermiDgemm_stream(transa, transa, jb, N, j, beta, A+j, lda, B, M, alpha, B+j, M, stream);
        }
      }
      if (jb == 128)
      {
        res = dtrsm_64_mm(N,A,(lda)*j+j,B+j,lda,ldb,stream);
        res = fermi_transpose(*(void **)&buffer64, B+j, N, 64, ldb, N, stream );
        res = fermiDgemm_stream('N','T',64,N,64,beta,A+(lda)*j+j+64,lda,*(void **)&buffer64,N,alpha,B+j+64,ldb, stream);
        res = dtrsm_64_mm(N,A,(lda)*j+j+lda*64+64,B+j+64,lda,ldb,stream);
      }
      else
      {
        cublasDtrsm (side, uplo, transa, diag, jb, N, alpha, A+(lda)*j+j, M, B+j, M);
      }
    }
  }
  else
  {
    if((res = cuTexRefSetAddress(&tex_offset, texref_Atex, rtptr_to_devptr(A), 2*(lda)*(N)*sizeof(double))) != CUDA_SUCCESS) return(ERRCODE(res));
    if(tex_offset!=0) printf("ERROR: TEXTURE OFFSET = %d \n",tex_offset);

    for (j=0; j<N; j+=nb2)
    {
      jb = imin(nb2, N-j);
      // Update row-wise
      if(j>0)
      {
        res = fermiDgemm_stream('N','T', M, jb, j, beta, B, ldb, A+j, lda, alpha, B+(ldb)*j, ldb, stream);
      }
      res = dtrsm_64_mm_RT(M,A,(lda)*j+j,B+(ldb)*j,lda,ldb,stream);
    }
  }

   return res;
#endif
}


#ifdef USE_FERMI_DGEMM
static int dgemm_nt_e(int m, int n, int k,
                      double alpha, const double *A, int lda, const double *B, int ldb,
                      double beta, double *C, int ldc, int stream)
{
  struct dgemm_nt_e_kernel_args args;
  CUresult res;

  args.alpha = alpha;
  args.beta = beta;
  args.a_pitch = lda * sizeof(double);
  args.a_start = *(long long *)&A;
  args.a_step1 = args.a_pitch * 8;
  args.b_pitch = ldb * sizeof(double);
  args.b_start = *(long long *)&B;
  args.b_step1 = args.b_pitch * 8;
  args.c_pitch = ldc * sizeof(double);
  args.c_start = *(long long *)&C;
  args.k_major = k / 16;
  args.k_minor = 0;

  if((res = cuParamSetv(func_dgemm_nt_e, 0, &args, DGEMM_NT_E_KERNEL_ARGS_SIZE)) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuFuncSetBlockShape(func_dgemm_nt_e, 256, 1, 1)) != CUDA_SUCCESS) return(ERRCODE(res));

  if(stream) {
    if((res = cuLaunchGridAsync(func_dgemm_nt_e, n / 64, m / 64, streams[stream])) != CUDA_SUCCESS) return(ERRCODE(res));
  } else {
    if((res = cuLaunchGrid(func_dgemm_nt_e, n / 64, m / 64)) != CUDA_SUCCESS) return(ERRCODE(res));
  }

  return 0;
}

#ifdef USE_TEXTURE_FOR_GLOBAL_LOADS
static int dgemm_nt_tex(int m, int n, int k,
                        double alpha, const double *A, int lda, const double *B, int ldb,
                        double beta, double *C, int ldc, int stream)
{
  struct dgemm_nt_tex_kernel_args args;
  CUresult res;
  CUDA_ARRAY_DESCRIPTOR a_fmtdesc, b_fmtdesc;

  a_fmtdesc.Width = m >> 1;  // each int4 texel holds 2 doubles
  a_fmtdesc.Height = k;
  a_fmtdesc.NumChannels = 4;
  a_fmtdesc.Format = CU_AD_FORMAT_SIGNED_INT32;
  b_fmtdesc.Width = n >> 1;  // each int4 texel holds 2 doubles
  b_fmtdesc.Height = k;
  b_fmtdesc.NumChannels = 4;
  b_fmtdesc.Format = CU_AD_FORMAT_SIGNED_INT32;

  if((res = cuTexRefSetAddress2D(texref_a, &a_fmtdesc, rtptr_to_devptr(A), lda * sizeof(double))) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuTexRefSetAddress2D(texref_b, &b_fmtdesc, rtptr_to_devptr(B), ldb * sizeof(double))) != CUDA_SUCCESS) return(ERRCODE(res));

  args.alpha = alpha;
  args.beta = beta;
  args.a_step1_dy = 8;
  args.b_step1_dy = 8;
  args.c_pitch = ldc * sizeof(double);
  args.c_start = *(long long *)&C;
  args.k_major = k / 16;
  args.k_minor = 0;

  // pick the right kernel based on pointer size of C
  CUfunction tex_func = func_dgemm_nt_tex;
  if((sizeof(CUdeviceptr) > 4) && ((args.c_start >> 32) != 0)) {
    //printf("C is not in bottom 32-bits of address space - using 64-bit kernel!\n");
    tex_func = func_dgemm_nt_tex_64;
    // kernel isn't fully 64-bit safe - all of C needs to sit in same aligned 4GB
    long long c_end = args.c_start + n * args.c_pitch;
    if((args.c_start >> 32) != (c_end >> 32)) {
      printf("HELP!  C straddles a 4GB boundary!\n");
      return(ERRCODE(0));
    }
  }

  if((res = cuParamSetv(tex_func, 0, &args, DGEMM_NT_TEX_KERNEL_ARGS_SIZE)) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuFuncSetBlockShape(tex_func, 256, 1, 1)) != CUDA_SUCCESS) return(ERRCODE(res));

  if(stream) {
    if((res = cuLaunchGridAsync(tex_func, n / 64, m / 64, streams[stream])) != CUDA_SUCCESS) return(ERRCODE(res));
  } else {
    if((res = cuLaunchGrid(tex_func, n / 64, m / 64)) != CUDA_SUCCESS) return(ERRCODE(res));
  }

  return 0;
}
#endif

static int dgemm_nn_e(int m, int n, int k,
                      double alpha, const double *A, int lda, const double *B, int ldb,
                      double beta, double *C, int ldc, int stream)
{
#ifndef FORCE_NN_VIA_TRANSPOSE
  if(use_native_nn_kernel  || n==k || m<8*k) {
    struct dgemm_nn_e_kernel_args args;
    CUresult res;

    args.alpha = alpha;
    args.beta = beta;
    args.a_pitch = lda * sizeof(double);
    args.a_start = *(long long *)&A;
    args.a_step1 = args.a_pitch * 4;
    args.b_pitch = ldb * sizeof(double);
    args.b_start = *(long long *)&B;
    args.b_step1 = 128LL; //args.b_pitch * 16;
    args.b_step2 = 0; //256LL - args.b_pitch * 16;
    args.c_pitch = ldc * sizeof(double);
    args.c_start = *(long long *)&C;
    args.k_major = k / 32;
    args.k_minor = 0;

    if((res = cuParamSetv(func_dgemm_nn_e, 0, &args, DGEMM_NN_E_KERNEL_ARGS_SIZE)) != CUDA_SUCCESS) return(ERRCODE(res));
    if((res = cuFuncSetBlockShape(func_dgemm_nn_e, 256, 1, 1)) != CUDA_SUCCESS) return(ERRCODE(res));
    if(stream) {
      if((res = cuLaunchGridAsync(func_dgemm_nn_e, n / 32, m / 128, streams[stream])) != CUDA_SUCCESS) return(ERRCODE(res));
    } else {
      if((res = cuLaunchGrid(func_dgemm_nn_e, n / 32, m / 128)) != CUDA_SUCCESS) return(ERRCODE(res));
    }

    return 0;
  }
#endif

  // if we fall through to here, we're doing NN via a transpose (either because we were asked to, or if that's
  //   all we know how to do)
  struct transpose_args args;
  CUresult res;

  args.dst = buffers[stream];
  args.src = *(long long *)&B;
  args.width = n;
  args.height = k;
  args.in_pitch = ldb;
  args.out_pitch = n;
  args.elem_size = sizeof(double);

  if((res = cuParamSetv(func_transpose, 0, &args, TRANSPOSE_ARGS_SIZE)) != CUDA_SUCCESS) return(ERRCODE(res));
  if((res = cuFuncSetBlockShape(func_transpose, 16, 16, 1)) != CUDA_SUCCESS) return(ERRCODE(res));
  if(stream) {
    if((res = cuLaunchGridAsync(func_transpose, ceil(n / 16), ceil(k / 16), streams[stream])) != CUDA_SUCCESS) return(ERRCODE(res));
  } else {
    if((res = cuLaunchGrid(func_transpose, ceil(n / 16), ceil(k / 16))) != CUDA_SUCCESS) return(ERRCODE(res));
  }

  long long buffer64 = buffers[stream];

#ifdef USE_TEXTURE_FOR_GLOBAL_LOADS
      if(use_tex_based_kernels&&m<128000&&n<128000)
      return dgemm_nt_tex(m, n, k, alpha, A, lda, *(void **)&buffer64, n, beta, C, ldc, stream);
      else
#endif
  return dgemm_nt_e(m, n, k, alpha, A, lda, *(void **)&buffer64, n, beta, C, ldc, stream);
}
#endif

// returns 0 on success
// non-zero return means nth parameter (starting counting at 1) has an unsupported value
int fermiDgemm(char transa, char transb, int m, int n, int k,
               double alpha, const double *A, int lda, 
               const double *B, int ldb, double beta, double *C, 
               int ldc)
{
  return fermiDgemm_stream(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 0);
}

int fermiDgemm_stream(char transa, char transb, int m, int n, int k,
                      double alpha, const double *A, int lda, 
                      const double *B, int ldb, double beta, double *C, 
                      int ldc, int stream)
{

#ifndef USE_FERMI_DGEMM
  cublasSetKernelStream(streams[stream]);
  cublasDgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
  return 0;
#else

#ifdef FERMI_DGEMM_VERBOSE_PRINT
  printf("fermiDgemm('%c', '%c', %d, %d, %d, %f, %p, %d, %p, %d, %f, %p, %d)\n",
         transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif

  int res;
  if(!done_init) {
    res = fermi_dgemm_init();
    if(res != 0) return res;
  }

  if((stream < 0) || (stream > num_streams)) return(14);

  switch(transa) {
  case 'N':
  case 'n':
    switch(transb) {
    case 'N':
    case 'n':
#ifndef FORCE_NN_VIA_TRANSPOSE
      // NN requires M to be multiple of 128, N to be multiple of 32, K to be multiple of 64
      if(m % 128) return 3;
      if(n % 32) return 4;
      if(k % 64) return 5;
#endif
      // we support NN via transpose also, so for now, enforce the NT constraints too
      if(m % 64) return 3;
      if(n % 64) return 4;
      if(k % 16) return 5;
      return dgemm_nn_e(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);

    case 'T':
    case 't':
      // NT requires M to be multiple of 64, N to be multiple of 64, K to be multiple of 16
      if(m % 64) return 3;
      if(n % 64) return 4;
      if(k % 16) return 5;
#ifdef USE_TEXTURE_FOR_GLOBAL_LOADS
      if(use_tex_based_kernels&&m<128000&&n<128000)
      return dgemm_nt_tex(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
      else
#endif
      return dgemm_nt_e(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);

    default:
      // illegal value for transb
      return 2;
    }

  case 'T':
  case 't':
    // no support for transposed A yet
    return 1;

  default:
    // illegal value for transa
    return 1;
  }
#endif
}

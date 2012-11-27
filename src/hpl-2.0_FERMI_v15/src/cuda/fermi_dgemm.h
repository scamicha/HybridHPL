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

// hand-coded DGEMM kernels for Fermi

// intended to work just like cublas (i.e. runtime-style parameters), although only for a subset of 
//   the parameter space

#ifndef FERMI_DGEMM_H
#define FERMI_DGEMM_H

// creates one or more streams for use with other fermiDgemm/fermi{Get,Set}Matrix calls
// streams will be numbered 1 .. 'count'
// returns 0 on success
int fermiCreateStreams(int count);
int fermiSyncStream(int stream);

// memcpy's from host2dev and dev2host using a given stream - otherwise identical to 
// cublas{Get,Set}Matrix
int fermiSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, int stream);
int fermiGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, int stream);

// returns 0 on success
// non-zero return means nth parameter (starting counting at 1) has an unsupported value
int fermiDgemm(char transa, char transb, int m, int n, int k,
               double alpha, const double *A, int lda, 
               const double *B, int ldb, double beta, double *C, 
               int ldc);

int fermiDgemm_stream(char transa, char transb, int m, int n, int k,
                      double alpha, const double *A, int lda, 
                      const double *B, int ldb, double beta, double *C, 
                      int ldc, int stream);

int fermi_transpose(double *dst, double *src, int width, int height, int in_pitch, int out_pitch, int stream );

int dtrsm_gpu( char side, char uplo, char transa,
               char diag, int M, int N, double alpha,
               double *A, int lda, double *B, int ldb, int stream);

#endif

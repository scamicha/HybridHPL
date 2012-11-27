###########################################################
#  Makefile for Fermi DGEMM/DTRSM library                 #
###########################################################
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

all: libdgemm.so.1.0.1

OBJS = cuda_dgemm.o fermi_dgemm.o

.PRECIOUS: $(OBJS)

all : libdgemm.so.1.0.1 

fermi_dgemm.o : fermi_dgemm.c fermi_dgemm.h helper_nvcc.cubin_bits.h dgemm_kernels_sass.cubin_bits.h

DEFINES = -DMPI
#DEFINES += -DUSE_FERMI_DGEMM
#DEFINES += -DVERBOSE_PRINT
#DEFINES += -DACML
#DEFINES += -DGOTO

%.o: %.c
	mpicc -O0 -c -fPIC $(DEFINES) $*.c -o $*.o -I/usr/local/cuda/include

libdgemm.so.1.0.1: $(OBJS)

	mpicc -O3 -shared -Wl,-soname,libdgemm.so.1 -o libdgemm.so.1.0.1 $(OBJS) -L/usr/local/cuda/lib64 -lcudart -lcuda
	ln -sf libdgemm.so.1.0.1 libdgemm.so.1.0
	ln -sf libdgemm.so.1.0 libdgemm.so.1
	ln -sf libdgemm.so.1 libdgemm.so

clean:
	rm -f $(OBJS) $(CUBINS) libdgemm.so.1.0.1 libdgemm.so.1.0 libdgemm.so.1 libdgemm.so

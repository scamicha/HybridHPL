#########################################################
This is a CUDA-enabled version of HPL optimized for Tesla 20-series GPUs

version 1.5
Authors: Everett Phillips and Massimiliano Fatica 
#########################################################

You will need a Linux workstation or cluster with:

1) MPI:
    the software has been tested with Openmpi 1.4.1 and Intel MPI, both with the gcc and Intel compiler.
    As of version 1.3 you can use infiniband with RDMA enabled
2) Intel MKL 10.x or gotoBlas2 or ACML
3) CUDA 4.X (driver 270+)
4) Tesla GPUs

There is one makefile:

1) Make.CUDA: it will build a CUDA accelerated HPL using pinned/page-locked system memory. This is required to 
   overlap GPU computations with data transfers.

The code has been tested with RHEL/CENTOS 5.3 and 6.2

Details of the implementations are in the paper: 
 "Accelerating linpack with CUDA on heterogenous clusters"
 ACM International Conference Proceeding Series; Vol. 383 archive
 Proceedings of 2nd Workshop on General Purpose Processing on Graphics Processing Units 
 ISBN:978-1-60558-517-8
 Author:	Massimiliano Fatica, NVIDIA Corporation, Santa Clara, CA

#########################################################
How to build:
#########################################################

In the Make.CUDA:

1) TOPdir (line 82) should point to the full path of the hpl-2.0_FERMI_v15 directory

2) LADir ( line 110) should point to the MKL,gotoblas2, or ACML directory

3) CCFLAGS (line 198 or 202) Choose either gcc or intel compiler flags  

4) Compile with
        make

NOTE:   if using GOTOBlas2 or ACML you will need to enable the line "DEFINES += -DGOTO" or "DEFINES += -DACML
        in the src/cuda/Makefile. This makefile is also assuming that CUDA is installed in the default location 
        /usr/local/cuda. If this is not the case on your system, you will need to edit it.

#########################################################
How to run:
#########################################################

In the  bin/CUDA directory, there are:
  1)  HPL.dat benchmark configuration
  2)  run_linpack script

After the build is complete, the xhpl executable should be there too.

in HPL.dat:

N is the problem size, larger values of N typically give better performance, however the size is 
  limited by the available system memory.  A rough estimate of memory size required is (N)*(N)*8 bytes.  

NB should be a multiple of 128 (for best performance). It will also work with
NB being a multiple of 64 but with lower performance.  768 typically gives best results
larger values may give better results (1024) if several GPUs share the same PCIe connection

The code is also expecting L1 in no-transposed form. U could be either transposed or non transposed.

The HPL.dat file should have two lines similar to these:
1            L1 in (0=transposed,1=no-transposed) form
1 or 0       U  in (0=transposed,1=no-transposed) form

Other HPL.dat settings are described in the "TUNING" document in the hpl directory

In run_linpack:

 you will need to change HPL_DIR to reflect the location on your system.

The other parameters are important for tuning:
 CPU_CORES_PER_GPU  is the number of cpu cores per GPU used for the benchmark
    For example, a system with 2 GPUs and 8 cpu cores will have CPU_CORES_PER_GPU=4

 CUDA_DGEMM_SPLIT is the percentage of work sent to the GPU for DGEMM and is roughly equal to (GPU GFLOPS)/(GPU GFLOPS + CPU GFLOPS)
    or approximately ( 350 ) / ( 350 + cpu cores per gpu * 4 * cpu frequency in GHz )

 CUDA_DTRSM_SPLIT is the percentage of work sent to the GPU for DTRSM and is typically lower than DGEMM split by 0.05-0.10

The code is expecting a 1:1 mapping between GPUs and MPI processes.
P*Q should be equal to the number of GPUs on your system or cluster.

Start the benchmark as a normal MPI run :
  mpirun -np 1 ./run_linpack 
  mpirun -np 1 -host compute-0-3 ./run_linpack
  mpirun -np 16 -hostfile myhostfile ./run_linpack

memory/cpu affinity is important for optimal performance on NUMA systems

  example: an 8 node cluster with 2 GPU and 2 CPU socket per node:
    mpirun -np 16 -hostfile myhostfile -bysocket -bind-to-socket ./run_linpack

  example: 8 node cluster with 1 GPU and 2 CPU sockets per node:
    mpirun -np 8 -hostfile myhostfile numactl --interleave=all ./run_linpack

########################################
Typical results
########################################

single node: dual X5550 (8 cores) + M2050: 
(vary block size)

mpirun -np 1 -host c0-4 /home/mfatica/Testing/hpl-2.0_FERMI_v3/bin/CUDA_pinned/run_linpack 
================================================================================
HPLinpack 2.0  --  High-Performance Linpack benchmark  --   September 10, 2008
Written by A. Petitet and R. Clint Whaley,  Innovative Computing Laboratory,
UTK
Modified by Piotr Luszczek, Innovative Computing Laboratory, UTK
Modified by Julien Langou, University of Colorado Denver
================================================================================

An explanation of the input/output parameters follows:
T/V    : Wall time / encoded variant.
N      : The order of the coefficient matrix A.
NB     : The partitioning blocking factor.
P      : The number of process rows.
Q      : The number of process columns.
Time   : Time in seconds to solve the linear system.
Gflops : Rate of execution for solving the linear system.

The following parameter values will be used:

N      :   25000 
NB     :     768      384      512 
PMAP   : Row-major process mapping
P      :       1 
Q      :       1 
PFACT  :    Left 
NBMIN  :       2 
NDIV   :       2 
RFACT  :    Left 
BCAST  :   1ring 
DEPTH  :       1 
SWAP   : Mix (threshold = 192)
L1     : no-transposed form
U      : no-transposed form
EQUIL  : yes
ALIGN  : 8 double precision words

--------------------------------------------------------------------------------

- The matrix A is randomly generated for each test.
- The following scaled residual check will be computed:
      ||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || b ||_oo ) * N )
- The relative machine precision (eps) is taken to be
  1.110223e-16
- Computational tests pass if scaled residuals are less than
  16.0

================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       25000   768     1     1              35.21              2.959e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0039927 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       25000   384     1     1              35.79              2.911e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0048654 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       25000   512     1     1              34.16              3.050e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0041599 ...... PASSED
================================================================================

Finished      3 tests with the following results:
              3 tests completed and passed residual checks,
              0 tests completed and failed residual checks,
              0 tests skipped because of illegal input values.
--------------------------------------------------------------------------------

End of Tests.
================================================================================


2 nodes:  dual X5550 + M2050 + QDR infiniband: 
(vary block size)

 mpirun -np 2 --mca btl_openib_flags 1 -host c0-2,c0-4 /home/mfatica/Testing/hpl-2.0_FERMI_v3/bin/CUDA_pinned/run_linpack 
================================================================================
HPLinpack 2.0  --  High-Performance Linpack benchmark  --   September 10, 2008
Written by A. Petitet and R. Clint Whaley,  Innovative Computing Laboratory, UTK
Modified by Piotr Luszczek, Innovative Computing Laboratory, UTK
Modified by Julien Langou, University of Colorado Denver
================================================================================

An explanation of the input/output parameters follows:
T/V    : Wall time / encoded variant.
N      : The order of the coefficient matrix A.
NB     : The partitioning blocking factor.
P      : The number of process rows.
Q      : The number of process columns.
Time   : Time in seconds to solve the linear system.
Gflops : Rate of execution for solving the linear system.

The following parameter values will be used:

N      :   25000 
NB     :     768      384      512 
PMAP   : Row-major process mapping
P      :       1 
Q      :       2 
PFACT  :    Left 
NBMIN  :       2 
NDIV   :       2 
RFACT  :    Left 
BCAST  :   1ring 
DEPTH  :       1 
SWAP   : Mix (threshold = 192)
L1     : no-transposed form
U      : no-transposed form
EQUIL  : yes
ALIGN  : 8 double precision words

--------------------------------------------------------------------------------

- The matrix A is randomly generated for each test.
- The following scaled residual check will be computed:
      ||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || b ||_oo ) * N )
- The relative machine precision (eps) is taken to be               1.110223e-16
- Computational tests pass if scaled residuals are less than                16.0

================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       25000   768     1     2              26.68              3.905e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0043331 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       25000   384     1     2              31.44              3.314e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0036738 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       25000   512     1     2              27.73              3.757e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0040684 ...... PASSED
================================================================================

Finished      3 tests with the following results:
              3 tests completed and passed residual checks,
              0 tests completed and failed residual checks,
              0 tests skipped because of illegal input values.
--------------------------------------------------------------------------------

End of Tests.
================================================================================


sinlge node, dual X5550 + M2050: 
(vary problem size)

mpirun -np 1 ./run_linpack

================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       10000   512     1     1               3.37              1.979e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0049402 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       15000   512     1     1               9.10              2.473e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0051511 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       20000   512     1     1              19.13              2.789e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0040776 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       25000   512     1     1              35.49              2.935e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0039469 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       30000   512     1     1              55.40              3.250e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0049416 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       35000   512     1     1              86.43              3.307e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0045906 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       40000   512     1     1             127.78              3.339e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0045028 ...... PASSED
================================================================================


single node, 2 GPU: dual X5670 (12-cores) 72 GB ram + dual M2050: 
(vary problem size)

mpirun -np 1 numactl -l -cpunodebind=0 ./run_linpack : -np 1 numatcl -l -cpunodebind=1 ./run_linpack
================================================================================
HPLinpack 2.0  --  High-Performance Linpack benchmark  --   September 10, 2008
Written by A. Petitet and R. Clint Whaley,  Innovative Computing Laboratory, UTK
Modified by Piotr Luszczek, Innovative Computing Laboratory, UTK
Modified by Julien Langou, University of Colorado Denver
================================================================================

An explanation of the input/output parameters follows:
T/V    : Wall time / encoded variant.
N      : The order of the coefficient matrix A.
NB     : The partitioning blocking factor.
P      : The number of process rows.
Q      : The number of process columns.
Time   : Time in seconds to solve the linear system.
Gflops : Rate of execution for solving the linear system.

The following parameter values will be used:

N      :    8000    16000    23040    32000    38000    40000    44800    50000 
           51200    91000    95232
NB     :     768 
PMAP   : Row-major process mapping
P      :       1 
Q      :       2 
PFACT  :    Left 
NBMIN  :       2 
NDIV   :       2 
RFACT  :    Left 
BCAST  :   1ring 
DEPTH  :       1 
SWAP   : Mix (threshold = 192)
L1     : no-transposed form
U      : no-transposed form
EQUIL  : yes
ALIGN  : 8 double precision words

--------------------------------------------------------------------------------

- The matrix A is randomly generated for each test.
- The following scaled residual check will be computed:
      ||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || b ||_oo ) * N )
- The relative machine precision (eps) is taken to be               1.110223e-16
- Computational tests pass if scaled residuals are less than                16.0
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2        8000   768     1     2               1.86              1.832e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0046684 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       16000   768     1     2               8.08              3.382e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0048961 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       23040   768     1     2              19.55              4.170e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0045637 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       32000   768     1     2              42.91              5.092e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0049350 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       38000   768     1     2              66.44              5.506e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0039959 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       40000   768     1     2              76.02              5.613e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0042210 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       44800   768     1     2             102.72              5.836e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0042740 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       50000   768     1     2             136.54              6.103e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0042087 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       51200   768     1     2             149.95              5.968e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0038833 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       91000   768     1     2             710.23              7.074e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0037939 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       95232   768     1     2             817.33              7.045e+02
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0037939 ...... PASSED
================================================================================

Finished     11 tests with the following results:
             11 tests completed and passed residual checks,
              0 tests completed and failed residual checks,
              0 tests skipped because of illegal input values.
--------------------------------------------------------------------------------

End of Tests.
================================================================================


8 nodes cluster:  dual X5550 quad-core CPU (8 cores), 48GB ram + M2050 + QDR infiniband: 
(vary problem size, block size, broadcast method)

mpirun -np 8 -machinefile mfile --mca btl_openib_flags 1 ./run_linpack
================================================================================
HPLinpack 2.0  --  High-Performance Linpack benchmark  --   September 10, 2008
Written by A. Petitet and R. Clint Whaley,  Innovative Computing Laboratory, UTK
Modified by Piotr Luszczek, Innovative Computing Laboratory, UTK
Modified by Julien Langou, University of Colorado Denver
================================================================================

An explanation of the input/output parameters follows:
T/V    : Wall time / encoded variant.
N      : The order of the coefficient matrix A.
NB     : The partitioning blocking factor.
P      : The number of process rows.
Q      : The number of process columns.
Time   : Time in seconds to solve the linear system.
Gflops : Rate of execution for solving the linear system.

The following parameter values will be used:

N      :   60000   120000   180000 
NB     :     512      640      768 
PMAP   : Row-major process mapping
P      :       2 
Q      :       4 
PFACT  :    Left 
NBMIN  :       2 
NDIV   :       2 
RFACT  :    Left 
BCAST  :   1ring    2ring 
DEPTH  :       1 
SWAP   : Mix (threshold = 192)
L1     : no-transposed form
U      : no-transposed form
EQUIL  : yes
ALIGN  : 8 double precision words

--------------------------------------------------------------------------------

- The matrix A is randomly generated for each test.
- The following scaled residual check will be computed:
      ||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || b ||_oo ) * N )
- The relative machine precision (eps) is taken to be               1.110223e-16
- Computational tests pass if scaled residuals are less than                16.0

================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       60000   512     2     4              80.97              1.779e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0034350 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR12L2L2       60000   512     2     4              78.68              1.830e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0035919 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       60000   640     2     4              76.67              1.878e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0036865 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR12L2L2       60000   640     2     4              74.94              1.922e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0033675 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       60000   768     2     4              80.82              1.782e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0036951 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR12L2L2       60000   768     2     4              80.66              1.785e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0035146 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2      120000   512     2     4             476.01              2.420e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0034855 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR12L2L2      120000   512     2     4             487.75              2.362e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0030334 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2      120000   640     2     4             461.91              2.494e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0030477 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR12L2L2      120000   640     2     4             457.95              2.516e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0029293 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2      120000   768     2     4             478.78              2.406e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0033352 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR12L2L2      120000   768     2     4             476.92              2.416e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0030474 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2      180000   512     2     4            1504.91              2.584e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0032407 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR12L2L2      180000   512     2     4            1505.75              2.582e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0031792 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2      180000   640     2     4            1500.12              2.592e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0028839 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR12L2L2      180000   640     2     4            1489.50              2.610e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0032978 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2      180000   768     2     4            1483.75              2.620e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0029121 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR12L2L2      180000   768     2     4            1495.07              2.601e+03
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0033042 ...... PASSED
================================================================================

Finished     18 tests with the following results:
             18 tests completed and passed residual checks,
              0 tests completed and failed residual checks,
              0 tests skipped because of illegal input values.
--------------------------------------------------------------------------------

End of Tests.
================================================================================

IBM Idataplex cluster, Intel MPI:

mpirun -np 64 -env I_MPI_PIN_DOMAIN socket $HPL_DIR/bin/CUDA/xhpl
================================================================================
HPLinpack 2.0  --  High-Performance Linpack benchmark  --   September 10, 2008
Written by A. Petitet and R. Clint Whaley,  Innovative Computing Laboratory,
UTK
Modified by Piotr Luszczek, Innovative Computing Laboratory, UTK
Modified by Julien Langou, University of Colorado Denver
================================================================================

An explanation of the input/output parameters follows:
T/V    : Wall time / encoded variant.
N      : The order of the coefficient matrix A.
NB     : The partitioning blocking factor.
P      : The number of process rows.
Q      : The number of process columns.
Time   : Time in seconds to solve the linear system.
Gflops : Rate of execution for solving the linear system.

The following parameter values will be used:

N      :  200000 
NB     :     768 
PMAP   : Row-major process mapping
P      :       8        4 
Q      :       8       16 
PFACT  :    Left 
NBMIN  :       2 
NDIV   :       2 
RFACT  :    Left 
BCAST  :  1ringM 
DEPTH  :       1 
SWAP   : Spread-roll (long)
L1     : no-transposed form
U      : transposed form
EQUIL  : yes
ALIGN  : 8 double precision words

--------------------------------------------------------------------------------

- The matrix A is randomly generated for each test.
- The following scaled residual check will be computed:
      ||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || b ||_oo ) * N )
- The relative machine precision (eps) is taken to be
  1.110223e-16
- Computational tests pass if scaled residuals are less than
  16.0

================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR11L2L2      200000   768     8     8             389.77              1.368e+04
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0023570 .....  PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR11L2L2      200000   768     4    16             421.94              1.264e+04
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0018508 ...... PASSED
================================================================================
Finished      2 tests with the following results:
              2 tests completed and passed residual checks,
              0 tests completed and failed residual checks,
              0 tests skipped because of illegal input values.
--------------------------------------------------------------------------------

End of Tests.
================================================================================


##########################################
Advanced tuning
##########################################
Adding -DVERBOSE_PRINT  to  the Makefile in src/cuda will enable verbose timing for each of the DGEMM and DTRSM calls sent to the GPU ( I/O time, GPU time, CPU time, overall performance).


#########################################################
Known issues:
#########################################################
1) When running multiple processes on the same host, please be sure that you
don't set KMP_AFFINITY=nowarnings,compact. OpenMPI will hang.
If you still want to use the flag, you will need to start each process 
mpirun --mca btl_openib_flags 1  -np 1 -host c0-2 taskset -c 0-3 /home/hpl-2.0_FERMI_v3/bin/CUDA_pinned/run_linpack : 
                                 -np 1 -host c0-2 taskset -c 4-7 /home/hpl-2.0_FERMI_v3/bin/CUDA_pinned/run_linpack
################################################################
 Fixed bugs
################################################################
1) fixed bug in the assignDevice code when multiple GPUs were present.



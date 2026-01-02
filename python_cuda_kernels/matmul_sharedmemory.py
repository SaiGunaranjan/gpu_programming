# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 15:11:39 2026

@author: Sai Gunaranjan
"""

"""
Important points to keep in mind:
    1. Inside numba cuda kernels, the datatype should be numba.float32/float64 and we should not use np. or cp.float32/64
    2. cuda.local.array and cuda.shared.array should have static sizes/shapes and cannot be dynamic!

Im getting the current warning for the mat_mul_shared_mem implementation:
    Grid size 1 will likely result in GPU under-utilization due to low occupancy.

Understand and fix this!

In this code, I have implemented the matrix multiplication using the shared memory. The idea is
as follows:
    1. Let us say we want to compute C = A * B (* denotes the matrix multiplication). For simplicity,
    lets assume all are square matrices of shape N x N.
    The output of a particular row of C requires fetching that particular row of A. In other words, to compute every element of a
    row of C, we need that particular row of A. So if there are N elements in a row of C, we need to fetch that particular row of A
    N times from global memory! (similarly for a given column of C, we need to fetch that colum of B N times!) And fetches from global memory are very slow! So, how do we avoid these multiple fetches of the same elements?
    2. We can view the large matrix multiplication as a tiled/block matrix multiplication of smaller blocks/tiles.
    3. If we view it this way, the row elements in the block of C require the same block row of A.
    4. But fetching the block of A,B(from global memory) and putting in shared memory, all the threads in the same block can share this block of A,B
    5. This reduces the number of fetches from global memory!

Right now, this implementation works only for square matrices with block size of C = block size of A,B
The CPU and GPU outputs are matching. Next, I will extend it to non square matrices as well and also
compare the timings with global memory only based mat mul on GPU.

My implementation is based on the below video lecture:
    https://www.youtube.com/watch?v=Q3GgbfGTnVc

"""

import numpy as np
import numba
from numba import cuda
import cupy as cp

@cuda.jit
def mat_mul_shared_mem(A,B,C):

    global_thrdid_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    global_thrdid_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if ((global_thrdid_x >= C.shape[1]) or (global_thrdid_y >= C.shape[0])):
        return

    local_thrdid_x = cuda.threadIdx.x
    local_thrdid_y = cuda.threadIdx.y

    # mat_a_tile = cuda.shared.array((cuda.blockDim.y,cuda.blockDim.x),dtype=cp.float32)
    # mat_b_tile = cuda.shared.array((cuda.blockDim.y,cuda.blockDim.x),dtype=cp.float32)

    mat_a_tile = cuda.shared.array((32,32),dtype=numba.float32) # This may not be true for non square matrices!
    mat_b_tile = cuda.shared.array((32,32),dtype=numba.float32)


    num_times_loading = cuda.gridDim.x #blocks_per_grid_x # This is also not true and will depend on how I tile

    cum_sum = 0# cuda.local.array, register
    for i in range(num_times_loading):

        mat_a_tile[local_thrdid_y,local_thrdid_x] = A[global_thrdid_y,local_thrdid_x + i*cuda.blockDim.x]
        mat_b_tile[local_thrdid_y,local_thrdid_x] = B[local_thrdid_y + i*cuda.blockDim.y,global_thrdid_x]

        cuda.syncthreads()
        k = cuda.blockDim.x # For a square matrix only! Need to modify for non square matrix
        for j in range(k): # k is the column dimension of the mat_a_tile = row dimension of mat_b_tile
            cum_sum += mat_a_tile[local_thrdid_y,j] * mat_b_tile[j,local_thrdid_x]
        cuda.syncthreads()

    C[global_thrdid_y,global_thrdid_x] = cum_sum


# @cuda.jit()
# def mat_mul_global_mem(A,B,C):




num_rows = 32
num_cols = 32
A = np.random.randn(num_rows,num_cols).astype(np.float32)
B = np.random.randn(num_rows,num_cols).astype(np.float32)
C = A @ B
threads_per_block_x = 32
threads_per_block_y = 32
threads_per_block = (threads_per_block_x,threads_per_block_y)
blocks_per_grid_x = np.ceil(num_cols/threads_per_block_x).astype(np.int32)
blocks_per_grid_y = np.ceil(num_rows/threads_per_block_y).astype(np.int32)
blocks_per_grid = (blocks_per_grid_x,blocks_per_grid_y)

A_ = cp.asarray(A)
B_ = cp.asarray(B)
# C_ = cp.zeros((A.shape[0],B.shape[1]),dtype=A.dtype)
C_ = cp.zeros((A.shape[0],B.shape[1]),dtype=cp.float32)
mat_mul_shared_mem[blocks_per_grid,threads_per_block](A_,B_,C_)
C_np = cp.asnumpy(C_)
total_err = np.sum(C-C_np)

print(f"Total abs error b/w cpu and gpu implementation = {total_err:.7f}")


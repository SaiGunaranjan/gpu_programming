# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 22:29:39 2025

@author: Sai Gunaranjan
"""

""" In this code, I have implemented 2 flavours of softmax function specifically
catering to very large number of elements ~ 10 million.
Flavour 1: When the compute memory is constrained

When we cannot accomodate 10 million elements in one go to compute the softmax, we perform
a tiling or sliding window based method of computing the maximum value per tile/window and
then use this to compute the local denominator sum. We then move to the next window and update the
max value and the denominator sum. This approach is similar to the Flash Attention for 1D case.

Flavour 2: When compute units are constrained

Softmax requires normalizing each value by the largest value to ensure stability of the exponential function.
Computing max element in an array is O(N) and when N is large (like 10 million), this can be huge.
This is where parallelization on GPU helps. By launching k threads, we can cut down this compute to
O(N/k). Each thread works on a chunk of data and computes the local maximum and the local denominator sum
using the local maxima. Then the outputs from each of the threads can be combined either again on GPU or
even on CPU (since by now majority of the compute is already done) to obatain the global maxima and denominator sum


I have implemented both these flavours of softmax and compared against the pytorch implementation.
I have used different metrics to compare the performance like:
    1. Average error per sample
    2. Sum of softmax must sum to 1
    3. KL divergence between true pytorch implementation and each method
"""

import numpy as np
import cupy as cp
import torch
from numba import cuda
import math
from time import time
np.random.seed(100)


size_matrix = 10000000

# Generate Logits
mean = 0
std_dev = 10
logits = np.random.normal(loc=mean, scale=std_dev, size=(size_matrix,)).astype(np.float32)

#%%
# This is the standard implementation of softmax using formula
t1 = time()
logits_ = logits - np.amax(logits)
numerator = np.exp(logits_)
denominator = np.sum(numerator)
soft_max_singleshot = numerator/denominator
t2 = time()
soft_max_singleshot_sum = np.sum(soft_max_singleshot)
print('Time for softmax computation using standard formula = {0:.0f} ms'.format((t2-t1)*1000))
#%%
# This is the implementation using torch library. This provides a bench mark to check if our implementation is right.
t3 = time()
logits_tensor = torch.from_numpy(logits)
soft_max_singleshot_torch = torch.softmax(logits_tensor,dim=0,dtype=torch.float32).numpy()
t4 = time()
soft_max_singleshot_torch_sum = np.sum(soft_max_singleshot_torch)
print('Time for softmax computation using pytorch implementation = {0:.0f} ms'.format((t4-t3)*1000))
#%%
# This implementation is a sliding window based softmax computation when there is a constraint on the memory.
# We perform a sliding window based computation of the softmax denominator by carefully updating the max value
# and the denominator sum from the previous window
chunk_size = np.int32(1024) # This is the window size of memory available to perform the computation
denom = 0
old_max_val = float('-inf')
t5 = time()
for i in range(0,size_matrix,chunk_size):
    if i+chunk_size > size_matrix:
        window_data = logits[i::]
    else:
        window_data = logits[i:i+chunk_size]

    new_max_val = max(old_max_val, np.amax(window_data)) # Update the max value from previous windows and current window
    x = np.exp(window_data - new_max_val)
    denom = np.exp(old_max_val-new_max_val)*denom + np.sum(x) # Adjustment factor
    old_max_val = new_max_val

soft_max_windowbased = np.exp(logits - new_max_val)/denom
t6 = time()
print('Time for softmax computation using Sliding window with fixed memory = {0:.0f} ms'.format((t6-t5)*1000))
soft_max_windowbased_sum = np.sum(soft_max_windowbased)
eps=1e-12
kl_torch_vs_window = np.sum(soft_max_singleshot_torch * (np.log(soft_max_singleshot_torch + eps) - np.log(soft_max_windowbased + eps)))
error_per_sample_window = np.sum(soft_max_singleshot_torch - soft_max_windowbased)/size_matrix
#%%

# Below kernel partitions the input vector of 10 million elements into chunks and each thread works on a chunk of data
# to compute the local max and the local denominator sum using the local max
@cuda.jit('void(float32[:],float32[:],float32[:],int32, int32)')
def local_sum_max_gpu(logits_gpu,max_per_chunk,sum_per_chunk,num_chunks,chunk_size):

    thrd_id = (cuda.blockIdx.x * cuda.blockDim.x) + cuda.threadIdx.x


    if thrd_id >= num_chunks:
        return

    if thrd_id*chunk_size + chunk_size > len(logits_gpu):
        window_data = logits_gpu[thrd_id*chunk_size::]
    else:
        window_data = logits_gpu[thrd_id*chunk_size:thrd_id*chunk_size + chunk_size]

    max_per_chunk[thrd_id] = max(window_data)

    for i in range(len(window_data)):
        sum_per_chunk[thrd_id] += math.exp(window_data[i]-max_per_chunk[thrd_id])



# Parallelizing across chunks
num_chunks = cp.int32(np.ceil(len(logits)/chunk_size))
threads_per_block = 32 # Threads per block or warp size
blocks_per_grid = int(np.ceil(num_chunks/threads_per_block))
t7 = time()
logits_gpu = cp.asarray(logits,dtype=cp.float32)
max_per_chunk = cp.zeros((num_chunks,),dtype=cp.float32)
sum_per_chunk = cp.zeros((num_chunks,),dtype=cp.float32)
local_sum_max_gpu[blocks_per_grid,threads_per_block](logits_gpu,max_per_chunk,sum_per_chunk,num_chunks,chunk_size)
cuda.synchronize()
# After obtaining the local max for each chunk/thread, we could launch another kernel to combine the local maxima and
# obtain a global maxima. But, since the majority of the compute is already done by each thread, the remaining compute
# of obtaining a global maxima from local maxima and adjusting the scaling facotors is much smaller and hence
# can be done even outside on the host.
global_max = cp.amax(max_per_chunk)
denom_gpu = cp.sum(cp.exp(max_per_chunk-global_max) * sum_per_chunk) # Adjust the scaling for each local denominator sum
soft_max_gpu = cp.exp(logits_gpu - global_max)/denom_gpu
t8 = time()
soft_max_gpu = cp.asnumpy(soft_max_gpu)
soft_max_gpu_sum = np.sum(soft_max_gpu)
print('Time for softmax computation using GPU = {0:.0f} ms'.format((t8-t7)*1000))
kl_torch_vs_gpu = np.sum(soft_max_singleshot_torch * (np.log(soft_max_singleshot_torch + eps) - np.log(soft_max_gpu + eps)))
error_per_sample_gpu = np.sum(soft_max_singleshot_torch - soft_max_gpu)/size_matrix

print("\nSum of softmax output using each approach:")
print(f"Sum of softmax from formula implementation = {soft_max_singleshot_sum: .4f}")
print(f"Sum of softmax from pytorch implementation = {soft_max_singleshot_torch_sum: .4f}")
print(f"Sum of softmax from sliding window implementation = {soft_max_windowbased_sum: .4f}")
print(f"Sum of softmax from GPU implementation = {soft_max_gpu_sum: .4f}")


print("\nKL divergence of each implementation wrt torch softmax:")
print(f"KL divergence b/w torch and window based softmax = {kl_torch_vs_window: .4f}")
print(f"KL divergence b/w torch and GPU based softmax = {kl_torch_vs_gpu: .4f}")

print("\n Average error per sample (EPS):")
print(f" EPS between torch implementation and window based implementation of softmax = {error_per_sample_window}")
print(f" EPS between torch implementation and GPU based implementation of softmax = {error_per_sample_gpu}")



# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:25:12 2020

@author: saiguna
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
import cupy as cp
from numbaCudaKernels import conv_1d


plt.close('all')
signal_len = 2**10
impulse_response_len = 2**4
# np.random.seed(10)
inp_signal_cpu = np.random.randn(signal_len).astype('float32');
impulse_response_cpu = np.random.randn(impulse_response_len).astype('float32');
t1 = time();
inp_signal = cp.asarray(inp_signal_cpu.copy(),dtype=cp.float32);
impulse_response = cp.asarray(impulse_response_cpu.copy(),dtype=cp.float32);
inp_signal_ext = cp.hstack((cp.zeros(impulse_response_len-1,dtype=cp.float32),inp_signal,cp.zeros(impulse_response_len-1,cp.float32)));
output_signal = cp.zeros((signal_len+impulse_response_len-1),dtype=cp.float32);
GPU_memorySize = (inp_signal_ext.nbytes + impulse_response.nbytes + output_signal.nbytes)/(1024);
print('Total memory on GPU = {0:.2f} kB'.format(GPU_memorySize));
signal_ext_len = len(inp_signal_ext);
t2 = time();
thrdPerBlock = 32;
blkPerGrid = int(cp.ceil(signal_ext_len/thrdPerBlock));

conv_1d[blkPerGrid,thrdPerBlock](inp_signal_ext,impulse_response,output_signal,signal_len,impulse_response_len);

output_signal_ker = cp.asnumpy(output_signal)
t3 = time();
# output_signal_cp = cp.asnumpy(cp.convolve(inp_signal,impulse_response));
output_signal_cpu = np.convolve(inp_signal_cpu,impulse_response_cpu);
t4 = time();

print('Data transfer time = {0:.0f} ms'.format((t2-t1)*1000));
print('GPU compute + Fetch time = {0:.0f} ms'.format((t3-t2)*1000));
print('Total GPU transfer + compute + fetch time = {0:.0f} ms'.format((t3-t1)*1000))
print('CPU compute time = {0:.0f} ms'.format((t4-t3)*1000));


if 1:
    plt.figure(1);
    plt.subplot(121)
    plt.plot(output_signal_cpu,'-o',label='cpu');
    # plt.plot(output_signal_cp,'-o',label='cupy');
    plt.plot(output_signal_ker,'-o',label='numba');
    plt.legend();
    plt.grid(True)
    plt.subplot(122)
    plt.title('Error b/w cpu and gpu versions')
    plt.plot(output_signal_cpu-output_signal_ker,'-o');
    plt.grid(True)


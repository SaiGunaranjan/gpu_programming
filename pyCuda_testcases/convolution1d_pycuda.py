# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:20:37 2020

@author: saiguna
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:25:12 2020

@author: saiguna
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from time import time
from pycuda import gpuarray
from pyCudaKernels import conv_1d


plt.close('all')
signal_len = np.int32(2**10)
impulse_response_len = np.int32(2**4)
# np.random.seed(10)
inp_signal_cpu = np.random.randn(signal_len).astype('float32');
impulse_response_cpu = np.random.randn(impulse_response_len).astype('float32');
inp_signal_ext = np.hstack((np.zeros(impulse_response_len-1,dtype=np.float32),inp_signal_cpu,np.zeros(impulse_response_len-1,np.float32)));
output_signal = np.zeros((signal_len+impulse_response_len-1),dtype=np.float32);
t1 = time();

inp_signal_ext_gpu = gpuarray.to_gpu(inp_signal_ext);
impulse_response_gpu = gpuarray.to_gpu(impulse_response_cpu.copy());
output_signal_gpu = gpuarray.to_gpu(output_signal);

GPU_memorySize = (inp_signal_ext.nbytes + impulse_response_cpu.nbytes + output_signal.nbytes)/(1024);
print('Total memory on GPU = {0:.2f} kB'.format(GPU_memorySize));
signal_ext_len = len(inp_signal_ext);
t2 = time();
thrdPerBlock = 32;
blkPerGrid = int(np.ceil(signal_ext_len/thrdPerBlock));

conv_1d(inp_signal_ext_gpu,impulse_response_gpu,output_signal_gpu,signal_len,impulse_response_len,grid=(blkPerGrid,1,1),block=(thrdPerBlock,1,1));

output_signal_ker = output_signal_gpu.get();

inp_signal_ext_gpu.gpudata.free();
impulse_response_gpu.gpudata.free();
output_signal_gpu.gpudata.free();



t3 = time();

output_signal_cpu = np.convolve(inp_signal_cpu,impulse_response_cpu);
t4 = time();

print('Data transfer time = {0:.0f} ms'.format((t2-t1)*1000));
print('pyCUDA GPU compute + Fetch time = {0:.0f} ms'.format((t3-t2)*1000));
print('Total GPU transfer + compute + fetch time using pyCUDA= {0:.0f} ms'.format((t3-t1)*1000))
print('CPU compute time = {0:.0f} ms'.format((t4-t3)*1000));


if 1:
    plt.figure(1,figsize=(20,10));
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


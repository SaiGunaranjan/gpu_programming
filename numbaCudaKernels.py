# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:22:01 2020

@author: Sai Gunaranjan Pelluri
"""

from numba import cuda


@cuda.jit('void(float32[:],float32[:],float32[:],int32,int32)')
def conv_1d(inp_signal_ext,impulse_response,output_signal,signal_len,impulse_response_len):
    
    thrdID = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if thrdID >= signal_len+impulse_response_len-1:
        return;
        
    temp_sum = 0;
    for ele in range(impulse_response_len):
        temp_sum += inp_signal_ext[thrdID+ele]*impulse_response[impulse_response_len-ele-1]
    output_signal[thrdID] = temp_sum
    


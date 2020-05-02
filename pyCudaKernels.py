# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:40:53 2020

@author: saiguna
"""

import sys
import os
import pycuda.autoinit
from pycuda.compiler import SourceModule

""" Add C compiler exe path to the os path"""
cCompilerExePath = 'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.23.28105\\bin\\Hostx64\\x64\\'
os.environ['PATH'] += ';'+ cCompilerExePath;

""" Add the NVIDIA CUDA toolkit path to the system path"""
cuda_include_path = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\include\\'
sys.path.append(cuda_include_path)

kernel = SourceModule("""
                      
                      
                      # include <math.h>
                      # include <cuComplex.h>
                      # define THREAD_ID (blockIdx.x * blockDim.x + threadIdx.x)
                      # define DBG 0

                      __global__ void conv_1d(float *inp_signal_ext, float *impulse_response, float *output_signal, int signal_len, int impulse_response_len)
                      {
                      
                      int thrdID =  THREAD_ID;
                      float temp_sum = 0;
                      int ele;
                      
                      
                      
                      if (thrdID >= signal_len + impulse_response_len -1 )
                      
                      {
                          return;
                      }
                      
                      
                      for (ele=0; ele< impulse_response_len; ele++)
                      {
                          temp_sum += inp_signal_ext[thrdID+ele]*impulse_response[impulse_response_len-ele-1];
                      }
                            
                      output_signal[thrdID] = temp_sum;
                      
                      }
       


                      
""", include_dirs=[cuda_include_path])
# """, options = ['--generate-line-info', '-O3', '--use_fast_math'], keep=True, include_dirs=[cuda_include_path])
                      
conv_1d = kernel.get_function("conv_1d")
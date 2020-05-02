# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:30:40 2019

@author: Sai Gunaranjan Pelluri
"""
####################################CFAR-OS###################################################
# FFT_SignalVector:Give Range FFT or Doppler FFT as your signal vector
# GuardBandLength : To discard some of the adjacent samples surrounding the Cell under test
# CFAR_Window_Length: Sliding window length 
# Threshold_Beta : Linear multiplicative factor
# OrderedStatisticIndex:Pick the Kth index peak out of the window values


import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import cfar_lib
from numbaCudaKernels import CFAR_CA_GPU, CFAR_OS_GPU
import cupy as cp
from time import time

 




""" Test case generation"""
plt.close('all')
num_fft = 1024
n_vec = np.arange(2*num_fft)
num_bjs = 5
freq_vec = np.array([20,150,300,350,400])
object_snr = np.array([15,3,12,10,5])
noise_power_db = -30 # Noise Power in dB
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = noise_variance*10**(object_snr/10)
fs = 1e3
radar_signal = np.zeros((2*num_fft)).astype('complex64')
for ele in np.arange(num_bjs):
    radar_signal += weights[ele]*np.exp(1j*2*np.pi*freq_vec[ele]*n_vec/fs)
wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),2*num_fft) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),2*num_fft)
radar_signal = radar_signal + wgn_noise
radar_signal_fft = np.fft.fft(radar_signal)
FFT_SignalVector = radar_signal_fft[0:num_fft]
freq_grid = np.arange(num_fft)*fs/(2*num_fft)
signal_mag = np.abs(FFT_SignalVector)**2
GuardBandLength = 1
valid_samp_len = 18
false_alarm_rate = 1e-5
OrderedStatisticIndex = 3 # 
##################################################################
""" CPU version"""

t1 = time()
bool_array_OS = cfar_lib.CFAR_OS(signal_mag,GuardBandLength,valid_samp_len,false_alarm_rate,OrderedStatisticIndex)
det_indices_OS = np.where(bool_array_OS>0)
det_freq_OS = det_indices_OS[0]*fs/(2*num_fft)
t2 = time()

bool_array_CA = cfar_lib.CFAR_CA(signal_mag,GuardBandLength,valid_samp_len,false_alarm_rate)
det_indices_CA = np.where(bool_array_CA>0)
det_freq_CA = det_indices_CA[0]*fs/(2*num_fft)
t3 = time()




""" CFAR CA GPU version """
t4 = time()
signal_mag_ext = np.hstack((np.flipud(signal_mag[1::]), signal_mag, np.flipud(signal_mag[0:-1])))  # Typically done to handle boundary condition for 1st sample of the Range FFT sample
signal_mag_ext = cp.asarray(signal_mag_ext.copy(),dtype=cp.float32);
guardBandLen_1side = np.int32(GuardBandLength);
validSampLen_1side = np.int32(valid_samp_len);
noiseMargin = np.float32(2*validSampLen_1side*(false_alarm_rate**(-1/(2*validSampLen_1side)) -1)) # multiplication by a factor of 2  to include valid samples both sides of the CUT
outputBoolVector = cp.zeros((len(signal_mag),),dtype=cp.int32)

t5 = time()
lenSpectrumSignal = len(signal_mag)
threadsPerBlock = 32;
blocksPerGrid = np.int32(np.ceil(len(signal_mag_ext)/threadsPerBlock));
scratchPad = cp.zeros((len(signal_mag),2*validSampLen_1side),dtype=cp.float32) # Make a separate scratch pad for every thread.
GPU_memorySize = (signal_mag_ext.nbytes + scratchPad.nbytes + outputBoolVector.nbytes)/(1024);

CFAR_CA_GPU[blocksPerGrid, threadsPerBlock](signal_mag_ext, lenSpectrumSignal, guardBandLen_1side, validSampLen_1side, scratchPad, noiseMargin, outputBoolVector)
outputBoolVector = cp.asnumpy(outputBoolVector)
t6 = time()
det_indices_CA_gpu = np.where(outputBoolVector>0)
det_freq_CA_gpu = det_indices_CA_gpu[0]*fs/(2*num_fft)
t7 = time()

""" CFAR OS GPU version """
signal_mag_ext = np.hstack((np.flipud(signal_mag[1::]), signal_mag, np.flipud(signal_mag[0:-1])))  # Typically done to handle boundary condition for 1st sample of the Range FFT sample
signal_mag_ext = cp.asarray(signal_mag_ext.copy(),dtype=cp.float32);
guardBandLen_1side = np.int32(GuardBandLength);
validSampLen_1side = np.int32(valid_samp_len);
outputBoolVector_gpuOS = cp.zeros((len(signal_mag),),dtype=cp.int32)
ordStat = np.int32(OrderedStatisticIndex)
scratchPad = cp.zeros((len(signal_mag),2*validSampLen_1side),dtype=cp.float32) # Make a separate scratch pad for every thread.
CFAR_OS_GPU[blocksPerGrid, threadsPerBlock](signal_mag_ext, lenSpectrumSignal, guardBandLen_1side, validSampLen_1side, scratchPad, noiseMargin, ordStat, outputBoolVector_gpuOS)
outputBoolVector_gpuOS = cp.asnumpy(outputBoolVector_gpuOS)
t8 = time()
det_indices_OS_gpu = np.where(outputBoolVector_gpuOS>0)
det_freq_OS_gpu = det_indices_OS_gpu[0]*fs/(2*num_fft)
t9 = time()

print('True frequencies', freq_vec)
print('Estimated frequencies OS: ', np.round(det_freq_OS))
print('Estimated GPU frequencies OS: ', np.round(det_freq_OS_gpu))
print('Estimated frequencies CA: ', np.round(det_freq_CA))
print('Estimated GPU frequencies CA: ', np.round(det_freq_CA_gpu))

print('CFAR OS CPU compute time = {0:.0f} ms'.format((t2-t1)*1000))
print('CFAR CA CPU compute time = {0:.0f} ms'.format((t3-t2)*1000))

print('Total memory on GPU = {0:.2f} kB'.format(GPU_memorySize));
print('Data transfer time = {0:.0f} ms'.format((t5-t4)*1000));
# print('GPU compute + Fetch time = {0:.0f} ms'.format((t6-t5)*1000));
# print('Total GPU transfer + compute + fetch time = {0:.0f} ms'.format((t6-t4)*1000))

print('CFAR CA GPU time = {0:.0f} ms'.format((t7-t4)*1000))
print('CFAR OS GPU time = {0:.0f} ms'.format((t9-t7)*1000))




plt.figure(1,figsize=(20,10))
plt.subplot(1,2,1)
plt.title('CFAR OS')
plt.plot(freq_grid,20*np.log10(np.abs(FFT_SignalVector)))
plt.plot(det_freq_OS,20*np.log10(np.abs(FFT_SignalVector[det_indices_OS[0]])),'rD', label='CFAR detected peaks');
plt.plot(det_freq_OS_gpu,20*np.log10(np.abs(FFT_SignalVector[det_indices_OS_gpu[0]])),'k*', alpha=0.5, label='CFAR GPU detected peaks')
plt.legend()
plt.xlabel('Frequency in Hz')
plt.ylabel('Power (dB)')
plt.grid(True)
plt.subplot(1,2,2)
plt.title('CFAR CA')
plt.plot(freq_grid,20*np.log10(np.abs(FFT_SignalVector)))
plt.plot(det_freq_CA,20*np.log10(np.abs(FFT_SignalVector[det_indices_CA[0]])),'rD',label='CFAR CPU detected peaks')
plt.plot(det_freq_CA_gpu,20*np.log10(np.abs(FFT_SignalVector[det_indices_CA_gpu[0]])),'k*', alpha=0.5, label='CFAR GPU detected peaks')
plt.legend()
plt.xlabel('Frequency in Hz')
plt.ylabel('Power (dB)')
plt.grid(True)






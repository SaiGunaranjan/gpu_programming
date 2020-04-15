# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:01:56 2020

@author: saiguna
"""


import numpy as np
import matplotlib.pyplot as plt
import cfar_lib
import cupy as cp
from numbaCudaKernels import CFAR_CA_2D_cross_GPU
from time import time


plt.close('all')
num_fft = 512
num_ramps = 512
num_objs = 4
fs_range = 1e3
fs_dopp = 1e3
range_freq_vec = np.array([-200,200,350,400]) # change frequencies to digital frequency index
doppler_freq_vec = np.array([-50,250,300,450]) # change frequencies to digital frequency index

range_freq_grid = np.arange(-num_fft//2,num_fft//2)*fs_range/(num_fft)
dopp_freq_grid = np.arange(-num_ramps//2,num_ramps//2)*fs_dopp/(num_ramps)

#object_snr = np.array([15,12,10,5])
object_snr = np.array([30,50,50,50])
noise_power_db = 0 # Noise Power in dB
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = noise_variance*10**(object_snr/10)

radar_signal = np.zeros((num_fft,num_ramps)).astype('complex64')
for ele in np.arange(num_objs):
    range_signal = weights[ele]*np.exp(1j*2*np.pi*range_freq_vec[ele]*np.arange(num_fft)/fs_range)
    doppler_signal = np.exp(1j*2*np.pi*doppler_freq_vec[ele]*np.arange(num_ramps)/fs_dopp)
    radar_signal += range_signal[:,None]*doppler_signal[None,:] # [range,num_ramps]
    
    
wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),num_fft*num_ramps) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_fft*num_ramps)
noise_signal = wgn_noise.reshape(num_fft,num_ramps)
radar_signal = radar_signal + noise_signal
radar_signal_range_fft = np.fft.fft(radar_signal,axis=0)/num_fft
radar_signal_range_fft_dopp_fft = np.fft.fft(radar_signal_range_fft,axis=1)/num_ramps
range_freq_vec[range_freq_vec<0] = range_freq_vec[range_freq_vec<0] + fs_range
range_bins_ind = (range_freq_vec/(fs_range/num_fft)).astype(int)
doppler_freq_vec[doppler_freq_vec<0] = doppler_freq_vec[doppler_freq_vec<0] + fs_dopp
doppler_bins_ind = (doppler_freq_vec/(fs_dopp/num_ramps)).astype(int)
signal_mag = (np.abs(radar_signal_range_fft_dopp_fft)**2).astype(np.float32)
guardband_len_x = np.int32(1)
guardband_len_y = np.int32(1)
valid_samp_len_x = np.int32(20)
valid_samp_len_y = np.int32(20)
false_alarm_rate = 1e-5#1e-4
OrderedStatisticIndex = 3 # 

print('True range bins:',range_bins_ind)
print('True Doppler bins:', doppler_bins_ind,'\n')


t1 = time()
bool_array_os = cfar_lib.CFAR_OS_2D(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate, OrderedStatisticIndex)
det_indices_os = np.where(bool_array_os>0)
print('CFAR OS det range bins:', det_indices_os[0])
print('CFAR OS det doppler bins:', det_indices_os[1],'\n')
t2 = time()
bool_array_osCross = cfar_lib.CFAR_OS_2D_cross(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate, OrderedStatisticIndex)
det_indices_osCross = np.where(bool_array_osCross>0)
print('CFAR OS cross det range bins:', det_indices_osCross[0])
print('CFAR OS cross det doppler bins:', det_indices_osCross[1],'\n')
t3 = time()
bool_array_ca = cfar_lib.CFAR_CA_2D(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate)
det_indices_ca = np.where(bool_array_ca>0)
print('CFAR CA det range bins:', det_indices_ca[0])
print('CFAR CA det doppler bins:', det_indices_ca[1],'\n')
t4 = time()

bool_array_caCross = cfar_lib.CFAR_CA_2D_cross(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate)
det_indices_caCross = np.where(bool_array_caCross>0)
print('CFAR CA cross det range bins:', det_indices_caCross[0])
print('CFAR CA cross det doppler bins:', det_indices_caCross[1],'\n')


signalExtend = np.zeros((signal_mag.shape[0] + 2*guardband_len_y + 2*valid_samp_len_y, signal_mag.shape[1] + 2*guardband_len_x + 2*valid_samp_len_x),dtype=np.float32)
signalExtend[guardband_len_y+valid_samp_len_y:signal_mag.shape[0]+guardband_len_y+valid_samp_len_y, \
             guardband_len_x+valid_samp_len_x:signal_mag.shape[1]+guardband_len_x+valid_samp_len_x] = signal_mag;
    
signalExtend[0:valid_samp_len_y+guardband_len_y, \
             guardband_len_x+valid_samp_len_x:signal_mag.shape[1]+guardband_len_x+valid_samp_len_x] = np.flipud(signal_mag[1:valid_samp_len_y+guardband_len_y+1,:]);
signalExtend[signal_mag.shape[0]+valid_samp_len_y+guardband_len_y::, \
             guardband_len_x+valid_samp_len_x:signal_mag.shape[1]+guardband_len_x+valid_samp_len_x] = np.flipud(signal_mag[-(valid_samp_len_y+guardband_len_y)::,:]);

signalExtend[guardband_len_y+valid_samp_len_y:signal_mag.shape[0]+guardband_len_y+valid_samp_len_y, \
             0:valid_samp_len_x+guardband_len_x] = np.fliplr(signal_mag[:,1:valid_samp_len_x+guardband_len_x+1]);
signalExtend[guardband_len_y+valid_samp_len_y:signal_mag.shape[0]+guardband_len_y+valid_samp_len_y, \
             signal_mag.shape[1]+valid_samp_len_x+guardband_len_x::] = np.fliplr(signal_mag[:,-(valid_samp_len_x+guardband_len_x)::]);

t5 = time()
signalExtend = cp.asarray(signalExtend.copy(),dtype=cp.float32)    
    
origSigShapeX = np.int32(signal_mag.shape[1]);
origSigShapeY = np.int32(signal_mag.shape[0]);
thrdsPerBlkx = 16;
thrdsPerBlky = 16;
thrdsPerBlk = (thrdsPerBlky,thrdsPerBlkx)
blksPerGridx = np.int32(np.ceil(origSigShapeX/thrdsPerBlkx));
blksPerGridy = np.int32(np.ceil(origSigShapeY/thrdsPerBlky));
blksPerGrid = (blksPerGridy,blksPerGridx)

valid_samp_num = 2*(valid_samp_len_x + valid_samp_len_y)
scratchPad = cp.zeros((signal_mag.shape[0],signal_mag.shape[1],valid_samp_num),dtype=cp.float32);


noiseMargin = cp.float32(valid_samp_num*(false_alarm_rate**(-1/valid_samp_num) -1))
outPutBoolArray_CAcross = cp.zeros((signal_mag.shape[0], signal_mag.shape[1]),dtype=cp.int32)

# GPU_memorySize = (signal_mag.nbytes + origSigShapeX.nbytes + origSigShapeY.nbytes + guardband_len_x.nbytes, guardband_len_y.nbytes + valid_samp_len_x.nbytes + valid_samp_len_y.nbytes + scratchPad.nbytes + noiseMargin.nbytes + outPutBoolArray_CAcross.nbytes)/(1024*1024);
GPU_memorySize = (signal_mag.nbytes + scratchPad.nbytes + outPutBoolArray_CAcross.nbytes)/(1024*1024);
print('Total memory on GPU = {0:.2f} MB\n'.format(GPU_memorySize));

CFAR_CA_2D_cross_GPU[blksPerGrid,thrdsPerBlk](signalExtend, origSigShapeX, origSigShapeY, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y,  scratchPad, noiseMargin, outPutBoolArray_CAcross)
outPutBoolArray_CAcross = cp.asnumpy(outPutBoolArray_CAcross)
det_indices_caCross_gpu = np.where(outPutBoolArray_CAcross>0)
t6 = time()

print('CFAR CA cross GPU det range bins:', det_indices_caCross_gpu[0])
print('CFAR CA cross GPU det doppler bins:', det_indices_caCross_gpu[1],'\n')
  
    
## Timings
print('CFAR CA cross CPU compute time = {0:.0f} ms'.format((t5-t4)*1000))
print('CFAR CA cross GPU compute time = {0:.0f} ms'.format((t6-t5)*1000))



plt.figure(1,figsize=(20,10))

plt.subplot(2,2,1)
plt.title('CFAR OS 2D')
plt.imshow(10*np.log10(signal_mag),aspect='auto')
plt.scatter(det_indices_os[1],det_indices_os[0],c='r',marker='*',s=20)
plt.ylabel('Range Index')
plt.xlabel('Doppler Index')


plt.subplot(2,2,2)
plt.title('CFAR OS 2D Cross')
plt.imshow(10*np.log10(signal_mag),aspect='auto')
plt.scatter(det_indices_osCross[1],det_indices_osCross[0],c='r',marker='*',s=20)
plt.ylabel('Range Index')
plt.xlabel('Doppler Index')


plt.subplot(2,2,3)
plt.title('CFAR CA 2D')
plt.imshow(10*np.log10(signal_mag),aspect='auto')
plt.scatter(det_indices_ca[1],det_indices_ca[0],c='r',marker='*',s=20)
plt.ylabel('Range Index')
plt.xlabel('Doppler Index')


plt.subplot(2,2,4)
plt.title('CFAR CA 2D cross')
plt.imshow(10*np.log10(signal_mag),aspect='auto');
plt.colorbar()
plt.scatter(det_indices_caCross[1],det_indices_caCross[0],c='r',marker='*',s=20,label='CPU')
plt.scatter(det_indices_caCross_gpu[1],det_indices_caCross_gpu[0],c='k',marker='D',s=30,alpha=0.3,label='GPU')
plt.ylabel('Range Index')
plt.xlabel('Doppler Index');
# plt.legend()



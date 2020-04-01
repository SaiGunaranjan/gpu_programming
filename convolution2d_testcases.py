# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:16:38 2020

@author: saiguna
"""

import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from time import time
import cupy as cp
import imageio
from numbaCudaKernels import conv_2d
from scipy import signal


plt.close('all')

imageFolderPath = 'test_images\\'
imageName = imageFolderPath + 'barbara.png' # 'CheckboardWithNoise.jpg' 
inputImage = imageio.imread(imageName)

if len(inputImage.shape)==2:
    inputImage = inputImage[:,:,None]
inputImageLenX = inputImage.shape[1]
inputImageLenY = inputImage.shape[0]
psfLenX = 5 # should always be an odd number
psfLenY = 5 # should also always be an odd number
psfOneSideLenX = int(np.floor(psfLenX//2));
psfOneSideLenY = int(np.floor(psfLenY//2));
pointSpreadFn = np.ones((psfLenY, psfLenX));
inputImageExtLenX = inputImageLenX + 2*psfLenX -2;
inputImageExtLenY = inputImageLenY + 2*psfLenY -2;

t1 = time();

inputImage_gpu = cp.asarray(inputImage.copy(),dtype=cp.float32);
pointSpreadFn_gpu = cp.asarray(pointSpreadFn.copy(),dtype=cp.float32);
inputImage_gpu_ext = cp.zeros((inputImageExtLenY, inputImageExtLenX, inputImage.shape[2]),dtype=cp.float32);
inputImage_gpu_ext[psfLenY-1:-(psfLenY-1),psfLenX-1:-(psfLenX-1),:] = inputImage_gpu;
outputImage = cp.zeros((inputImageLenY+psfLenY-1,inputImageLenX+psfLenX-1),dtype=cp.float32);

t2 = time();

GPU_memorySize = (inputImage_gpu_ext.nbytes + pointSpreadFn_gpu.nbytes + outputImage.nbytes)/(1024*1024);
print('Total memory on GPU = {0:.2f} MB'.format(GPU_memorySize));


thrdPerBlockx = 16;
thrdPerBlocky = 16;
thrdPerBlock = (thrdPerBlocky,thrdPerBlockx);
blkPerGridx = int(cp.ceil(inputImage_gpu_ext.shape[1]/thrdPerBlockx));
blkPerGridy = int(cp.ceil(inputImage_gpu_ext.shape[0]/thrdPerBlocky));
blkPerGrid = (blkPerGridy, blkPerGridx)


outputImage_ker = np.zeros((outputImage.shape[0],outputImage.shape[1],inputImage.shape[2]),dtype=np.float32)
t3 = time();
for ele in range(inputImage.shape[2]):
    conv_2d[blkPerGrid,thrdPerBlock](inputImage_gpu_ext[:,:,ele], pointSpreadFn_gpu, outputImage, inputImageLenX, inputImageLenY, psfOneSideLenX, psfOneSideLenY)
    outputImage_ker[:,:,ele] = cp.asnumpy(outputImage)


t4 = time();

outputImage_cpu = signal.convolve2d(inputImage.squeeze(),pointSpreadFn);

t5 = time();

print('Data transfer time = {0:.0f} ms'.format((t2-t1)*1000));
print('GPU compute + Fetch time = {0:.0f} ms'.format((t4-t3)*1000));
print('Total GPU transfer + compute + fetch time = {0:.0f} ms'.format((t4-t1)*1000))
print('CPU compute time = {0:.0f} ms'.format((t5-t4)*1000))

plt.figure(1,figsize=(20,10));
plt.subplot(131);
plt.title('original')
plt.imshow(inputImage.squeeze(),cmap='gray')
plt.subplot(132);
plt.title('CPU convolved')
plt.imshow(outputImage_cpu.squeeze(),cmap='gray')
plt.subplot(133);
plt.title('GPU convolved')
plt.imshow(outputImage_ker.squeeze(),cmap='gray')





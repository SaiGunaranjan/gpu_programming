# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:24:39 2020

@author: saiguna
"""

import sys
sys.path.append("..")
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import imageio
from time import time
from numbaCudaKernels import conv_2d
from scipy import signal


def prewittOperator(numSamples):
    
    a = np.hstack((-1*np.ones((int(np.floor(numSamples//2)))),0,1*np.ones((int(np.floor(numSamples//2))))));
    b = np.ones((numSamples,));
    kernel = b[:,None]*a[None,:]
    return kernel

def sobelOperator(numSamples):
    """
    

    Parameters
    ----------
    numSamples : TYPE: int
        DESCRIPTION.

    Returns
    -------
    kernel : TYPE: 2-D matrix of numSamples x numSamples
        DESCRIPTION.

    """
    
    a = np.hstack((-1*np.ones((int(np.floor(numSamples//2)))),0,1*np.ones((int(np.floor(numSamples//2))))));
    b = np.hstack((1*np.ones((int(np.floor(numSamples//2)))),2*int(np.floor(numSamples//2)),1*np.ones((int(np.floor(numSamples//2))))));
    kernel = b[:,None]*a[None,:]
    return kernel


plt.close('all')

imageFolderPath = '..\\test_images\\'
imageName = imageFolderPath + 'boat.png' #'1Orig.jpg' #'barbara.png' # 'CheckboardWithNoise.jpg' 
inputImage = imageio.imread(imageName)


if len(inputImage.shape)==2:
    inputImage = inputImage[:,:,None]
inputImageDtype = type(inputImage[0,0,0])
inputImageLenX = inputImage.shape[1]
inputImageLenY = inputImage.shape[0]

psfSamples_1d = 5
psfLenX = psfSamples_1d # should always be an odd number
psfLenY = psfSamples_1d # should also always be an odd number
psfOneSideLenX = int(np.floor(psfLenX//2));
psfOneSideLenY = int(np.floor(psfLenY//2));

# horzGradKernel = prewittOperator(psfSamples_1d)
# vertGradKernel = prewittOperator(psfSamples_1d).T

horzGradKernel = sobelOperator(psfSamples_1d)
vertGradKernel = sobelOperator(psfSamples_1d).T

inputImageExtLenX = inputImageLenX + 2*psfLenX -2;
inputImageExtLenY = inputImageLenY + 2*psfLenY -2;

t1 = time();

inputImage_gpu = cp.asarray(inputImage.copy(),dtype=cp.float32);
horzGradKernel_gpu = cp.asarray(horzGradKernel.copy(),dtype=cp.float32);
vertGradKernel_gpu = cp.asarray(vertGradKernel.copy(),dtype=cp.float32)
inputImage_gpu_ext = cp.zeros((inputImageExtLenY, inputImageExtLenX, inputImage.shape[2]),dtype=cp.float32);
inputImage_gpu_ext[psfLenY-1:-(psfLenY-1),psfLenX-1:-(psfLenX-1),:] = inputImage_gpu;
outputImage = cp.zeros((inputImageLenY+psfLenY-1,inputImageLenX+psfLenX-1),dtype=cp.float32);

t2 = time();

GPU_memorySize = (inputImage_gpu_ext.nbytes + horzGradKernel_gpu.nbytes + outputImage.nbytes)/(1024*1024);
print('Total memory on GPU = {0:.2f} MB'.format(GPU_memorySize));


thrdPerBlockx = 16;
thrdPerBlocky = 16;
thrdPerBlock = (thrdPerBlocky,thrdPerBlockx);
blkPerGridx = int(cp.ceil(inputImage_gpu_ext.shape[1]/thrdPerBlockx));
blkPerGridy = int(cp.ceil(inputImage_gpu_ext.shape[0]/thrdPerBlocky));
blkPerGrid = (blkPerGridy, blkPerGridx)




t3 = time();
image_horzGradient = cp.zeros((outputImage.shape[0],outputImage.shape[1],inputImage.shape[2]),dtype=np.float32)
for ele in range(inputImage.shape[2]):
    conv_2d[blkPerGrid,thrdPerBlock](inputImage_gpu_ext[:,:,ele], horzGradKernel_gpu, outputImage, inputImageLenX, inputImageLenY, psfOneSideLenX, psfOneSideLenY)
    image_horzGradient[:,:,ele] = outputImage
image_horzGradient = cp.asnumpy(image_horzGradient);

image_vertGradient = cp.zeros((outputImage.shape[0],outputImage.shape[1],inputImage.shape[2]),dtype=np.float32)    
for ele in range(inputImage.shape[2]):
    conv_2d[blkPerGrid,thrdPerBlock](inputImage_gpu_ext[:,:,ele], vertGradKernel_gpu, outputImage, inputImageLenX, inputImageLenY, psfOneSideLenX, psfOneSideLenY)
    image_vertGradient[:,:,ele] = outputImage
image_vertGradient = cp.asnumpy(image_vertGradient)

t4 = time();
image_horzGradient_cpu = np.zeros((outputImage.shape[0],outputImage.shape[1],inputImage.shape[2]),dtype=np.float32)
for ele in range(inputImage.shape[2]):
    image_horzGradient_cpu[:,:,ele] = signal.convolve2d(inputImage[:,:,ele],horzGradKernel);
    
image_vertGradient_cpu = np.zeros((outputImage.shape[0],outputImage.shape[1],inputImage.shape[2]),dtype=np.float32)
for ele in range(inputImage.shape[2]):
    image_vertGradient_cpu[:,:,ele] = signal.convolve2d(inputImage[:,:,ele],vertGradKernel);

t5 = time();


gradientMag_gpu = np.sqrt(image_horzGradient**2 + image_vertGradient**2);
gradientMag_cpu = np.sqrt(image_horzGradient_cpu**2 + image_vertGradient_cpu**2);

numAngles = 100
phi= np.arange(-np.pi,np.pi,2*np.pi/numAngles);
cosVec = np.cos(phi);
sinVec = np.sin(phi);

gradientImageMatrix_gpu = image_horzGradient[:,:,:,None]*cosVec[None,None,None,:] + image_vertGradient[:,:,:,None]*sinVec[None,None,None,:]
edgeDetImage_gpu = np.amax(np.abs(gradientImageMatrix_gpu),axis=3);

gradientImageMatrix_cpu = image_horzGradient_cpu[:,:,:,None]*cosVec[None,None,None,:] + image_vertGradient_cpu[:,:,:,None]*sinVec[None,None,None,:]
edgeDetImage_cpu = np.amax(np.abs(gradientImageMatrix_cpu),axis=3);


"""Normalize to 255 (to convert to 8 bit int in next step"""

image_horzGradient = (image_horzGradient/np.amax(image_horzGradient,axis=(0,1))[None,None,:])*255
image_vertGradient = (image_vertGradient/np.amax(image_vertGradient,axis=(0,1))[None,None,:])*255
gradientMag_gpu = (gradientMag_gpu/np.amax(gradientMag_gpu,axis=(0,1))[None,None,:])*255
outputImage_ker = (edgeDetImage_gpu/np.amax(edgeDetImage_gpu,axis=(0,1))[None,None,:])*255

gradientMag_cpu = (gradientMag_cpu/np.amax(gradientMag_cpu,axis=(0,1))[None,None,:])*255
image_horzGradient_cpu = (image_horzGradient_cpu/np.amax(image_horzGradient_cpu,axis=(0,1))[None,None,:])*255
image_vertGradient_cpu = (image_vertGradient_cpu/np.amax(image_vertGradient_cpu,axis=(0,1))[None,None,:])*255
outputImage_cpu = (edgeDetImage_cpu/np.amax(edgeDetImage_cpu,axis=(0,1))[None,None,:])*255


""" convert to unsigned 8 bit int after havng normalized all the values to 2^8 -1"""
image_horzGradient_uint8 = np.uint8(image_horzGradient).squeeze()
image_vertGradient_uint8 = np.uint8(image_vertGradient).squeeze()
gradientMag_gpu_uint8 = np.uint8(gradientMag_gpu).squeeze()
outputImage_ker_uint8 = np.uint8(outputImage_ker).squeeze()

image_horzGradient_cpu_uint8 = np.uint8(image_horzGradient_cpu).squeeze()
image_vertGradient_cpu_uint8 = np.uint8(image_vertGradient_cpu).squeeze()
gradientMag_cpu_uint8 = np.uint8(gradientMag_cpu).squeeze()
outputImage_cpu_uint8 = np.uint8(outputImage_cpu).squeeze()



print('Data transfer time = {0:.0f} ms'.format((t2-t1)*1000));
print('GPU compute + Fetch time = {0:.0f} ms'.format((t4-t3)*1000));
print('Total GPU transfer + compute + fetch time = {0:.0f} ms'.format((t4-t1)*1000))
print('CPU compute time = {0:.0f} ms'.format((t5-t4)*1000))




plt.figure(1,figsize=(20,10))
plt.subplot(251)
plt.title('Original image')
plt.imshow(inputImage.squeeze(),cmap='gray')
plt.subplot(252)
plt.title('Horizontal gradient: CPU')
plt.imshow(image_horzGradient_cpu_uint8,cmap='gray')
plt.subplot(253)
plt.title('Vertical gradient: CPU')
plt.imshow(image_vertGradient_cpu_uint8,cmap='gray')
plt.subplot(254)
plt.title('Gradient Magnitude: CPU')
plt.imshow(gradientMag_cpu_uint8,cmap='gray')
plt.subplot(255)
plt.title('Edge detected image: CPU')
plt.imshow(outputImage_cpu_uint8,cmap='gray')

plt.subplot(256)
plt.title('Original image')
plt.imshow(inputImage.squeeze(),cmap='gray')
plt.subplot(257)
plt.title('Horizontal gradient: GPU')
plt.imshow(image_horzGradient_uint8,cmap='gray')
plt.subplot(258)
plt.title('Vertical gradient: GPU')
plt.imshow(image_vertGradient_uint8,cmap='gray')
plt.subplot(259)
plt.title('Gradient Magnitude: GPU')
plt.imshow(gradientMag_gpu_uint8,cmap='gray')
plt.subplot(2,5,10)
plt.title('Edge detected image: GPU')
plt.imshow(outputImage_ker_uint8,cmap='gray')
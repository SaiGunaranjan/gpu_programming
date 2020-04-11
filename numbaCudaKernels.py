# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:22:01 2020

@author: Sai Gunaranjan Pelluri
"""

from numba import cuda
import cupy as cp

@cuda.jit('void(float32[:],float32[:],float32[:],int32,int32)')
def conv_1d(inp_signal_ext,impulse_response,output_signal,signal_len,impulse_response_len):
    
    thrdID = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if thrdID >= signal_len+impulse_response_len-1:
        return;
        
    temp_sum = 0;
    for ele in range(impulse_response_len):
        temp_sum += inp_signal_ext[thrdID+ele]*impulse_response[impulse_response_len-ele-1]
    output_signal[thrdID] = temp_sum
    



@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:], int32, int32, int32, int32)')
def conv_2d(inputImageExtnd, pointSpreadFn, outputImage, InputLenX, InputLenY, psfOneSideLenX, psfOneSideLenY):
    
    thrdIDx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    thrdIDy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    psfLenX = 2*psfOneSideLenX + 1
    psfLenY = 2*psfOneSideLenY + 1
    
    if (thrdIDx >= psfOneSideLenX + InputLenX + psfLenX -1) or (thrdIDx < psfOneSideLenX)  or (thrdIDy >= psfOneSideLenY + InputLenY + psfLenY -1) or (thrdIDy < psfOneSideLenY):
        return;
    
    convSum = cp.float32(0)
    for x in range(-psfOneSideLenX,psfOneSideLenX+1):
        for y in range(-psfOneSideLenY,psfOneSideLenY+1):
            convSum += inputImageExtnd[thrdIDy+y,thrdIDx+x]*pointSpreadFn[psfOneSideLenY-y,psfOneSideLenX-x];
            # if (thrdIDx == psfOneSideLenX + InputLenX + psfLenX -2) and (thrdIDy == psfOneSideLenY + InputLenY + psfLenY -2):
            #     print('x:',x,'y:',y,'imagVal:',inputImageExtnd[thrdIDy+y,thrdIDx+x],'psfval:',pointSpreadFn[2*psfOneSideLenY-y,2*psfOneSideLenX-x])
   
    # if (thrdIDx == psfOneSideLenX + InputLenX + psfLenX -2) and (thrdIDy == psfOneSideLenY + InputLenY + psfLenY -2):
    #     print('conSum:',convSum)
    outputImage[thrdIDy-psfOneSideLenY,thrdIDx-psfOneSideLenX] = convSum;
    
    

@cuda.jit('void(float32[:], int32, int32, int32, float32[:,:], float32, float32[:])')
def CFAR_CA_GPU(signal_ext, origSignalLen , guardBandLen_1side, validSampLen_1side, scratchPad, noiseMargin, outputBoolVector):
    
    thrdID = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    
    if (thrdID < origSignalLen-1) or (thrdID > 2*origSignalLen-2):
        return;
            
    # check for local maxima on the CUT i.e. signal_ext[thrdID]
    if (signal_ext[thrdID] >= signal_ext[thrdID-1]) and (signal_ext[thrdID] >= signal_ext[thrdID+1]):

        count = cp.int32(0)
        for i in range(thrdID-guardBandLen_1side-validSampLen_1side, thrdID-guardBandLen_1side):
            # scratchPad[count] = signal_ext[i]; # This should not be done. There should be a separate scratch pad for each thread when it is vector/matrix copying
            scratchPad[thrdID-(origSignalLen-1),count] = signal_ext[i]
            count += 1;
        
        for j in range(thrdID+guardBandLen_1side+1, thrdID+guardBandLen_1side+validSampLen_1side+1):
            # scratchPad[count] = signal_ext[j]; # This should not be done. There should be a separate scratch pad for each thread when it is vector/matrix copying
            scratchPad[thrdID-(origSignalLen-1),count] = signal_ext[j];
            count += 1
        avgNoisePower = cp.float32(0)
        for ele in range(2*validSampLen_1side):
            avgNoisePower += scratchPad[thrdID-(origSignalLen-1),ele];
        avgNoisePower = avgNoisePower/(2*validSampLen_1side)
                
        if (signal_ext[thrdID] > noiseMargin*avgNoisePower):
            outputBoolVector[thrdID-(origSignalLen-1)] = 1
        
        

@cuda.jit('void(float32[:], int32, int32, int32, float32[:,:], float32, int32, float32[:])')
def CFAR_OS_GPU(signal_ext, origSignalLen , guardBandLen_1side, validSampLen_1side, scratchPad, noiseMargin, ordStat, outputBoolVector):
    
    thrdID = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    
    if (thrdID < origSignalLen-1) or (thrdID > 2*origSignalLen-2):
        return;
    
    # check for local maxima on the CUT i.e. signal_ext[thrdID]
    if (signal_ext[thrdID] >= signal_ext[thrdID-1]) and (signal_ext[thrdID] >= signal_ext[thrdID+1]):

        count = cp.int32(0)
        for i in range(thrdID-guardBandLen_1side-validSampLen_1side, thrdID-guardBandLen_1side):
            scratchPad[thrdID-(origSignalLen-1),count] = signal_ext[i]
            count += 1;
        
        for j in range(thrdID+guardBandLen_1side+1, thrdID+guardBandLen_1side+validSampLen_1side+1):
            scratchPad[thrdID-(origSignalLen-1),count] = signal_ext[j];
            count += 1
        
        temp = cp.float32(0);
        ordStat_largestVal = cp.float32(0)
        # sort in decreasing order of strength upto the ordStat kth largest value
        for i in range(ordStat):
            for j in range(i+1,2*validSampLen_1side):
                if (scratchPad[thrdID-(origSignalLen-1),i] < scratchPad[thrdID-(origSignalLen-1),j]):
                        temp = scratchPad[thrdID-(origSignalLen-1),i];
                        scratchPad[thrdID-(origSignalLen-1),i] = scratchPad[thrdID-(origSignalLen-1),j];
                        scratchPad[thrdID-(origSignalLen-1),j] = temp
        
        ordStat_largestVal = scratchPad[thrdID-(origSignalLen-1),ordStat-1]
        
        if (signal_ext[thrdID] > noiseMargin*ordStat_largestVal):
            outputBoolVector[thrdID-(origSignalLen-1)] = 1
        
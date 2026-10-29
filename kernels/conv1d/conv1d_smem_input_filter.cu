/*
conv1d, one thread per output. Dynamic shared memory (extern __shared__, sized at launch) holds the
block's input tile and the filter, so the compute loop reads only shared memory.

Build and run: see kernels/README.md. To confirm that CPU/GPU differences are only FMA rounding,
build with -fmad=false (diagnosis only).
*/


#include <cuda_runtime.h>
#include <vector>
using namespace std;
#include <iostream>
#include <random>


__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size, int num_ip_el_thrdblock) {

                                        int global_tid = (blockIdx.x * blockDim.x) + threadIdx.x;
                                        int local_tid = threadIdx.x;

                                        // return conditioan take care of in loads and compute itself so no explicit return needed.

                                        // extern - tell compiler that this memory has already been allocated at some other point in the code
                                        // char so that the memory pool can be partitioned into single bytes. Helpful when two different datatype need to be put in shared memory
                                        extern __shared__ char smem_bytes[];

                                        // Cast the smem_bytes to float dtype and initialize a new pointer variable pointing to float
                                        float *smem_input_per_thrdblock = reinterpret_cast<float*>(smem_bytes);

                                        // Offset the smem_input_per_thrdblock pointer by num_input_elements_per_threadblock, get the new address which again points to float and initialize another pointer to this new address
                                        float *smem_kernel = reinterpret_cast<float*>(smem_input_per_thrdblock + num_ip_el_thrdblock); 

                                        // Load the input elements required per thread block
                                        for (int i=0; i<num_ip_el_thrdblock; i+=blockDim.x)
                                        {
                                            if ((global_tid+i < input_size) && (local_tid+i < num_ip_el_thrdblock))
                                            {
                                                smem_input_per_thrdblock[local_tid+i] = input[global_tid+i];
                                            }
                                        }

                                        // __synchthreads() not required here since the data load and kernel load are independent operations and write to different parts of the shared memory
                                        

                                        // Load the kernel also to shared memory
                                        for (int j=0; j<kernel_size; j+=blockDim.x)
                                        {
                                            if (local_tid+j < kernel_size)
                                            {
                                                smem_kernel[local_tid+j] = kernel[local_tid+j];
                                            }
                                        }

                                        __syncthreads();

                                        float acc = 0.0f;
                                        for (int k=0; k<kernel_size; k++)
                                        {
                                            acc += smem_input_per_thrdblock[local_tid+k]*smem_kernel[k];
                                        }

                                        if (global_tid < input_size -kernel_size + 1)
                                        {
                                            output[global_tid] = acc;
                                        }

                                      }



// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    int num_ip_el_thrdblock = threadsPerBlock + kernel_size - 1;
    size_t smem_bytes = num_ip_el_thrdblock*sizeof(float) + kernel_size*sizeof(float);


    /*Pass smem size in bytes at kernel launch (or runtime)*/                  
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, smem_bytes>>>(input, kernel, output, input_size,
                                                              kernel_size, num_ip_el_thrdblock);
    
    cudaDeviceSynchronize();
}


int main()
{
    int input_size = 300;
    int kernel_size = 5;
    int output_size = input_size - kernel_size + 1;

    random_device rd; // Obtains a random seed from hardware
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f,1.0f);

    vector<float> input_h(input_size);
    vector<float> kernel_h(kernel_size);

    for (int i = 0; i < input_size; i++)
    {
        input_h[i] = dis(gen);
    }

    for (int i = 0; i < kernel_size; i++)
    {
        kernel_h[i] = dis(gen);
    }

    vector<float> output_cpu(output_size,0.0f);
    vector<float> output_gpu(output_size,0.0f);

    float *input_d;
    float *kernel_d;
    float *output_d;

    size_t size_input = input_size * sizeof(float);
    size_t size_kernel = kernel_size * sizeof(float);
    size_t size_output = output_size * sizeof(float);

    cudaMalloc(&input_d,size_input);
    cudaMalloc(&kernel_d,size_kernel);
    cudaMalloc(&output_d,size_output);

    cudaMemcpy(input_d,input_h.data(),size_input,cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d,kernel_h.data(),size_kernel,cudaMemcpyHostToDevice);

    solve(input_d, kernel_d, output_d, input_size,
                      kernel_size);
    
    cudaMemcpy(output_gpu.data(),output_d,size_output,cudaMemcpyDeviceToHost);

    for (int i=0; i<output_size; i++)
    {
        float acc = 0.0f;
        for (int j = 0; j < kernel_size; j++)
        {
            acc += input_h[i+j]*kernel_h[j];
        }
        output_cpu[i] = acc;
    }
    cout << "\n";
    // for (int i=output_size-10; i<output_size; i++)
    // {
    //     cout << "output_cpu[" << i << "] = " << output_cpu[i] << "\n";
    //     cout << "output_gpu[" << i << "] = " << output_gpu[i] << "\n";
    // }
    
    for (int i=0; i<output_size; i++)
    {
        if (output_cpu[i] != output_gpu[i])
        {
            cout << "output_cpu[" << i << "] = " << output_cpu[i] << ", output_gpu[" << i << "] = " << output_gpu[i] << "\n";
        }
    }

    // if (output_cpu == output_gpu)
    // {
    //     cout << "CPU and GPU outputs are equal\n";
    // }
    // else
    // {
    //     cout << "CPU and GPU outputs are NOT equal\n";
    // }


}
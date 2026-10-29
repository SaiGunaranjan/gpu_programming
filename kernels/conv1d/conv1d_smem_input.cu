/*
conv1d, one thread per output. The block's input tile (BLOCK_SIZE + kernel_size - 1 elements) is
staged in static shared memory, sized for the largest filter (MAX_KERNEL_SIZE); the filter is read
from global memory.

Build and run: see kernels/README.md. To confirm that CPU/GPU differences are only FMA rounding,
build with -fmad=false (diagnosis only).
*/

#include <cuda_runtime.h>
#include <vector>
using namespace std;
#include <iostream>
#include <random>

#define MAX_KERNEL_SIZE 2047
#define BLOCK_SIZE 256
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {

                                        int global_tid = (blockIdx.x * blockDim.x) + threadIdx.x;
                                        int local_tid = threadIdx.x;

                                        if (global_tid > (input_size)) //(input_size - kernel_size)
                                        {
                                            return;
                                        }
                                        // int num_elements_max = blockDim.x + MAX_KERNEL_SIZE - 1;
                                        const int num_elements_max = BLOCK_SIZE + MAX_KERNEL_SIZE - 1;
                                        __shared__ float shared_mem_array[num_elements_max];

                                        for (int i=0; i<num_elements_max; i+=blockDim.x)
                                        {   
                                            if ((global_tid+i < input_size) && (local_tid+i < num_elements_max))
                                            {
                                                shared_mem_array[local_tid+i] = input[global_tid+i];
                                                // if (blockIdx.x == 0 && threadIdx.x == 3)
                                                // {
                                                //     printf("\n Block %d local thread %d: shared_mem_array[%d] = %f",
                                                //     blockIdx.x, threadIdx.x, local_tid+i, shared_mem_array[local_tid+i]);
                                                // }
                                            }
                                            
                                        }

                                        __syncthreads();

                                        float acc = 0.0f;

                                        for (int j=0; j<kernel_size; j++)
                                        {   
                                            // Move the kernel also to shared memory of every block. Right now, each thread is fetching the kernel from DRAM!
                                            acc += shared_mem_array[local_tid+j]*kernel[j];
                                            // if (blockIdx.x == 0 && threadIdx.x == 1)
                                            // {
                                            //     printf("\n Block %d local thread %d: shared_mem_array[%d] = %f",
                                            //     blockIdx.x, threadIdx.x, local_tid+j, shared_mem_array[local_tid+j]);
                                            // }
                                        }
                                        if (global_tid < input_size - kernel_size + 1)
                                        {
                                            output[global_tid] = acc;
                                        }
                                        
                                      }



// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    // int threadsPerBlock = 256;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    
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

/* 
No, it isn't. `MAX_KERNEL_SIZE` is fixed at compile time, but `blockDim.x` is not.

## Why the compiler rejects it

Look at what `num_elements_max` is built from on line 15:

```cpp
int num_elements_max = blockDim.x + MAX_KERNEL_SIZE - 1;
```

- `MAX_KERNEL_SIZE` is a `#define`. The preprocessor pastes in `2047` before compiling, so that part is a constant.
- `blockDim.x` is a built-in variable that gets its value when the kernel is **launched**. Your host code passes 256 in `<<<blocksPerGrid, threadsPerBlock>>>`, but the kernel is compiled before that launch exists. The same compiled kernel could be launched with 128, 512 or 1024 threads, so nvcc can't know the value.

A runtime value plus a constant is still a runtime value. Line 16 therefore asks for an array whose size is only known at runtime, which is a variable length array (VLA). `__shared__` memory is allocated per block, and its size has to be known either at compile time or at launch. A VLA gives neither, so you get "cannot have static storage duration."

(A side point: even when everything in an expression is constant, a plain `int` variable doesn't count as a compile-time constant here. You'd need `constexpr`, or a `const` variable initialised from constants.)

## Two ways to fix it (you choose)

1. **Make the block size a compile-time constant.** Define the threads-per-block value once as a constant, and use it both for the shared array size in the kernel and for the launch in `solve`. Then everything in the array size is known at compile time.
2. **Use dynamic shared memory.** Look up `extern __shared__` and the **third** parameter of the `<<< >>>` launch syntax. With this, the host decides the shared memory size in bytes at launch time. You could even size it from the real `kernel_size` instead of `MAX_KERNEL_SIZE`.

Option 1 is simpler for a first kernel. Option 2 is worth learning because it shows up everywhere.

## Bugs to look at once it compiles

You said you know there are some issues, so here are questions rather than fixes:

1. **The early `return` on line 11 comes before `__syncthreads()` on line 23.** What happens at a barrier if some threads in the block have already exited? And those threads also skip the loading loop, so what happens to the parts of the tile they were supposed to load?
2. **The load on line 20, `input[global_tid + i]`.** `i` starts at `local_tid`, so `global_tid + i` counts the thread index twice. Which global index should shared slot `i` hold? Think about where *this block's* chunk of `input` starts.
3. **Bounds on that load.** For the last block, does `global_tid + i` ever go past `input_size`? Nothing guards against it.
4. **The condition on line 11.** Is `>` correct given `output_size = input_size - kernel_size + 1`? Check the last valid output index by hand.
5. **Tile size vs. `kernel_size`.** You load `blockDim.x + MAX_KERNEL_SIZE - 1` elements, but each block only needs `blockDim.x + kernel_size - 1`. That isn't wrong, but it does wasted loads, and it connects to option 2 above.


*/
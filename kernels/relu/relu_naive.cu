
#include <cuda_runtime.h>
#include <vector>
using namespace std;
#include <cstdio>
#include <random>

template <int BLOCK_SIZE>
__global__ void relu_kernel(const float* input, float* output, int N)
{   

    int tid_local = threadIdx.x;
    int block_start_idx = blockIdx.x * BLOCK_SIZE;

    for (int i = tid_local; i < BLOCK_SIZE; i+=blockDim.x)
    {
        int tid_global = block_start_idx + i;
        if (tid_global < N)
        {
            // Will this cause warp divergence?
            if (input[tid_global] < 0) output[tid_global] = 0.0f;
            else output[tid_global] = input[tid_global];
        }
        
    }

    
    
}

extern "C" void solve(const float* input, float* output, int N)
{
    const int BLOCK_SIZE = 1024;
    // dim3 threadsPerBlock(256);
    // dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    relu_kernel<BLOCK_SIZE><<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

    // cudaDeviceSynchronize(); // Will do the sunchronize adter solve is called
}

int main()
{

    int N = 25'000'000;
    size_t bytes = N * sizeof(float);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1000.0f, 1000.0f);

    vector<float> input_h(N);
    vector<float> output_h(N);
    vector<float> output_from_device(N);

    for (int i = 0; i < N; i++) 
    {
        input_h[i] = dis(gen);
        if (input_h[i] < 0) output_h[i] = 0;
        else output_h[i] = input_h[i];
    }

    // Define pointers for device array
    float* input_d;
    float* output_d;

    // Allocate memory on device and return pointer to the address and store in address of device pointers
    cudaMalloc(&input_d, bytes);
    cudaMalloc(&output_d, bytes);

    cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice);
    // Populate every output byte with FF (max value), so float (4 bytes) becomes 0xFFFFFFFF
    cudaMemset(output_d, 0xFF, bytes); // NaN pattern: elements the kernel skips stay NaN

    solve(input_d, output_d, N);

    cudaError_t launch_err = cudaGetLastError();
    cudaError_t run_err = cudaDeviceSynchronize();

    if (launch_err != cudaSuccess || run_err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(launch_err != cudaSuccess ? launch_err : run_err));
        return 1;
    }


    cudaMemcpy(output_from_device.data(), output_d, bytes, cudaMemcpyDeviceToHost);

    // Count mismatches; print only the first few so a broken kernel doesn't print millions of lines
    long long mismatches = 0;
    const int max_print = 5;
    for (int i=0; i<N; i++)
    {
 
        if (output_h[i] != output_from_device[i])
        {
            if (mismatches < max_print)
            {
                printf("Mismatch at output[%d]: expected %f, got %f\n",
                        i, output_h[i], output_from_device[i]);
            }
            mismatches++;
        }

    }
    if (mismatches == 0) printf("ReLU correct\n");
    else                 printf("ReLU incorrect: %lld of %lld elements differ\n", mismatches, (long long)N);



    cudaFree(input_d);
    cudaFree(output_d);





}
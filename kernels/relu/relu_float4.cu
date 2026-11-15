
/*
For vectorlozed loads/stoeres in this problem, the BLOCK_SIZE and N should be a multiple of 4, else we need to do tail handling

KNOWN ISSUE (remove this note once fixed): the loop counts floats (i < BLOCK_SIZE), but each
iteration handles a float4 (idx4 = tid_global/4). So 4 consecutive iterations map to the same
float4, and every float4 is loaded and stored 4 times (25M float4 accesses instead of 6.25M).
DRAM traffic is unchanged, so it still runs at copy speed. Fix: loop in float4 units
(i < BLOCK_SIZE/4) with the start index and bounds check in float4 units, plus a scalar tail
for N % 4 != 0.
*/

#include <cuda_runtime.h>
#include <vector>
using namespace std;
#include <cstdio>
#include <random>

template <int BLOCK_SIZE>
__global__ void relu_kernel(const float* input, float* output, int N)
{   
    // Same address as input, viewed as float4: input4[k] = floats 4k..4k+3 (each index steps 16 B)
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);


    int tid_local = threadIdx.x;
    int block_start_idx = blockIdx.x * BLOCK_SIZE;

    
    for (int i = tid_local; i < BLOCK_SIZE; i += blockDim.x)
    {
        int tid_global = block_start_idx + i;
        if (tid_global < N) // Need to handle tails when N or BLOCK_SIZE is not a multiple of 4
        {   
            const int idx4 = tid_global/4;
            const float4 vals4_input = input4[idx4];
            float4 vals4_output;
            // Will this cause warp divergence?
            if (vals4_input.x < 0) vals4_output.x = 0.0f;
            else vals4_output.x = vals4_input.x;
            if (vals4_input.y < 0) vals4_output.y = 0.0f;
            else vals4_output.y = vals4_input.y;
            if (vals4_input.z < 0) vals4_output.z = 0.0f;
            else vals4_output.z = vals4_input.z;
            if (vals4_input.w < 0) vals4_output.w = 0.0f;
            else vals4_output.w = vals4_input.w;

            output4[idx4] = vals4_output;
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
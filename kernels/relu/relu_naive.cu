
#include <cuda_runtime.h>
#include <vector>
using namespace std;
#include <cstdio>
#include <random>
#include <algorithm>


struct Timing {
    float median_ms;
    float min_ms;
    float mean_ms;
};

// Times fn() on the GPU with cudaEvents.
// - `warmup` untimed calls first (JIT, clocks, caches), then `reps` timed calls.
// - One event between consecutive calls gives a per-call time for each rep -> median and min.
// - mean = (first event to last event) / reps.
// Note: this times everything fn() does, including any host-side cudaDeviceSynchronize() inside it.
template <typename F>
Timing time_gpu(F fn, int warmup, int reps)
{
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();

    vector<cudaEvent_t> ev(reps+1);
    for (auto& e: ev) cudaEventCreate(&e);

    cudaEventRecord(ev[0]);
    for (int r = 0; r < reps; r++)
    {
        fn();
        cudaEventRecord(ev[r+1]);
    }

    cudaEventSynchronize(ev[reps]);

    vector<float> ms(reps);
    for (int r = 0; r < reps; r++) cudaEventElapsedTime(&ms[r], ev[r], ev[r+1]);

    float total = 0.0f;
    cudaEventElapsedTime(&total, ev[0], ev[reps]);

    for (auto& e: ev) cudaEventDestroy(e);

    sort(ms.begin(), ms.end());
    return {ms[reps / 2], ms[0], total/reps};
}

// Bytes moved / seconds, in GB/s (decimal GB, like NVIDIA's spec sheets).
double gb_per_s(double bytes, float ms) { return bytes / (ms * 1e-3) / 1e9; }

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

    // --- Timing ---
    const int warmup = 20, reps = 100;

    // Practical peak: device-to-device copy. Reads `bytes`, writes `bytes`.
    Timing copy = time_gpu([&] { cudaMemcpy(output_d, input_d, bytes, cudaMemcpyDeviceToDevice);},
                    warmup, reps);

    // Reads input, writes output: the same bytes as the copy
    Timing kern = time_gpu([&] {solve(input_d, output_d, N);}, warmup, reps);

    // Check for errors from the timed launches too (120 more kernel launches)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    double copy_bytes = 2.0 * bytes, kern_bytes = 2.0 * bytes;
    double peak_spec = 1008; // RTX 4090: 10501 MHz x 2 (GDDR6X) x 48 B bus. Update when the pod's GPU changes
    double copy_bw = gb_per_s(copy_bytes, copy.median_ms);
    double kern_bw = gb_per_s(kern_bytes, kern.median_ms);

    printf("\nN = %d, warm-up %d, reps %d (times per call: median / min / mean)\n", N, warmup, reps);
    printf("D2D copy  : %8.1f / %8.1f / %8.1f us  %6.1f GB/s  (%5.1f%% of spec %.0f)\n",
           copy.median_ms * 1e3, copy.min_ms * 1e3, copy.mean_ms * 1e3, copy_bw,
           100 * copy_bw / peak_spec, peak_spec);
    printf("reluNaive : %8.1f / %8.1f / %8.1f us  %6.1f GB/s  (%5.1f%% of spec, %5.1f%% of D2D copy)\n",
           kern.median_ms * 1e3, kern.min_ms * 1e3, kern.mean_ms * 1e3, kern_bw,
           100 * kern_bw / peak_spec, 100 * kern_bw / copy_bw);




    cudaFree(input_d);
    cudaFree(output_d);





}

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
#include <cmath>
// #include <numeric> // Required for std::accumulate

template <int BLOCK_SIZE, int NUM_WARPS>
__global__ void reduction_kernel(const float* input, float* output, int N) 
{
    int tid = threadIdx.x;
    int block_start = blockIdx.x*BLOCK_SIZE;

    __shared__ float warp_partialsum[NUM_WARPS];
    float acc = 0.0f;

    for (int i = tid; i < BLOCK_SIZE; i += blockDim.x)
    {
        if ((block_start + i) < N)
        {
            acc += input[block_start + i];
        } 
        
    }

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int stride = 16; // 32/2
    
    while (stride > 0)
    {
        acc += __shfl_down_sync(0xffffffff, acc, stride);
        stride /= 2;
    }

    // Copy the sum of values from each warp (residing in lane_id = 0) to the shared memory array for one final __shfl_down_sync reduction
    if (lane_id == 0) warp_partialsum[warp_id] = acc;
    __syncthreads(); // Ensures all the lane_id = 0, of all the warps have written their reduced values to the smem buffer

    // Perform a final shuffle reduction using only warp 0
    if (warp_id == 0)
    {
        int new_stride = 16; // Will be using all the 32 threads but populate the remaining to 0.
        // Note: works for any NUM_WARPS <= 32 (always true: max 1024 threads per block). Needs threadsPerBlock % 32 == 0,
        // since the first-stage shuffle uses a full mask and every warp must have all 32 lanes.

        bool active = lane_id < NUM_WARPS;
        // Define a new register value 'val' which uses the smem values for the valid lanes and uses 0 for the remaining lanes. 
        float val = active ? warp_partialsum[lane_id]: 0.0f;

        while (new_stride > 0)
        {
            val += __shfl_down_sync(0xffffffff, val, new_stride);
            new_stride /= 2;
        }
            
        if (lane_id == 0) atomicAdd(output, val);
        
    }

    

}

extern "C" void solve(const float* input, float* output, int N)
{
    
    cudaMemset(output, 0x00, sizeof(float)); // Set sum value to 0 initially

    const int threadsPerBlock = 128;
    const int NUM_WARPS = (threadsPerBlock + 32 - 1) / 32;
    const int BLOCK_SIZE = 1024;
    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduction_kernel<BLOCK_SIZE, NUM_WARPS><<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

    // cudaDeviceSynchronize(); // Will do the sunchronize adter solve is called
}





int main()
{

    int N = 100'000'000; // Performance with N = 4'194'304, max N = 100'000'000;
    size_t bytes = N * sizeof(float);
    size_t output_bytes = sizeof(float);

    random_device rd;
    mt19937 gen(42); // mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1000.0f, 1000.0f);

    vector<float> input_h(N);
    double output_h = 0.0; // Making this double, so that rounding error during accumulation is reduced. fp32 accumulation + rounding has more errors compared to fp64
    float output_from_device;

    for (int i = 0; i < N; i++) 
    {
        input_h[i] = dis(gen);
        output_h += input_h[i]; // float is converted to double, then added in double
    }

    // Define pointers for device array
    float* input_d;
    float* output_d;

    // Allocate memory on device and return pointer to the address and store in address of device pointers
    cudaMalloc(&input_d, bytes);
    cudaMalloc(&output_d, output_bytes);

    cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice);

    solve(input_d, output_d, N);

    cudaError_t launch_err = cudaGetLastError();
    cudaError_t run_err = cudaDeviceSynchronize();

    if (launch_err != cudaSuccess || run_err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(launch_err != cudaSuccess ? launch_err : run_err));
        return 1;
    }


    // cudaMemcpy(output_from_device.data(), output_d, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&output_from_device, output_d, output_bytes, cudaMemcpyDeviceToHost);

    double atol = 1e-5, rtol = 1e-5; // leetgpu uses atol = rtol = 1e-5
    double abs_err = fabs((double)output_from_device - output_h);
    double tol = atol + rtol * fabs(output_h);


    if (abs_err > tol)
    {
        
        printf("Reduction Wrong! ref %.6f, got %.6f, abs err %.3e > tol %.3e\n",
           output_h, output_from_device, abs_err, tol);
    }
    else
    {
        printf("Reduction correct. ref %.6f, got %.6f, abs err %.3e (tol %.3e)\n",
           output_h, output_from_device, abs_err, tol);
    }



    // --- Timing ---
    const int warmup = 20, reps = 100;

    // Practical peak: device-to-device copy. Reads `bytes`, writes `bytes`.
    // output_d holds only one float, so the copy gets its own destination buffer.
    float* copy_d;
    cudaMalloc(&copy_d, bytes);
    Timing copy = time_gpu([&] { cudaMemcpy(copy_d, input_d, bytes, cudaMemcpyDeviceToDevice);},
                    warmup, reps);

    // Reads input once; writes a single float (4 bytes, negligible)
    Timing kern = time_gpu([&] {solve(input_d, output_d, N);}, warmup, reps);

    // Check for errors from the timed launches too (120 more kernel launches)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    double copy_bytes = 2.0 * bytes, kern_bytes = 1.0 * bytes;
    double peak_spec = 1008; // RTX 4090: 10501 MHz x 2 (GDDR6X) x 48 B bus. Update when the pod's GPU changes
    double copy_bw = gb_per_s(copy_bytes, copy.median_ms);
    double kern_bw = gb_per_s(kern_bytes, kern.median_ms);

    printf("\nN = %d, warm-up %d, reps %d (times per call: median / min / mean)\n", N, warmup, reps);
    printf("D2D copy  : %8.1f / %8.1f / %8.1f us  %6.1f GB/s  (%5.1f%% of spec %.0f)\n",
           copy.median_ms * 1e3, copy.min_ms * 1e3, copy.mean_ms * 1e3, copy_bw,
           100 * copy_bw / peak_spec, peak_spec);
    printf("reductionWarpShuffle : %8.1f / %8.1f / %8.1f us  %6.1f GB/s  (%5.1f%% of spec, %5.1f%% of D2D copy)\n",
           kern.median_ms * 1e3, kern.min_ms * 1e3, kern.mean_ms * 1e3, kern_bw,
           100 * kern_bw / peak_spec, 100 * kern_bw / copy_bw);

    cudaFree(copy_d);

    cudaFree(input_d);
    cudaFree(output_d);





}
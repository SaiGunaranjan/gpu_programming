/*
To run a C++ cuda code on GPU, do the following:
1. From the Windows Start Menu, open x64 Native Tools Command Prompt for VS
2. It will open a bash shell.
3. There, you run the command
nvcc <file_name.cu> -o <file_name>
eg: nvcc vector_add.cu -o vector_add
4. This will compile the file and create an executable named vector_add
5. Now type file name and hit enter. eg: vector_add
*/


#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
using namespace std;

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ---------------------------------------------------------------------------
// Timing harness
// ---------------------------------------------------------------------------

struct Timing {
    float median_ms;
    float min_ms;
    float mean_ms;
};

// Times fn() on the GPU. fn is anything callable with no arguments that queues
// GPU work, e.g. a lambda that launches a kernel or calls cudaMemcpy.
//   1. Warm-up: `warmup` untimed calls (lazy module load, clock ramp-up).
//   2. Timed: `reps` calls, with an event recorded before the first call and
//      after every call, so each call gets its own GPU time:
//          ev[0] fn ev[1] fn ev[2] ... fn ev[reps]
//      Call r took ev[r] -> ev[r+1]. Each interval also includes the small GPU
//      gap between consecutive kernels.
//   3. Median (robust to outliers), min (best case) and mean (= total / reps).
template <typename F>
Timing time_gpu(F fn, int warmup, int reps)
{
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();   // warm-up must finish before timing starts

    vector<cudaEvent_t> ev(reps + 1);
    for (auto& e : ev) cudaEventCreate(&e);

    cudaEventRecord(ev[0]);           // marker in the queue; returns immediately
    for (int r = 0; r < reps; ++r) {
        fn();                         // queued, returns immediately
        cudaEventRecord(ev[r + 1]);
    }
    cudaEventSynchronize(ev[reps]);   // the only blocking call: wait for the last marker

    vector<float> ms(reps);
    for (int r = 0; r < reps; ++r) cudaEventElapsedTime(&ms[r], ev[r], ev[r + 1]);

    float total = 0.0f;
    cudaEventElapsedTime(&total, ev[0], ev[reps]);
    for (auto& e : ev) cudaEventDestroy(e);

    sort(ms.begin(), ms.end());
    return {ms[reps / 2], ms[0], total / reps};
}

// Bytes moved / seconds, in GB/s (decimal GB, like NVIDIA's spec sheets).
double gb_per_s(double bytes, float ms) { return bytes / (ms * 1e-3) / 1e9; }

int main() {
    int N = 25'000'000;  // 25M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    /*
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    */
   vector<float> h_A(N);
   vector<float> h_B(N);
   vector<float> h_C(N);

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // Launch kernel with N threads, grouped into blocks of 256
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            success = false;
            std::cout << "Error at index " << i << ": " << h_C[i] << "\n";
            break;
        }
    }

    std::cout << (success ? "Vector addition successful!\n" : "Vector addition failed!\n");

    // ---- Timing ----
    const int warmup = 20, reps = 100;

    // Practical peak: device-to-device copy. Reads `size` bytes, writes `size` bytes.
    Timing copy = time_gpu([&] { cudaMemcpy(d_C, d_A, size, cudaMemcpyDeviceToDevice); },
                           warmup, reps);
    // The kernel: reads A and B, writes C.
    Timing kern = time_gpu([&] { vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); },
                           warmup, reps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error during timing: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    double copy_bytes = 2.0 * size, kern_bytes = 3.0 * size;
    double peak_spec = 448.0;   // RTX A4000: 7001 MHz x 2 (GDDR6) x 32 B bus
    double copy_bw = gb_per_s(copy_bytes, copy.median_ms);
    double kern_bw = gb_per_s(kern_bytes, kern.median_ms);

    printf("\nN = %d, warm-up %d, reps %d (times per call: median / min / mean)\n", N, warmup, reps);
    printf("D2D copy  : %8.1f / %8.1f / %8.1f us  %6.1f GB/s  (%5.1f%% of spec %.0f)\n",
           copy.median_ms * 1e3, copy.min_ms * 1e3, copy.mean_ms * 1e3, copy_bw,
           100 * copy_bw / peak_spec, peak_spec);
    printf("vectorAdd : %8.1f / %8.1f / %8.1f us  %6.1f GB/s  (%5.1f%% of spec, %5.1f%% of D2D copy)\n",
           kern.median_ms * 1e3, kern.min_ms * 1e3, kern.mean_ms * 1e3, kern_bw,
           100 * kern_bw / peak_spec, 100 * kern_bw / copy_bw);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    /*
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    */

    return 0;
}

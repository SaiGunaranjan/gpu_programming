/*
Matrix transpose, naive (no shared memory): output (cols x rows) = input (rows x cols)^T.

Each block covers a BM x BK = 64 x 64 tile with 16 x 16 threads; each thread copies a TM x TK = 4 x 4
patch directly from input to output (global to global). Neighbouring threads in a warp are TK columns
apart, so the reads are strided (16 B apart) and the writes are strided by `rows`: neither is
coalesced. This is the baseline for matrix_transpose_smem.cu.

BM: BLOCKSIZE_M (64)
BK: BLOCKSIZE_K (64)
TM: OUTPUT_ROWS_PER_THREAD (4)
TK: OUTPUT_COLS_PER_THREAD (4)

Build and run: see kernels/README.md (Build and run section).
*/

#include <cuda_runtime.h>
#include <vector>
using namespace std;
#include <iostream>
#include <cstdio>
#include <random>
#include <algorithm>

struct Timing{
    float median_ms;
    float min_ms;
    float mean_ms;
};

template <typename F>
Timing time_gpu(F fn, int warmup, int reps)
{
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();

    vector<cudaEvent_t> ev(reps+1);
    for (int j = 0; j < reps+1; j++)  cudaEventCreate(&ev[j]);

    cudaEventRecord(ev[0]);
    for (int r = 0; r < reps; r++)
    {
        fn();
        cudaEventRecord(ev[r+1]);
    }

    cudaEventSynchronize(ev[reps]);

    vector<float> ms(reps);
    for (int j = 0; j < reps; j++) cudaEventElapsedTime(&ms[j],ev[j], ev[j+1]);

    float total = 0.0f;
    cudaEventElapsedTime(&total,ev[0],ev[reps]);

    for (int i = 0; i < reps+1; i++) cudaEventDestroy(ev[i]);

    sort(ms.begin(), ms.end());
    return {ms[reps/2], ms[0], total/reps};
   
}


template<int BM, int BK, int TM, int TK>
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols)
{
    int tid_x = (blockIdx.x * BK) + (threadIdx.x * TK); // Global Thread Id along cols
    int tid_y = (blockIdx.y * BM) + (threadIdx.y * TM); // Global Thread Id along rows

    for (int i=0; i<TM; i++)
    {
        for (int j=0; j<TK; j++)
        {
            int row_id = tid_y + i;
            int col_id = tid_x + j;
            if ((row_id < rows) && (col_id < cols))
            {
                output[col_id*rows + row_id] = input[row_id*cols + col_id];
            }
        }
    }


}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols)
{
    const int BM = 64;
    const int BK = 64;
    const int TM = 4;
    const int TK = 4;

    dim3 threadsPerBlock(BK/TK,BM/TM); // dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid(((cols+BK-1) / BK), ((rows+BM-1)/BM));

    matrix_transpose_kernel<BM, BK, TM, TK><<<blocksPerGrid,threadsPerBlock>>>(input, output, rows, cols);
    // No cudaDeviceSynchronize() here: kernel launches are asynchronous and the caller syncs when it
    // needs the result. A sync inside solve() would be timed by time_gpu, adding a host round-trip
    // (~5 us, ~1.5-2% here) to every call.
}

double gb_per_s(double bytes, float ms){ return (bytes/(ms*1e-3))/1e9; }

int main()
{

    int rows = 7000;
    int cols = 6000;

    size_t size_input = rows * cols * sizeof(float);
    size_t size_output = rows * cols * sizeof(float);

    random_device rd; // Obtains a random seed for hardware
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f,1.0f);

    vector<float> input_h(rows*cols);
    vector<float> output_h(rows*cols);
    vector<float> output_from_device(rows*cols);
    
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            input_h[i*cols + j] = dis(gen);
            output_h[j*rows + i] = input_h[i*cols + j];
        }
    }

    // Define pointers for device memory
    float* input_d;
    float* output_d;

    cudaMalloc(&input_d,size_input); // Allocates size_input bytes in device memory (DRAM) and stores the device address in input_d
    cudaMalloc(&output_d,size_output);

    // Copy data from host to device
    cudaMemcpy(input_d,input_h.data(),size_input,cudaMemcpyHostToDevice);

    // Populate every output byte with FF (max value), so float (4 bytes) becomes 0xFFFFFFFF
    cudaMemset(output_d, 0xFF, size_output); // NaN pattern: elements the kernel skips stay NaN

    // --- Correctness run ---
    solve(input_d, output_d, rows, cols);

    // solve() doesn't sync (see the comment in solve), so sync here before using the result.
    // Check both kinds of error: launch errors (cudaGetLastError, e.g. a bad launch config) and errors
    // while the kernel runs (returned by cudaDeviceSynchronize, e.g. an illegal memory access).
    cudaError_t launch_err = cudaGetLastError();
    cudaError_t run_err = cudaDeviceSynchronize();
    if (launch_err != cudaSuccess || run_err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(launch_err != cudaSuccess ? launch_err : run_err));
        return 1;
    }

    cudaMemcpy(output_from_device.data(),output_d,size_output,cudaMemcpyDeviceToHost);

    // Count mismatches; print only the first few so a broken kernel doesn't print millions of lines
    long long mismatches = 0;
    const int max_print = 5;
    for (int i=0; i<cols; i++)
    {
        for (int j=0; j<rows; j++)
        {
            if (output_h[i*rows + j] != output_from_device[i*rows + j])
            {
                if (mismatches < max_print)
                {
                    printf("Mismatch at output[%d][%d]: expected %f, got %f\n",
                           i, j, output_h[i*rows + j], output_from_device[i*rows + j]);
                }
                mismatches++;
            }
        }
    }
    if (mismatches == 0) printf("Transpose correct\n");
    else                 printf("Transpose incorrect: %lld of %lld elements differ\n", mismatches, (long long)rows*cols);


    const int warmup = 20, reps = 100;

    Timing copy = time_gpu([&] {cudaMemcpy(output_d, input_d, size_input, cudaMemcpyDeviceToDevice);},
                warmup, reps);
    
    Timing kern = time_gpu([&] {solve(input_d, output_d, rows, cols);}, warmup, reps);

    // Check for errors from the timed launches too (120 more kernel launches)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Both read every element once and write it once, so the same byte count
    double copy_bytes = 2 * size_input, kern_bytes = 2 * size_input;
    double peak_spec = 2039; // A100 SXM: 1593 MHz x 2 (HBM2e) x 640 B bus. Update when the pod's GPU changes
    double copy_bw = gb_per_s(copy_bytes, copy.median_ms);
    double kern_bw = gb_per_s(kern_bytes, kern.median_ms);

    printf("\nrows x cols = %d x %d, warm-up %d, reps %d (times per call: median / min / mean)\n", rows, cols, warmup, reps);
    printf("D2D copy  : %8.1f / %8.1f / %8.1f us  %6.1f GB/s  (%5.1f%% of spec %.0f)\n",
           copy.median_ms * 1e3, copy.min_ms * 1e3, copy.mean_ms * 1e3, copy_bw,
           100 * copy_bw / peak_spec, peak_spec);
    printf("matrixTranspose : %8.1f / %8.1f / %8.1f us  %6.1f GB/s  (%5.1f%% of spec, %5.1f%% of D2D copy)\n",
           kern.median_ms * 1e3, kern.min_ms * 1e3, kern.mean_ms * 1e3, kern_bw,
           100 * kern_bw / peak_spec, 100 * kern_bw / copy_bw);



    cudaFree(input_d);
    cudaFree(output_d);

    
}
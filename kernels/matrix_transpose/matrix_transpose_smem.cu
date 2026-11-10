/*
Matrix transpose, shared-memory tiled: output (cols x rows) = input (rows x cols)^T.

Each block transposes one BM x BK = 64 x 64 tile with 16 x 16 = 256 threads (16 elements per thread):
  Phase 1: coalesced load of the tile from DRAM into shared memory (row-wise).
  Phase 2: coalesced store of the transposed tile back to DRAM (read column-wise from shared memory).
The shared-memory tile is padded by one column (BK+1) so the column-wise reads in phase 2 hit
different banks (no bank conflicts).

BM: tile rows    (BLOCKSIZE_M, 64)
BK: tile columns (BLOCKSIZE_K, 64)

Launch-config constraint: phase 2 assumes the thread count per block is a multiple of BM
(num_rows = threads / BM). 16 x 16 and 32 x 32 threads work with BM = 64; other configs are untested.

Build and run: see kernels/README.md (Build and run section).

TODO: float4 loads and stores. Each thread currently moves one 4-byte float per access; 16-byte
      (float4) accesses keep more bytes in flight per memory instruction. This is the main reason the
      kernel stays below the D2D cudaMemcpy (see README.md, "Why the D2D copy isn't an
      apples-to-apples target"). Would be a new variant file.
*/

#include <cuda_runtime.h>
#include <vector>
using namespace std;
#include <iostream>
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


template<int BM, int BK>
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols)
{

    int lin_tid = threadIdx.y*blockDim.x + threadIdx.x;
    const int num_threads_perblock = (blockDim.x*blockDim.y);

    int block_start_row = blockIdx.y*BM;
    int block_start_col = blockIdx.x*BK;

    __shared__ float input_tile_smem[BM][BK+1]; // One extra padded column to avoid bank conflicts! Can use xor or circular row rotation as well!

    // Phase 1: load the tile into shared memory. Consecutive threads read consecutive columns of a
    // row, so each warp's global reads are coalesced.
    for (int i=lin_tid; i<BM*BK; i+=num_threads_perblock)
    {
        int row = i / BK;
        int col = i % BK;
        int abs_row_id = block_start_row + row;
        int abs_col_id = block_start_col + col;
        if ((abs_row_id < rows) && (abs_col_id < cols))
        {
            // Edge blocks leave some smem entries unwritten (junk). They are never stored to DRAM:
            // phase 2 has the same bounds check.
            input_tile_smem[row][col] = input[abs_row_id*cols + abs_col_id];
        }
    }

    // Wait for all threads to finish copying from DRAM to shared memory before starting to write back to DRAM in transposed manner
    __syncthreads();

    // Phase 2: store the transposed tile. Consecutive threads take consecutive tile rows (= consecutive
    // output columns), so each warp's global writes are coalesced. They read a column of the smem
    // tile; the +1 padding puts those words in different banks.
    int num_rows =  num_threads_perblock / BM;

    // row and col are flipped to ensure warps spread across rows of a given column (This works out well only after avoiding bank conflicts). Different words --> Different banks
    int col = lin_tid / BM;
    int row = lin_tid % BM;

    for (int new_col=col; new_col < BK; new_col+=num_rows)
    {
        int global_row = blockIdx.y*BM + row;
        int global_col = blockIdx.x*BK + new_col;
        if ((global_row < rows) && (global_col < cols))
        {
            output[global_col*rows + global_row] = input_tile_smem[row][new_col];
        }

    }


}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols)
{
    const int BM = 64;
    const int BK = 64;

    dim3 threadsPerBlock(16,16); //dim3 threadsPerBlock(32,32); also works (thread count is a multiple of BM)
    dim3 blocksPerGrid(((cols+BK-1) / BK), ((rows+BM-1)/BM));

    matrix_transpose_kernel<BM, BK><<<blocksPerGrid,threadsPerBlock>>>(input, output, rows, cols);
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

    // solve() no longer syncs (see the comment in solve), so sync here before using the result.
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

    // --- Timing ---
    const int warmup = 20, reps = 100;

    // Practical peak: device-to-device copy. Reads size_input bytes, writes size_input bytes.
    Timing copy = time_gpu([&] { cudaMemcpy(output_d, input_d, size_input, cudaMemcpyDeviceToDevice);},
                    warmup, reps);

    // Reads input, writes input^T
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
    printf("matrixTransposeSMEM : %8.1f / %8.1f / %8.1f us  %6.1f GB/s  (%5.1f%% of spec, %5.1f%% of D2D copy)\n",
           kern.median_ms * 1e3, kern.min_ms * 1e3, kern.mean_ms * 1e3, kern_bw,
           100 * kern_bw / peak_spec, 100 * kern_bw / copy_bw);


    cudaFree(input_d);
    cudaFree(output_d);


}

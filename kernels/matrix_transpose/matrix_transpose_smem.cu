/*
BM: BLOCKSIZE_M (64)
BK: BLOCKSIZE_K (64)
TM: OUTPUT_ROWS_PER_THREAD (4)
TK: OUTPUT_COLS_PER_THREAD (4)
*/

#include <cuda_runtime.h>
#include <vector>
using namespace std;
#include <iostream>
#include <random>


template<int BM, int BK>
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols)
{
    
    int lin_tid = threadIdx.y*blockDim.x + threadIdx.x;
    const int num_threads_perblock = (blockDim.x*blockDim.y);

    int block_start_row = blockIdx.y*BM;
    int block_start_col = blockIdx.x*BK;

    __shared__ float input_tile_smem[BM][BK+1]; // One extra padded column to avoid bank conflicts! Can use xor or circular row rotation as well!

    for (int i=lin_tid; i<BM*BK; i+=num_threads_perblock)
    {
        int row = i / BK;
        int col = i % BK;
        int abs_row_id = block_start_row + row;
        int abs_col_id = block_start_col + col;
        if ((abs_row_id < rows) && (abs_col_id < cols))
        {
            input_tile_smem[row][col] = input[abs_row_id*cols + abs_col_id]; // Note: For the edge threadBlocks, there might be junk entries in the input_tile_smem since they are not populated with this line!
        }  
    }

    // Wait for all threads to finish copying from DRAM to shared memory before starting to write back to DRAM in transposed manner
    __syncthreads();
    
    int num_rows =  num_threads_perblock / BM;
    // int num_cols = BM;
    
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

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid(((cols+BK-1) / BK), ((rows+BM-1)/BM));

    matrix_transpose_kernel<BM, BK><<<blocksPerGrid,threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize(); // Wait for all threads to finish
}

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
    
    float count = 0;
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            input_h[i*cols + j] = dis(gen);
            output_h[j*rows + i] = input_h[i*cols + j];
            count += 1;
        }
    }

    // Define pointers for device memory
    float* input_d;
    float* output_d;

    cudaMalloc(&input_d,size_input); // Cudammalloc allocates size_input amount of bytes on device memory(DRAM) and returns a pointer to this device memory. Which is then populated into input_d. input_d now points at the device memory
    cudaMalloc(&output_d,size_output);

    // Copy data from host to device
    cudaMemcpy(input_d,input_h.data(),size_input,cudaMemcpyHostToDevice);

    // Populate every output byte with FF (max value), so float (4 bytes) becomes 0xFFFFFFFF
    cudaMemset(output_d, 0xFF, size_output); // NaN pattern: elements the kernel skips stay NaN

    solve(input_d, output_d, rows, cols);

    // Check for kernel launch error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s", cudaGetErrorString(err));
    }

    cudaMemcpy(output_from_device.data(),output_d,size_output,cudaMemcpyDeviceToHost);

    for (int i=0; i<cols; i++)
    {
        for (int j=0; j<rows; j++)
        {
            if (output_h[i*rows + j] != output_from_device[i*rows + j])
            {
                cout << "Transpose incorrect!";
                // cout << "\noutput_h[" << i << "][" << j << "] = " << output_h[i*rows + j] << "; output_d[" << i << "][" << j << "] = " << output_from_device[i*rows + j];
            }
        }
    }

    cudaFree(input_d);
    cudaFree(output_d);

    
}
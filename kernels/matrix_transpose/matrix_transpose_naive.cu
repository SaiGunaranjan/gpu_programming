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


template<int BM, int BK, int TM, int TK>
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols)
{
    int tid_x = (blockIdx.x * blockDim.x * TK) + (threadIdx.x * TK); // Global Thread Id along cols
    int tid_y = (blockIdx.y * blockDim.y * TM) + (threadIdx.y * TM); // Global Thread Id along rows

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
            }
        }
    }

    cudaFree(input_d);
    cudaFree(output_d);

    
}
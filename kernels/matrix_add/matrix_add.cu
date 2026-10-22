#include <iostream>
#include <cuda_runtime.h>
#include <vector>
using namespace std;


__global__ void matrixAdd(const float* A, const float* B, float* C, int num_rows, int num_cols)
{
    int thrd_id_x = (blockDim.x * blockIdx.x) + threadIdx.x; // cols
    int thrd_id_y = (blockDim.y * blockIdx.y) + threadIdx.y; // rows

    if ((thrd_id_x >= num_cols) || (thrd_id_y >= num_rows))
    {
        return;
    }

    C[num_cols*thrd_id_y + thrd_id_x] = A[num_cols*thrd_id_y + thrd_id_x] + B[num_cols*thrd_id_y + thrd_id_x];
}

int main()
{
    // Define number of rows and columns
    int num_rows = 1000;
    int num_cols = 1000;
    size_t size = num_rows * num_cols * sizeof(float);

    // Initialize host memory. Allocate the memory as a flat 1D array
    vector<float> h_A(num_rows*num_cols);
    vector<float> h_B(num_rows*num_cols);
    vector<float> h_C(num_rows*num_cols);

    // Define host matrices
    for (int i = 0; i < num_rows; i++) 
    {
        for (int j = 0; j < num_cols; j++)
        {
            h_A[i*num_cols + j] = static_cast<float>(i*j);
            h_B[i*num_cols + j] = static_cast<float>(2 * i * j);
        }
        
    }

    // Define device pointers
    float *d_A, *d_B, *d_C;

    // Initialize device memory
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    // Copy data from host to device
    cudaMemcpy(d_A,h_A.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B.data(),size,cudaMemcpyHostToDevice);

    // Launch kernel from host to execute on device
    int threads_per_block_x = 16;
    int threads_per_block_y = 16;

    int blocks_per_grid_x = (num_cols + threads_per_block_x - 1)/threads_per_block_x;
    int blocks_per_grid_y = (num_rows + threads_per_block_y - 1)/threads_per_block_y;

    dim3 threads_per_block(threads_per_block_x,threads_per_block_y);
    dim3 blocks_per_grid(blocks_per_grid_x,blocks_per_grid_y);

    matrixAdd<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, num_rows, num_cols);

    // Synchronize all the threads after execution and before copying back the data back to the host
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(),d_C,size,cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < num_rows; i++) 
    {
        for (int j = 0; j < num_cols; j++)
        {
            if (fabs(h_C[i*num_cols + j] - (h_A[i*num_cols + j] + h_B[i*num_cols + j])) > 1e-5) 
            {
            success = false;
            cout << "Error at index " << i << ", " << j << ": " << h_C[i*num_cols + j] << "\n";
            break;
            }
        }  
    }

    cout << (success ? "Matrix addition successful!\n" : "Matrix addition failed!\n");


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
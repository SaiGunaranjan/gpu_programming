#include <iostream>
#include <cuda_runtime.h>
#include <vector>
using namespace std;

__global__ void matMul(const float* A, const float* B, float* C, int num_rows_C, int num_cols_C, int num_cols_A)
{
    int thrd_id_x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int thrd_id_y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if ((thrd_id_x >= num_cols_C) || (thrd_id_y >= num_rows_C))
    {
        return;
    }

    float sum = 0;
    for (int i = 0; i < num_cols_A; i++)
    {
        sum += (A[thrd_id_y*num_cols_A + i] * B[i*num_cols_C + thrd_id_x]);
    }
    C[thrd_id_y*num_cols_C + thrd_id_x] = sum;

    
}

int main()
{
    int num_rows_A = 1024;//1024;
    int num_cols_A = 256;//256;
    int num_rows_B = num_cols_A;
    int num_cols_B = 2048;//2048;
    int num_rows_C = num_rows_A;
    int num_cols_C = num_cols_B;

    size_t size_A = num_rows_A*num_cols_A*sizeof(float);
    size_t size_B = num_rows_B*num_cols_B*sizeof(float);
    size_t size_C = num_rows_C*num_cols_C*sizeof(float);


    
    // Initialize host matrices
    vector<float> h_A(num_rows_A * num_cols_A);
    vector<float> h_B(num_rows_B * num_cols_B);
    vector<float> h_C(num_rows_C * num_cols_C);
    vector<float> h_C_cpu(num_rows_C * num_cols_C);

    // Define values for host matrices

    // Define A matrix
    for (int i = 0; i < num_rows_A; i++)
    {
        for (int j = 0; j < num_cols_A; j++)
        {
            h_A[i*num_cols_A + j] = static_cast<float>(i+2+j-1);
        }
    }

    // Define B matrix
    for (int i = 0; i < num_rows_B; i++)
    {
        for (int j = 0; j < num_cols_B; j++)
        {
            h_B[i*num_cols_B + j] = static_cast<float>(i+j);
        }
    }

    // Compute C = AB on CPU
    for (int i = 0; i < num_rows_A; i++)
    {
        for (int j = 0; j < num_cols_B; j++)
        {
            float sum = 0;
            for (int k = 0; k < num_cols_A; k++)
            {
                sum += (h_A[i*num_cols_A + k]*h_B[k*num_cols_B + j]);
            }
            h_C_cpu[i*num_cols_B + j] = sum;
        }
    }

    // Define pointers for device memory
    float* d_A;
    float* d_B;
    float* d_C;

    // Allocate memory on the device
    cudaMalloc(&d_A,size_A);
    cudaMalloc(&d_B,size_B);
    cudaMalloc(&d_C,size_C);

    // Copy matrices from the host to the device
    cudaMemcpy(d_A,h_A.data(),size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B.data(),size_B,cudaMemcpyHostToDevice);


    // Launch the kernel
    int threads_per_block_x = 32;
    int threads_per_block_y = 32;

    int blocks_per_grid_x = (num_cols_C + threads_per_block_x - 1)/threads_per_block_x;
    int blocks_per_grid_y = (num_rows_C + threads_per_block_y - 1)/threads_per_block_y;

    dim3 threads_per_block(threads_per_block_x,threads_per_block_y);
    dim3 blocks_per_grid(blocks_per_grid_x, blocks_per_grid_y);

    matMul<<<blocks_per_grid,threads_per_block>>>(d_A, d_B, d_C, num_rows_C, num_cols_C, num_cols_A);

    // Synchronize the threads after execution
    cudaDeviceSynchronize();

    // Copy back data from device to host
    cudaMemcpy(h_C.data(),d_C,size_C,cudaMemcpyDeviceToHost);

    

    // Check for correctness of cuda implementation
    // Verify result
    bool success = true;
    for (int i = 0; i < num_rows_C; i++)
    {
        for (int j = 0; j < num_cols_C; j++)
        {
            if (fabs(h_C_cpu[i*num_cols_C + j] - h_C[i*num_cols_C + j]) > 1e-5)
            {
                success = false;
                cout << "Error at index " << i << ", " << j << ": " << h_C[i*num_cols_C + j] << "\n";
                break;
            }
        }
    }

    cout << (success ? "Matrix multiplication successfull!\n": "Matrix multiplication failed!\n");

    // Free up device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;







}
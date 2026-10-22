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
using namespace std;

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1 << 20;  // 1M elements
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

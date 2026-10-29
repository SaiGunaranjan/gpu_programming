/*
matmul, 2D blocktiling: A and B tiles are staged in shared memory, and each thread computes a
TM x TK block of outputs (an outer product per k step) held in registers.

Naming follows the LeetGPU problem: A is M x N, B is N x K, C is M x K, so N is the inner
dimension. (The usual convention is A (M x K) @ B (K x N).)

BM: BLOCKSIZE_M (64)
BN: BLOCKSIZE_N (64)
BK: BLOCKSIZE_K (64)
TM: OUTPUT_ROWS_PER_THREAD (4)
TK: OUTPUT_COLS_PER_THREAD (4)

Build and run: see kernels/README.md. Known issues: see the "Open issues" in this folder's README.md.


My queries:
1. Handle the boundries carefully!! Check for bugs.

*/

#include <cuda_runtime.h>
#include <vector>
using namespace std;
#include <iostream>

template <int BM, int BN, int BK, int TM, int TK>
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    // global_tid_load_x = blockIdx.x * blockDim.x + threadIdx.x;
    // global_tid_load_y = blockIdx.y * blockDim.y + threadIdx.y;

    // local_tid_load_x = threadIdx.x;
    // local_tid_load_y = threadIdx.y;

    float acc[TM][TK]; // Accumulator register for each thread.
    // INitialize to all zeros
    for (int i=0; i<TM; i++)
    {
        for (int j=0; j<TK; j++)
        {
            acc[i][j] = 0.0f;
        }
    }
    float a[TM] = {0};
    float b[TK] = {0};

    // Load the data from DRAM to shared memory

    __shared__ float A_smem[BM][BN];
    __shared__ float B_smem[BN][BK];



    int local_lin_idx = threadIdx.y*blockDim.x + threadIdx.x;
    int local_row_A = local_lin_idx / BN; // Floor division
    int local_col_A = local_lin_idx % BN;
    int local_row_B = local_lin_idx / BK; // Floor division
    int local_col_B = local_lin_idx % BK;
    const int num_threads_per_block = blockDim.x * blockDim.y; // How to better define this?
    int offset_row_A = (num_threads_per_block) / BN; // Floor division
    int offset_col_A = (num_threads_per_block) % BN;
    int offset_row_B = (num_threads_per_block) / BK; // Floor division
    int offset_col_B = (num_threads_per_block) % BK;

    int num_thread_loads_A = (BM*BN + num_threads_per_block - 1) / num_threads_per_block; // ceil division
    int num_thread_loads_B = (BN*BK + num_threads_per_block - 1) / num_threads_per_block; // ceil division

    int global_base_row_A = blockIdx.y * blockDim.y * TM;
    int global_base_col_B = blockIdx.x * blockDim.x * TK;

    

    for (int bn_idx=0; bn_idx < N; bn_idx+=BN)
    {
        int global_base_col_A = bn_idx; // ??
        int global_base_row_B = bn_idx;

        
        // Inner loop for loading a tile of A
        for(int i=0; i < num_thread_loads_A; i++)
        {
            // All the below are for A, need to have similar for B, also for local_row_A/col, offset_row_A/col for B
            int new_local_row_A = local_row_A + i*offset_row_A;
            int new_local_col_A = local_col_A + i*offset_col_A; // Is this formula correct?
            int flatten_lin_idx_A = new_local_row_A*BN + new_local_col_A;
            new_local_row_A = flatten_lin_idx_A / BN;
            new_local_col_A = flatten_lin_idx_A % BN;


            int abs_row_A = global_base_row_A + new_local_row_A;
            int abs_col_A = global_base_col_A + new_local_col_A;
            flatten_lin_idx_A = abs_row_A*N + abs_col_A;

            // If condition for valid loads for global A as well? What happens to the last block which is not of size BMxBN? What happens at the boundries?
            // if (new_local_row_A < BM) 
            // {
            //     A_smem[new_local_row_A,new_local_col_A] = A[flatten_lin_idx_A];
            // }
            if ((abs_row_A < M) && (abs_col_A < N))
            {
                if (new_local_row_A < BM)
                {
                    A_smem[new_local_row_A][new_local_col_A] = A[flatten_lin_idx_A];
                }
            }
            // should we have an else statement and put zeros
            else
            {
                A_smem[new_local_row_A][new_local_col_A] = 0.0f;
            }
                
        }
        
        
        // __syncthreads(); // This is a debug print only synthreads, so printing happens only after the full tile is copied! Remove it once debug is done
        // if ((blockIdx.x ==0) && (blockIdx.y ==0) && (bn_idx==0) && (local_lin_idx==0))
        // {
        //     for (int r=0; r<BM; r++) // for (int r=0; r<BM; r++)
        //     {
        //         printf("\n");
        //         for (int c=0; c<BN; c++) // for (int c=0; c<BN; c++)
        //         {
        //             printf("%6.1f", A_smem[r][c]);
        //         }
        //     }
        // }


        // __syncthreads(); // This is a debug print only synthreads, so printing happens only after the full tile is copied! Remove it once debug is done
        // for (int i=local_lin_idx; i<BM*BN; i+=num_threads_per_block)
        // {
        //     int r = i / BN;
        //     int c = i % BN;
        //     int global_r = global_base_row_A + r;
        //     int global_c = global_base_col_A + c;
        //     int global_idx_A = global_r*N + global_c;
        //     if (global_r < M && global_c < N)
        //     {
        //         if (A_smem[r][c] != A[global_idx_A])
        //         {
        //             // printf("\nMismatch at A[%f][%f]",global_r,global_c);
        //             // printf("\nA_smem[%d][%d]=%f, A[%d][%d]=%f", r,c,A_smem[r][c],global_r,global_c,A[global_idx_A]);
        //             printf("\nA_smem[%d][%d]=%f, A[%d][%d]=%f", r,c,A_smem[r][c],global_r,global_c,A[global_idx_A]);
        //         }
        //     }
        //     else
        //     {
        //         if (A_smem[r][c] != 0.0f)
        //         {
        //             printf("Boundries are not matching!");
        //         }
        //     }
        // }



        // Inner loop for loading a tile of B
        for(int i=0; i < num_thread_loads_B; i++)
        {
            int new_local_row_B = local_row_B + i*offset_row_B;
            int new_local_col_B = local_col_B + i*offset_col_B;
            int flatten_lin_idx_B = new_local_row_B*BK + new_local_col_B;
            new_local_row_B = flatten_lin_idx_B  / BK;
            new_local_col_B = flatten_lin_idx_B % BK;

            int abs_row_B = global_base_row_B + new_local_row_B;
            int abs_col_B = global_base_col_B + new_local_col_B;
            flatten_lin_idx_B = abs_row_B*K + abs_col_B;

      
            if ((abs_row_B < N) && (abs_col_B < K))
            {
                if (new_local_row_B < BN)
                {
                    B_smem[new_local_row_B][new_local_col_B] = B[flatten_lin_idx_B];
                }
                
            }
            else
            {
                B_smem[new_local_row_B][new_local_col_B] = 0.0f;
            }
        }

        // __syncthreads(); // This is a debug print only synthreads, so printing happens only after the full tile is copied! Remove it once debug is done
        // for (int i=local_lin_idx; i<BN*BK; i+=num_threads_per_block)
        // {
        //     int r = i / BK;
        //     int c = i % BK;
        //     int global_r = global_base_row_B + r;
        //     int global_c = global_base_col_B + c;
        //     int global_idx_B = global_r*K + global_c;
        //     if (global_r < N && global_c < K)
        //     {
        //         if (B_smem[r][c] != B[global_idx_B])
        //         {
        //             printf("\nB_smem[%d][%d]=%f, B[%d][%d]=%f", r,c,B_smem[r][c],global_r,global_c,B[global_idx_B]);
        //         }
        //     }
        //     else
        //     {
        //         if (B_smem[r][c] != 0.0f)
        //         {
        //             printf("Boundries are not matching!");
        //         }
        //     }
        // }

        __syncthreads(); // Wait for all threads to finish copying that tile before starting compute
        // Compute
        
        for(int l=0; l<BN; l++)
        {   
            #pragma unroll
            for (int i=0; i<TM; i++)
            {
                a[i] = A_smem[threadIdx.y*TM + i][l];
            }

            // if (blockIdx.x==0 && blockIdx.y==0 && bn_idx==0 && local_lin_idx==0 && l==BN-1)
            // {

            //     for (int i=0; i<TM; i++)
            //     {
            //         printf("\na[%d]=%f",i,a[i]);
            //     }
            // }

            #pragma unroll
            for (int j=0; j<TK; j++)
            {
                b[j] = B_smem[l][threadIdx.x*TK + j];
            }

            // if (blockIdx.x==0 && blockIdx.y==0 && bn_idx==0 && local_lin_idx==0 && l==BN-1)
            // {

            //     for (int i=0; i<TK; i++)
            //     {
            //         printf("\nb[%d]=%f",i,b[i]);
            //     }
            // }

            #pragma unroll
            for (int i=0; i<TM; i++)
            {
                #pragma unroll
                for (int j=0; j<TK; j++)
                {
                    acc[i][j] += a[i]*b[j];
                }
            }
        }

        __syncthreads(); // Wait for all threads to finish compute before starting to overwrite A_smem and B_smem with next iteration j, else, without this sync barrier, faster threads will start reading the next j and overwriting smem while the slow thread is still reading smem. So memory corruption can happen
    }

    // if (blockIdx.x==0 && blockIdx.y==0 && local_lin_idx==1)
    // {
    //     #pragma unroll
    //     for (int i=0; i<TM; i++)
    //     {
    //         printf("\n");
    //         #pragma unroll
    //         for (int j=0; j<TK; j++)
    //         {
    //             printf(" acc[%d][%d] = %f",i,j,acc[i][j]);
    //         }
    //     }
    // }

    int global_tidx_C = blockIdx.x*blockDim.x*TK + threadIdx.x*TK;
    int global_tidy_C = blockIdx.y*blockDim.y*TM + threadIdx.y*TM;

    #pragma unroll
    for (int i=0; i<TM; i++)
    {   
        #pragma unroll
        for (int j=0; j<TK; j++)
        {
            int row_c = global_tidy_C + i;
            int col_c = global_tidx_C + j;
            if (row_c < M && col_c < K)
            {
                C[row_c*K + col_c] = acc[i][j];
            }
            
        }
    }
    

    

}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K)
{   

    const int BM = 64;
    const int BN = 64;
    const int BK = 64;
    const int TM = 4;
    const int TK = 4;

    dim3 threadsPerBlock(BK/TK,BM/TM);//dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid(((K+BK-1)/BK), ((M+BM-1)/BM)); // Order should be x,y --> cols, rows

    // printf("blocksPerGrid.x = %d, blocksPerGrid.y = %d", blocksPerGrid.x, blocksPerGrid.y);

    matrix_multiplication_kernel<BM, BN, BK, TM, TK><<<blocksPerGrid,threadsPerBlock>>>(A, B, C, M, N, K);
    // cudaGetLastError(); // What can this be used for?
    cudaDeviceSynchronize();
}


int main()
{

    int num_rows_A = 103;//96;
    int num_cols_A = 167;//160;
    int num_rows_B = num_cols_A;
    int num_cols_B = 129;//96;
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

    solve(d_A, d_B, d_C, num_rows_A, num_cols_A, num_cols_B);

    cudaMemcpy(h_C.data(),d_C,size_C,cudaMemcpyDeviceToHost);

    for (int i=0; i<(num_rows_C * num_cols_C); i++)
    {
        if (h_C_cpu[i] != h_C[i])
        {
            cout << "output_cpu[" << i << "] = " << h_C_cpu[i] << ", output_gpu[" << i << "] = " << h_C[i] << "\n";
        }
    }
}





import torch
import triton
import triton.language as tl

import os
import pdb
# Enable the Triton interpreter mode
# os.environ["TRITON_INTERPRET"] = "1"

torch.manual_seed(0)


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    BLOCK_SIZE_C_ROWS: tl.constexpr, 
    BLOCK_SIZE_C_COLS: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr
):
    
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)

    offsets_a_rows = (pid_y*BLOCK_SIZE_C_ROWS + tl.arange(0,BLOCK_SIZE_C_ROWS))
    abs_offsets_a_rows = offsets_a_rows*stride_am

    # if pid_y == 0 and pid_x == 0:
    #     pdb.set_trace()

    offsets_b_cols = (pid_x*BLOCK_SIZE_C_COLS + tl.arange(0,BLOCK_SIZE_C_COLS))
    abs_offsets_b_cols = offsets_b_cols*stride_bk

    acc = tl.zeros((BLOCK_SIZE_C_ROWS,BLOCK_SIZE_C_COLS),dtype=tl.float32)

    for i in range(0,tl.cdiv(N,BLOCK_SIZE_N)):

        offsets_a_cols_b_rows = i*BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
        offsets_a_cols =  offsets_a_cols_b_rows*stride_an
        offsets_a = abs_offsets_a_rows[:,None] + offsets_a_cols[None,:]
        mask_a = (offsets_a_rows[:,None] < M) & (offsets_a_cols_b_rows[None,:] < N)
        # check mask, offsets strides

        offsets_b_rows = offsets_a_cols_b_rows*stride_bn
        offsets_b = offsets_b_rows[:,None] + abs_offsets_b_cols[None,:]
        mask_b = (offsets_a_cols_b_rows[:,None] < N ) & (offsets_b_cols[None,:] < K)

        a_ = tl.load(a+offsets_a,mask=mask_a,other=0)
        b_ = tl.load(b+offsets_b,mask=mask_b,other=0)

        acc = tl.dot(a_,b_,acc,allow_tf32=False) # input_precision="ieee", # allow_tf32=False, input_precision="tf32x3"
    
    # if pid_y == 0 and pid_x == 0:
    #     pdb.set_trace()

    offsets_c = offsets_a_rows[:,None]*stride_cm + offsets_b_cols[None,:]*stride_ck
    mask_c = (offsets_a_rows[:,None] < M) & (offsets_b_cols[None,:] < K)

    tl.store(c+offsets_c,acc,mask=mask_c)




# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):

    stride_am, stride_an = a.stride(0), a.stride(1) # start changing from here
    stride_bn, stride_bk = b.stride(0), b.stride(1)
    stride_cm, stride_ck = c.stride(0), c.stride(1)

    M = a.shape[0]
    N_a = a.shape[1]
    N = b.shape[0]
    K = b.shape[1]

    assert N_a==N, f"Cols of A {N_a} should be equal to Rows of B {N}"
    

    grid = lambda meta: (triton.cdiv(M,meta["BLOCK_SIZE_C_ROWS"]), triton.cdiv(K,meta["BLOCK_SIZE_C_COLS"]))
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
        BLOCK_SIZE_C_ROWS=64, BLOCK_SIZE_C_COLS=64, BLOCK_SIZE_N=32
    )



M = 8001#64#8192#4
N = 2003#32#6144#8
K = 4017#64#4096#2
# a = torch.randint(high=10,size=(M,N),device='cuda').float()
# b = torch.randint(high=10,size=(N,K),device='cuda').float()
a = torch.randn(size=(M,N),device='cuda').float()
b = torch.randn(size=(N,K),device='cuda').float()
c = torch.empty(M,K,device='cuda')

# solve(a[0::2,0::2],b[0::2,0::3],c[0::2,0::3],M,N,K)
solve(a,b,c,M,N,K)

# c_torch = torch.matmul(a[0::2,0::2],b[0::2,0::3])
c_torch = torch.matmul(a,b)

# torch.testing.assert_close(c[0::2,0::3],c_torch)
torch.testing.assert_close(c,c_torch)

# print("OK  max abs diff =", (c[0::2,0::3] - c_torch).abs().max().item())
print("OK  max abs diff =", (c - c_torch).abs().max().item())

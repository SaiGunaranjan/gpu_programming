import torch
import triton
import triton.language as tl

import os
# os.environ["TRITON_INTERPRET"] = "1"
# torch.manual_seed(0)

@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc,BLOCK_M: tl.constexpr,BLOCK_K: tl.constexpr):

    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)

    block_start_y = pid_y*BLOCK_M
    offsets_y_2d = block_start_y + tl.arange(0,BLOCK_M)

    block_start_x = pid_x*BLOCK_K
    offsets_x_2d = block_start_x + tl.arange(0,BLOCK_K)

    offsets_input = (offsets_y_2d*stride_ir)[:,None] + (offsets_x_2d*stride_ic)[None,:]
    mask_input = (offsets_y_2d[:,None] < rows) & (offsets_x_2d[None,:]<cols)

    tile_data = tl.load(input+offsets_input,mask=mask_input)

    offsets_output = (offsets_y_2d*stride_oc)[None,:] + (offsets_x_2d*stride_or)[:,None]
    mask_output = (offsets_y_2d[None,:] < rows) & (offsets_x_2d[:,None]<cols)

    tl.store(output+offsets_output,tl.trans(tile_data),mask=mask_output)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = input.stride(0), input.stride(1)
    stride_or, stride_oc = output.stride(0), output.stride(1)

    BLOCK_M = 64
    BLOCK_K = 64
    grid = (triton.cdiv(rows,BLOCK_M), triton.cdiv(cols,BLOCK_K))
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc,
        BLOCK_M = BLOCK_M, BLOCK_K = BLOCK_K
    )


M = 7000
K = 6000
input = torch.randn(size=(M,K),device='cuda')
output = torch.randn(size=(K,M),device='cuda')

solve(input, output, M, K)

output_torch = input.T

# print(input)
# print(output)
torch.testing.assert_close(output_torch,output)
print("OK  max abs diff =", (output_torch - output).abs().max().item())

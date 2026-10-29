import torch
import triton
import triton.language as tl

import torch.nn.functional as F

torch.manual_seed(0)
# Take a look at DynamicOracle281 triton solution

@triton.jit
def conv1d_kernel(input, kernel, output, input_size, kernel_size, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE

    offsets_kernel = tl.arange(0,2048)
    mask_kernel = offsets_kernel < kernel_size

    kernel_ = tl.load(kernel+offsets_kernel,mask=mask_kernel,other=0)

    for i in range(0,BLOCK_SIZE):

        offsets_ip = block_start + i + tl.arange(0,2048)
        mask_ip = offsets_ip < input_size
        input_ = tl.load(input+offsets_ip,mask=mask_ip,other=0)

        sum_ = tl.sum(input_*kernel_,axis=0)

        offsets_op = block_start + i
        mask_op = offsets_op < input_size - kernel_size + 1
        tl.store(output+offsets_op,sum_,mask=mask_op)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel[grid](input, kernel, output, input_size, kernel_size, BLOCK_SIZE)


input = torch.randn((1_500_000,),device='cuda',dtype=torch.float32)[None,None,:]
kernel = torch.randn((2047,),device='cuda',dtype=torch.float32)[None,None,:]
output_triton = torch.empty((input.numel()-kernel.numel()+1),device='cuda',dtype=torch.float32)
solve(input,kernel,output_triton,input.numel(),kernel.numel())
output_torch = F.conv1d(input,kernel,padding=0).flatten()

print("torch_op:",output_torch[torch.tensor([475659,1029560])])
print("torch_op:",output_triton[torch.tensor([475659,1029560])])
torch.testing.assert_close(output_triton,output_torch)

print("OK  max abs diff =", (output_triton - output_torch).abs().max().item())
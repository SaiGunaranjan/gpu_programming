import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input+offsets,mask=mask)

    #zeros = tl.zeros_like(x)
    #y = tl.where(x<0, zeros, x)

    y = tl.maximum(x,0.0)

    tl.store(output+offsets,y,mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)


if __name__ == "__main__":
    
    N = 25_000_000
    input = torch.randn(N, device="cuda")
    output_triton = torch.full((N,), float('-inf'),device="cuda")
    output_torch = torch.relu(input)

    solve(input, output_triton, N)

    triton.testing.assert_close(output_triton, output_torch)
    print("OK  manual    max abs diff =", (output_triton - output_torch).abs().max().item())
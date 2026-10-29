import torch
import triton
import triton.language as tl


@triton.jit
def matrix_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets < n_elements

    a_ = tl.load(a+offsets,mask=mask,other=0)
    b_ = tl.load(b+offsets,mask=mask,other=0)

    tl.store(c+offsets,a_+b_,mask=mask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    matrix_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)

N = 4096
a = torch.randn((N,N), device='cuda', dtype=torch.float32)
b = torch.randn((N,N), device='cuda', dtype=torch.float32)
c = torch.full((N,N), float('-inf'), device='cuda', dtype=torch.float32)
# c = torch.zeros((N,N), device='cuda', dtype=torch.float32)
solve(a,b,c,N)
c_torch = a + b

torch.testing.assert_close(c, c_torch)
print("OK  max abs diff =", (c - c_torch).abs().max().item())

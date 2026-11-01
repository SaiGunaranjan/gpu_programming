import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, out_ptr,   # pointers to the three tensors
    n_elements,               # runtime size — a plain int
    BLOCK_SIZE: tl.constexpr,  # compile-time constant
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)

N = 25_000_000
a = torch.randn(N, device='cuda')
b = torch.randn(N, device='cuda')
c = torch.full((N,), float('-inf'), device='cuda')
solve(a,b,c,N)
c_torch = a + b

torch.testing.assert_close(c, c_torch)
print("OK  max abs diff =", (c - c_torch).abs().max().item())

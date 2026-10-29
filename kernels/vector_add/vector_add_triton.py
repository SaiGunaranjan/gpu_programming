import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
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


def add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out

x = torch.randn(100_000, device='cuda')
y = torch.randn(100_000, device='cuda')
out = add(x,y)
out_torch = x + y

torch.testing.assert_close(out, out_torch)
print("OK  max abs diff =", (out - out_torch).abs().max().item())

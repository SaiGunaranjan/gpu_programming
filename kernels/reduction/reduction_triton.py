import torch
import triton
import triton.language as tl


@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask = mask, other = 0)

    offsets_output = pid
    num_blocks = tl.num_programs(axis=0)
    mask_output = offsets_output < num_blocks

    # tl.store(output_ptr + offsets_output, tl.sum(x, axis = 0, dtype = tl.float32), mask = mask_output)
    tl.store(output_ptr + offsets_output, tl.sum(x, axis = 0), mask = mask_output)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    

    BLOCK_SIZE = 1024
    input_ = input
    while (N > BLOCK_SIZE):
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        output_ = torch.full(grid, float('-inf'), device = 'cuda')
        reduction_kernel[grid](input_, output_, N, BLOCK_SIZE = BLOCK_SIZE)
        input_ = output_
        N = input_.shape[0]

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    reduction_kernel[grid](input_, output, N, BLOCK_SIZE = BLOCK_SIZE)



if __name__ == '__main__':

    N = 4_194_304 # Performance with N = 4_194_304, max N = 100_000_000
    input = torch.randn(N, dtype = torch.float32, device = 'cuda')
    output_torch = torch.sum(input.double())
    output_triton = torch.full((1,), float('-inf'), device = 'cuda')
    solve(input, output_triton, N)

    triton.testing.assert_close(output_torch, output_triton.squeeze(), rtol=1e-5)
    print("OK  manual    max abs diff =", (output_triton.squeeze() - output_torch).abs().max().item())



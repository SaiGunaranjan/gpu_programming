import torch
import triton
import triton.language as tl
import numpy as np


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


def bench(fn, warmup_ms, reps_ms):
    """Times fn with do_bench; returns (median, min, mean) in ms per call."""
    rawtimes_ms = triton.testing.do_bench(fn, warmup=warmup_ms, rep=reps_ms, return_mode='all')
    return np.median(rawtimes_ms), np.min(rawtimes_ms), np.mean(rawtimes_ms)


def gb_per_s(num_bytes, ms):
    """Bytes moved / seconds, in GB/s (decimal GB, like NVIDIA's spec sheets)."""
    return num_bytes / (ms * 1e-3) / 1e9


def print_row(name, times_ms, bw, copy_bw, peak_spec):
    median_ms, min_ms, mean_ms = times_ms
    print(f"{name:<16}: {median_ms*1e3:.1f} / {min_ms*1e3:.1f} / {mean_ms*1e3:.1f} us {bw:.1f} GB/s "
          f"( {(bw/peak_spec)*100:.1f}% of spec, {(bw/copy_bw)*100:.1f}% of D2D copy)")



if __name__ == '__main__':

    N = 100_000_000 # Performance with N = 4_194_304, max N = 100_000_000
    torch.manual_seed(42)
    input = torch.randn(N, dtype = torch.float32, device = 'cuda')
    output_torch = torch.sum(input.double())
    output_triton = torch.full((1,), float('-inf'), device = 'cuda')
    solve(input, output_triton, N)

    triton.testing.assert_close(output_torch, output_triton.squeeze(), rtol=1e-5)
    print("OK  manual    max abs diff =", (output_triton.squeeze() - output_torch).abs().max().item())

    warmup_ms = 100
    reps_ms = 100

    copy_dst = torch.empty_like(input) # the copy needs a full-size destination; output_triton holds one float
    copy_times = bench(lambda: copy_dst.copy_(input), warmup_ms, reps_ms)
    manual_times = bench(lambda: solve(input, output_triton, N), warmup_ms, reps_ms)
    torch_times = bench(lambda: torch.sum(input), warmup_ms, reps_ms)

    copy_bytes = 2.0 * N * input.element_size() # read input, write copy_dst
    kern_bytes = 1.0 * N * input.element_size() # read input once; the single-float write is negligible
    peak_spec = 1008 # In GBps, RTX 4090: 10501 MHz x 2 (GDDR6X) x 48 B bus. Update when the pod's GPU changes

    copy_bw = gb_per_s(copy_bytes, copy_times[0])     # medians
    manual_bw = gb_per_s(kern_bytes, manual_times[0])
    torch_bw = gb_per_s(kern_bytes, torch_times[0])

    print(f"N={N}, warm-up {warmup_ms} ms, reps {reps_ms} ms (median / min / mean)")
    print_row("D2D copy", copy_times, copy_bw, copy_bw, peak_spec)
    print_row("triton 1024", manual_times, manual_bw, copy_bw, peak_spec)
    print_row("torch.sum", torch_times, torch_bw, copy_bw, peak_spec)



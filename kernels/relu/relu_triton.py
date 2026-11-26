import torch
import triton
import triton.language as tl
import numpy as np

# Search space for the autotuner (same as vector_add_triton.py).
BLOCK_SIZES = [2**x for x in range(7, 15)]   # 128 ... 16384 elements per program
NUM_WARPS = [1, 2, 4, 8, 16]                 # threads per program = num_warps x 32
CONFIGS = [triton.Config({'BLOCK_SIZE': bs}, num_warps=w) for bs in BLOCK_SIZES for w in NUM_WARPS]


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


# The same kernel wrapped in the autotuner (decorator syntax applied by hand), so
# relu_kernel itself stays usable with an explicit BLOCK_SIZE.
# An autotuned kernel can't be given BLOCK_SIZE at the call ("Conflicting meta-parameters").
# On the first call for each new n_elements it times every config and keeps the fastest.
relu_autotuned = triton.autotune(configs=CONFIGS, key=['n_elements'])(relu_kernel)


def solve_autotuned(input: torch.Tensor, output: torch.Tensor, N: int):
    """The autotuner picks BLOCK_SIZE and num_warps."""
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    relu_autotuned[grid](input, output, N)


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


if __name__ == "__main__":
    
    N = 25_000_000
    input = torch.randn(N, device="cuda")
    output_triton = torch.full((N,), float('-inf'),device="cuda")
    output_torch = torch.relu(input)

    solve(input, output_triton, N)

    triton.testing.assert_close(output_triton, output_torch)
    print("OK  manual    max abs diff =", (output_triton - output_torch).abs().max().item())

    # Correctness: autotuned. This first call also runs the autotuning (times all configs).
    output_triton = torch.full((N,), float('-inf'),device="cuda")
    solve_autotuned(input, output_triton, N)
    triton.testing.assert_close(output_triton, output_torch)
    print("OK  autotuned max abs diff =", (output_triton - output_torch).abs().max().item())
    print("autotuner picked:", relu_autotuned.best_config)

    warmup_ms = 100
    reps_ms = 100

    output = output_triton
    copy_times = bench(lambda: output.copy_(input), warmup_ms, reps_ms)
    manual_times = bench(lambda: solve(input, output, N), warmup_ms, reps_ms)
    autotuned_times = bench(lambda: solve_autotuned(input, output, N), warmup_ms, reps_ms)
    # torch baseline. torch.relu has no out= argument; clamp(min=0) with out=output writes into the
    # preallocated output like solve() does (torch.relu(input) would allocate a new tensor every call).
    torch_times = bench(lambda: torch.clamp(input, min=0.0, out=output), warmup_ms, reps_ms)

    copy_bytes = 2.0 * N * input.element_size() # read input, write output
    kern_bytes = 2.0 * N * input.element_size() # read input, write output
    peak_spec = 1008 # In GBps, RTX 4090: 10501 MHz x 2 (GDDR6X) x 48 B bus. Update when the pod's GPU changes

    copy_bw = gb_per_s(copy_bytes, copy_times[0])     # medians
    manual_bw = gb_per_s(kern_bytes, manual_times[0])
    autotuned_bw = gb_per_s(kern_bytes, autotuned_times[0])
    torch_bw = gb_per_s(kern_bytes, torch_times[0])

    print(f"N={N}, warm-up {warmup_ms} ms, reps {reps_ms} ms (median / min / mean)")
    print_row("D2D copy", copy_times, copy_bw, copy_bw, peak_spec)
    print_row("triton 1024/4", manual_times, manual_bw, copy_bw, peak_spec)
    print_row("triton autotuned", autotuned_times, autotuned_bw, copy_bw, peak_spec)
    print_row("torch clamp", torch_times, torch_bw, copy_bw, peak_spec)
import torch
import triton
import triton.language as tl
import numpy as np

# Search space for the autotuner and for the manual sweep in bench_configs.py.
BLOCK_SIZES = [2**x for x in range(7, 15)]   # 128 ... 16384 elements per program
NUM_WARPS = [1, 2, 4, 8, 16]                 # threads per program = num_warps x 32
CONFIGS = [triton.Config({'BLOCK_SIZE': bs}, num_warps=w) for bs in BLOCK_SIZES for w in NUM_WARPS]


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


# The same kernel wrapped in the autotuner (decorator syntax applied by hand), so
# vector_add_kernel itself stays usable with an explicit BLOCK_SIZE and num_warps.
# An autotuned kernel can't be given BLOCK_SIZE at the call ("Conflicting meta-parameters").
# On the first call for each new n_elements it times every config and keeps the fastest.
vector_add_autotuned = triton.autotune(configs=CONFIGS, key=['n_elements'])(vector_add_kernel)


def solve_autotuned(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    """The autotuner picks BLOCK_SIZE and num_warps."""
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    vector_add_autotuned[grid](a, b, c, N)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int, BLOCK_SIZE: int = 1024, num_warps: int = 4):
    """Fixed config. solve(a, b, c, N) is the LeetGPU signature; the extra arguments are for sweeps."""
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)


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
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    c_torch = a + b

    # Correctness: fixed config
    c = torch.full((N,), float('-inf'), device='cuda')
    solve(a,b,c,N)
    torch.testing.assert_close(c, c_torch)
    print("OK  manual    max abs diff =", (c - c_torch).abs().max().item())

    # Correctness: autotuned. This first call also runs the autotuning (times all configs).
    c = torch.full((N,), float('-inf'), device='cuda')
    solve_autotuned(a,b,c,N)
    torch.testing.assert_close(c, c_torch)
    print("OK  autotuned max abs diff =", (c - c_torch).abs().max().item())
    print("autotuner picked:", vector_add_autotuned.best_config)

    warmup_ms = 100
    reps_ms = 100

    copy_times = bench(lambda: c.copy_(a), warmup_ms, reps_ms)
    manual_times = bench(lambda: solve(a, b, c, N), warmup_ms, reps_ms)
    autotuned_times = bench(lambda: solve_autotuned(a, b, c, N), warmup_ms, reps_ms)
    # torch baseline. out=c writes into the preallocated c, like solve() does;
    # plain a + b would allocate a new output tensor on every call.
    torch_times = bench(lambda: torch.add(a, b, out=c), warmup_ms, reps_ms)

    copy_bytes = 2.0 * N * a.element_size() # read a, write c
    kern_bytes = 3.0 * N * a.element_size() # read a and b, write c
    peak_spec = 448.0 # In GBps, RTX A4000: 7001 MHz x 2 (GDDR6) x 32 B bus

    copy_bw = gb_per_s(copy_bytes, copy_times[0])     # medians
    manual_bw = gb_per_s(kern_bytes, manual_times[0])
    autotuned_bw = gb_per_s(kern_bytes, autotuned_times[0])
    torch_bw = gb_per_s(kern_bytes, torch_times[0])

    print(f"N={N}, warm-up {warmup_ms} ms, reps {reps_ms} ms (median / min / mean)")
    print_row("D2D copy", copy_times, copy_bw, copy_bw, peak_spec)
    print_row("triton 1024/4", manual_times, manual_bw, copy_bw, peak_spec)
    print_row("triton autotuned", autotuned_times, autotuned_bw, copy_bw, peak_spec)
    print_row("torch.add", torch_times, torch_bw, copy_bw, peak_spec)

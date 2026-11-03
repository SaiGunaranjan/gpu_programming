"""Manual config sweep for the Triton vector add at N = 25M: every BLOCK_SIZE x num_warps,
the same search space the autotuner uses, printed as a table so all configs are visible,
not only the winner.

Run: python kernels/vector_add/bench_configs.py
"""
import numpy as np
import torch
from vector_add_triton import BLOCK_SIZES, NUM_WARPS, solve, bench, gb_per_s

if __name__ == "__main__":
    warmup_ms = 100
    reps_ms = 100
    N = 25_000_000
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    c = torch.empty(N, device='cuda')
    kern_bytes = 3.0 * N * a.element_size() # read a and b, write c

    times_ms = np.zeros((len(BLOCK_SIZES), len(NUM_WARPS)))
    for i, blk_size in enumerate(BLOCK_SIZES):
        for j, num_warps in enumerate(NUM_WARPS):
            # The lambda reads blk_size and num_warps when it runs, not when it's created
            # (Python late binding). Fine here: bench() calls it before the loop moves on.
            times_ms[i, j] = bench(lambda: solve(a, b, c, N, blk_size, num_warps),
                                   warmup_ms, reps_ms)[0]

    print(f"N={N}: median us per call (elements per thread in brackets)")
    print(f"{'BLOCK_SIZE':>10} | " + " ".join(f"{f'{w} warps':>16}" for w in NUM_WARPS))
    for i, blk_size in enumerate(BLOCK_SIZES):
        cells = [f"{times_ms[i, j]*1e3:8.1f} ({blk_size / (32 * w):>5g})" for j, w in enumerate(NUM_WARPS)]
        print(f"{blk_size:>10} | " + " ".join(cells))

    best_i, best_j = np.unravel_index(np.argmin(times_ms), times_ms.shape)
    best_ms, worst_ms = times_ms.min(), times_ms.max()
    print(f"\nbest : BLOCK_SIZE = {BLOCK_SIZES[best_i]}, num_warps = {NUM_WARPS[best_j]}, "
          f"{best_ms*1e3:.1f} us, {gb_per_s(kern_bytes, best_ms):.1f} GB/s")
    print(f"worst: {worst_ms*1e3:.1f} us ({worst_ms / best_ms:.2f}x the best)")

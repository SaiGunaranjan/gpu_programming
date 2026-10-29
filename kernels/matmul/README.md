# matmul

C = A @ B.

Source: LeetGPU -- https://leetgpu.com/challenges/matrix-multiplication

**Naming.** The 2D-blocktiling kernel and the Triton kernels follow the LeetGPU problem: A is
M x N, B is N x K, C is M x K, so **N is the inner dimension**. The usual convention (and the one
`matmul_naive.cu`'s comments read most naturally in) is A (M x K) @ B (K x N).

FLOP count for the comparison table: `2 * M * N * K` (each of the M x K outputs needs N multiplies
and N adds).

## Variants

| File | Description | Status |
| --- | --- | --- |
| `matmul_naive.cu` | One thread per output element, all reads from global memory. Each element of A is re-read once per column of C, and each element of B once per row of C. | done |
| `matmul_2d_blocktiling.cu` | Shared-memory tiles of A and B (BM x BN and BN x BK); each thread computes a TM x TK block of C in registers (an outer product per k step). Default 64/64/64 tiles, 4x4 per thread, 256 threads. | correct for tile configs whose thread count divides the tiles; see open issues |
| `matmul_triton_naive.py` | One program per output element: loads a full row of A and column of B, `tl.sum` of the product. | done |
| `matmul_triton_tiled.py` | 64x64 output tile per program, loops over the inner dimension in steps of 32, `tl.dot` with `allow_tf32=False` (FP32 FMA, no Tensor Cores). | done |
| `matmul_torch.py` | `torch.matmul` / cuBLAS baseline. | todo |
| `bench.py` | Times every variant. | todo |

## Benchmark shapes

*To be filled in from the LeetGPU problem's benchmark configurations.*

## Performance

One row per revision. Time is per call, after warm-up; see the timing method in
[`../README.md`](../README.md#timing-method).

| Variant (file @ commit) | Change | M x N x K | Time (ms) | TFLOP/s | % of torch |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Open issues

`matmul_2d_blocktiling.cu`:

- [ ] **Overshooting loads write past the shared tiles.** When the thread count doesn't divide a
  tile, the last load pass covers slots past the end of it. The zero-fill `else` branches have no
  tile check (`new_local_row_A < BM`, and B's equivalent), so they write past `A_smem` (into
  `B_smem`) and past `B_smem`. This races with B's load and gives wrong C on some runs. Two ways to
  hit it: BM=48, BN=16, BK=40, TM=TK=4 (120 threads), or the power-of-2 config BM=16, BN=8, BK=16,
  TM=TK=1 (256 threads, 128-element tiles). `compute-sanitizer --tool racecheck` flags it every
  time; `memcheck` doesn't. Details: [`LEARNINGS_CUDA_CPP.md`](LEARNINGS_CUDA_CPP.md) §10, §12.
- [ ] **The launch is never checked.** `cudaGetLastError()` in `solve()` is commented out and its
  return value is not used, so a failed launch reports nothing.
- [ ] **The C check in `main` uses `!=`.** It works only because the test matrices are small
  integers, so every sum is exact. Random floats need a tolerance.

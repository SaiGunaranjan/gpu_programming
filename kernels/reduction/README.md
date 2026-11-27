# reduction

Sum of a 1-D float32 array: `output[0] = input[0] + ... + input[N-1]`.

Source: LeetGPU -- https://leetgpu.com/challenges/reduction

`solve(input, output, N)`, with `1 <= N <= 100,000,000` and values in [-1000, 1000]. LeetGPU checks
the result against a float64 `torch.sum` with `atol = rtol = 1e-5`, and creates `output` with
`torch.empty` (not zeroed).

Memory-bound: 1 add per 4 bytes read = 0.25 FLOP/byte, far below any GPU's ridge point. Unlike the
elementwise kernels, threads must combine their values (block-level tree, then across blocks).

## Variants

| File | Description | Status |
| --- | --- | --- |
| `reduction_triton.py` | Multi-pass: each program sums a `BLOCK_SIZE` (1024) slice with `tl.sum` and writes one partial sum; `solve()` launches passes over the partial sums until one program is left. Correct vs a float64 `torch.sum` (rtol 1e-5) up to N = 100M. | done, not optimal (see below) |
| `reduction_smem_tree.cu` | Each block sums 1024 elements with 128 threads: every thread accumulates into its own shared-memory slot, then a shared-memory tree reduction (sequential addressing; needs a power-of-2 thread count), then one `atomicAdd` per block into `output[0]`. | done |
| `reduction_smem_atomic.cu` | Same block layout, but every thread `atomicAdd`s each element into one shared-memory float, then one `atomicAdd` per block into `output[0]`. Kept as the contention example (see `LEARNINGS_CUDA_CPP.md` §3). | done |
| `reduction_smem_tree_warp_shuffle.cu` | `reduction_smem_tree.cu` with the last 5 tree rounds (stride 16 … 1, inside one warp) done with `__shfl_down_sync` instead of shared memory + `__syncthreads()`. First use of a shuffle: the values still go through the 128-float shared-memory array. Matches the tree version bit for bit per block; same speed (0.421 ms at N = 100M). See `LEARNINGS_CUDA_CPP.md` §5. | done, not optimal |
| CUDA C++, next | Register accumulator (`LEARNINGS_CUDA_CPP.md` §1); register-only warp shuffle (shuffle the register sum within each warp, one shared-memory slot per warp, one more shuffle in warp 0); grid-stride with many elements per thread. | todo |
| torch baseline | `torch.sum(input)`. | todo |

## Benchmark shape

N = 4,194,304 (2^22) elements (LeetGPU's), float32: 16 MB read per call.

**This size doesn't measure DRAM bandwidth.** 16 MB fits in the L2 cache (72 MB on the RTX 4090),
so timing it back to back without flushing L2 measures L2 hits (the tree kernel shows "135% of
spec"); with `do_bench`'s L2 flush it's too small a transfer to reach full DRAM speed. Use
**N = 100,000,000** (400 MB) for DRAM bandwidth. Both sizes, all variants:
[`LEARNINGS_CUDA_CPP.md`](LEARNINGS_CUDA_CPP.md) §4.

## Open issues / later

- [ ] Triton: replace the multi-pass `solve()`. Every pass allocates a new tensor with
  `torch.full` and launches a kernel from Python, so the host side costs more than the GPU work.
  Planned: a persistent (grid-stride) kernel where each program loops over many blocks into a
  vector accumulator and calls `tl.sum` once, then a fixed second pass over a preallocated
  partials buffer (2 launches, no allocation per call, deterministic).
- [ ] Triton alternative to compare (to revisit): instead of calling the kernel in a loop until the
  tensor is smaller than `BLOCK_SIZE`, launch once and have each program `tl.atomic_add` its
  `tl.sum` into `output[0]`, like the CUDA kernels' one `atomicAdd` per block (needs `output`
  zeroed first; the order of the adds, and so the last bits, can change run to run).
- [ ] CUDA C++ variants (see the Variants table).
- [x] Timing: GPU time (cudaEvents in the `.cu` files, `do_bench` in `reduction_triton.py`), added
  2026-11-26.
- [ ] Timing: back-to-back wall time per call (`do_bench` hides the multi-pass `solve()`'s host
  overhead: ~118 us per call vs ~36 us of GPU work at N = 4,194,304, measured 2026-11-17).

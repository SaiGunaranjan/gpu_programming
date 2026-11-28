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
| `reduction_warp_shuffle.cu` | Register-only warp shuffle: each thread sums its 8 elements in a register (`LEARNINGS_CUDA_CPP.md` §1), every warp shuffles its register sum down to lane 0, lane 0 writes it to `warp_partialsum[4]` (one slot per warp), one `__syncthreads()`, then warp 0 shuffles those 4 values and does one `atomicAdd` per block. Same speed as the tree versions. See `LEARNINGS_CUDA_CPP.md` §6. | done; fails tolerance at 100M (too many `atomicAdd`s, §2) |
| CUDA C++, next | Grid-stride: a fixed number of blocks, each summing many elements, so far fewer `atomicAdd`s go into `output[0]` (the fix for §2). | todo |
| torch baseline (in `reduction_triton.py`) | `torch.sum(input)`, timed with `do_bench`. | done, timed |

## Benchmark shape

**N = 100,000,000 floats** (400 MB read per call; the largest N LeetGPU allows), in the `main` of
every file. The input uses a fixed seed (42), so every run sums the same numbers.

> **Note: why not LeetGPU's N = 4,194,304?** LeetGPU times this problem at N = 4,194,304 (2^22),
> but that is only 16 MB, which **fits in the L2 cache** (72 MB on the RTX 4090). Timed back to
> back, every call after the first reads the input from L2, so the result is **not the true DRAM
> speed** (the kernels show ~135-148% of the DRAM spec, which DRAM can't deliver). With an L2 flush
> (`do_bench`) it's too small a transfer to reach full DRAM speed. So we **shifted to N = 100M**
> (400 MB, far bigger than L2): it measures real DRAM bandwidth, and "% of spec" reads correctly.
> Both sizes, all variants: [`LEARNINGS_CUDA_CPP.md`](LEARNINGS_CUDA_CPP.md) §4.

## Performance

One row per revision; see the timing method in [`../README.md`](../README.md#timing-method).
Bytes moved = N x 4 B (the input is read once; the single-float write is negligible). Times are
medians; each variant was run 3 times (the Triton script twice) and the runs agreed to within 1 us.
Measured 2026-11-28.

### Measured on

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, compute capability 8.9 (sm_89) |
| SMs, L2 | 128 SMs, 72 MB L2 |
| Memory | 24 GB GDDR6X, 384-bit bus, 10501 MHz |
| Theoretical peak DRAM BW | 10501 MHz x 2 x 48 B = **1,008 GB/s** |
| Practical peak (D2D `cudaMemcpy`) | **917 GB/s** (91.0% of spec; 0.8724 ms median for 100M floats, read + write) |
| Software | CUDA 12.8 (nvcc 12.8.93), driver 595.112.07, torch 2.8.0+cu128, Triton 3.4.0 |

| Variant (file @ commit) | Change | Launch config | N | Time (ms) | GB/s | % of spec | % of D2D copy | % of torch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reduction_smem_tree.cu` @ `a7396d1` | first timed version | 1024 floats/block, 128 threads | 100,000,000 | 0.4219 | 948 | 94.1% | 103.4% | 114.6% |
| `reduction_smem_atomic.cu` @ `a7396d1` | first timed version | 1024 floats/block, 128 threads | 100,000,000 | 2.5917 | 154 | 15.3% | 16.8% | 18.6% |
| `reduction_smem_tree_warp_shuffle.cu` @ `4dbea0d` | last 5 tree rounds via `__shfl_down_sync` | 1024 floats/block, 128 threads | 100,000,000 | 0.4219 | 948 | 94.1% | 103.4% | 114.6% |
| `reduction_warp_shuffle.cu` @ uncommitted | register sum + warp shuffle, one smem slot per warp | 1024 floats/block, 128 threads (4 warps) | 100,000,000 | 0.4219 | 948 | 94.1% | 103.4% | 114.6% |
| `reduction_triton.py` @ `a7396d1` | multi-pass `tl.sum` | BLOCK_SIZE 1024, 4 warps (default) | 100,000,000 | 0.4879 | 820 | 81.3% | 89.7% | 99.1% |
| torch `torch.sum` | baseline | - | 100,000,000 | 0.4833 | 828 | 82.1% | 90.6% | 100% |

- **% of torch** = torch time / variant time (above 100% = faster than torch).
- **% of D2D copy** uses each script's own copy: the CUDA C++ rows use `cudaMemcpy` timed with
  `time_gpu()` (917 GB/s); the Triton and torch rows use `copy_dst.copy_(input)` timed with
  `do_bench` (914 GB/s, L2 flushed). Above 100% is normal for a reduction (`LEARNINGS_CUDA_CPP.md` §4).
- **The three tree / shuffle kernels run at the same speed** (94.1% of spec, ~13 points above
  `torch.sum`). At this size they are limited by DRAM reads, so removing shared-memory traffic and
  `__syncthreads()` doesn't show up in the time.
- **Tolerance at 100M:** with seed 42 the sum is -4.89e6, so the LeetGPU tolerance is ~49. The
  CUDA kernels' error is 15-51 and changes run to run (the order of the `atomicAdd`s changes), so a
  run can print "Wrong" (the tree kernel did in 1 of 3 runs). See `LEARNINGS_CUDA_CPP.md` §2.
- At LeetGPU's N = 4,194,304, all three fast CUDA kernels and the `cudaMemcpy` show the same
  ~11.3 us (L2-resident, at a launch/timer floor), so that size can't separate them. All variants
  at both sizes: `LEARNINGS_CUDA_CPP.md` §4.

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

# vector_add

Elementwise C = A + B over a 1-D array.

Source: LeetGPU -- https://leetgpu.com/challenges/vector-addition

Memory-bound, same as `matrix_add` -- compare GB/s, not TFLOP/s. Intensity: 1 add per 12 bytes
(read A, read B, write C) = 0.083 FLOP/byte, far below the RTX A4000's ridge point of ~43.

## Variants

| File | Description | Status |
| --- | --- | --- |
| `vector_add.cu` | 1-D grid, one thread per element. `main()` checks against the CPU, then times the kernel and a device-to-device copy with `time_gpu()` (CUDA events), then sweeps threads per block 32..1024. | done |
| `vector_add_triton.py` | 1-D grid of programs, masked loads/stores for the tail. `solve()`: fixed config, `BLOCK_SIZE=1024`, `num_warps=4` by default. `solve_autotuned()`: the same kernel under `@triton.autotune` (BLOCK_SIZE 128..16384 x num_warps 1..16). Times both, `torch.add(out=c)` and a copy with `triton.testing.do_bench`. | done |
| `bench_configs.py` | Manual sweep of the autotuner's search space at 25M elements, printed as a table. | done |
| torch baseline | `torch.add(a, b, out=c)`, timed in `vector_add_triton.py`. | done |

## Benchmark shape

N = 25,000,000 elements (LeetGPU's), float32: 300 MB moved per call, 670 us at the 448 GB/s spec.

## Performance

RTX A4000 (sm_86), CUDA 12.8, driver 595.91.07, torch 2.8.0+cu128, Triton 3.4.0. Median time per
call; see the timing method in [`../README.md`](../README.md#timing-method). The copy is the
practical peak (2 x N x 4 bytes moved).

| Variant (file @ commit) | Change | N | Time (us) | GB/s | % of 448 spec | % of torch |
| --- | --- | --- | --- | --- | --- | --- |
| D2D copy (`cudaMemcpy` / `c.copy_(a)`) | reference | 25M | 518 | 386 | 86 | - |
| `vector_add.cu` @ `2b6fcf7` | 256 threads/block | 25M | 763 | 393 | 88 | 100.4 |
| `vector_add.cu` @ `2b6fcf7` | best of sweep: 1024 threads/block | 25M | 754 | 398 | 89 | 101.6 |
| `vector_add_triton.py` @ `2b6fcf7` | fixed 1024 / 4 warps | 25M | 767 | 391 | 87 | 99.9 |
| `vector_add_triton.py` @ `2b6fcf7` | autotuned: 128 / 8 warps | 25M | 756 | 397 | 89 | 101.4 |
| torch `add(out=c)` | baseline | 25M | 766 | 392 | 87 | 100 |

The CUDA C++ rows use `time_gpu()` (no L2 flush); the rest use `do_bench` (L2 flushed). At 300 MB
this makes no difference.

**Findings** (details in [`../../learnings/PROFILING_GENERAL.md`](../../learnings/PROFILING_GENERAL.md)):
- Every reasonable config saturates DRAM. 36 of 40 Triton configs and CUDA 64..1024 threads/block
  are within ~2%; tuning buys 1-2%.
- Slow corners: Triton with 256+ elements per thread spills registers (1.5-2.8x slower); Triton
  128 / 16 warps has 3/4 of its threads idle (902 us); CUDA 32 threads/block is capped at 16 warps
  per SM by the 16-block limit (793 us).
- Size sweep (2^10..2^28): launch-bound below ~512K elements (Python launch ~20 us for Triton,
  ~6 us for torch), L2 visible around 256K-512K, DRAM plateau (~392 GB/s) from ~4M elements.

## Open issues / later

- [ ] An elements-per-thread CUDA C++ variant (`template <int ELEMS>`, strided by `blockDim.x` so
  loads stay coalesced), to match the Triton table; then a `float4` variant.

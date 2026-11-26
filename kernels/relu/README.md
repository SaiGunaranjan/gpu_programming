# relu

Output = ReLU(Input), element-wise on N 32-bit floats: `output[i] = max(0, input[i])`.

Source: LeetGPU -- https://leetgpu.com/challenges/relu

Memory-bound, with almost no arithmetic: every element is read once and written once, so the
target is the device-to-device copy. Compare achieved bandwidth (GB/s). The lessons here are
`float4` vectorized loads/stores, tail handling when N isn't a multiple of 4, and how the compiler
turns the `if` into a branch-free select (see
[`learnings/CUDA_CPP_GENERAL.md`](../../learnings/CUDA_CPP_GENERAL.md) §10).

## Variants

| File | Description | Status |
| --- | --- | --- |
| `relu_naive.cu` | CUDA C++, scalar: each block covers `BLOCK_SIZE` = 1024 floats with 256 threads, so each thread handles 4 floats, 256 apart (a block-stride loop; reads and writes coalesced). | done, timed |
| `relu_float4.cu` | CUDA C++, `float4`: the same 1024 floats per block viewed as 256 `float4`s, one 16-byte load and one 16-byte store per thread (`LDG.E.128` / `STG.E.128`). The thread whose `float4` index equals `N/4` writes the last `N % 4` floats with scalar accesses. | done, timed |
| `relu_triton.py` | One program per `BLOCK_SIZE` slice, `tl.maximum(x, 0.0)`. `solve()`: fixed config (1024, 4 warps by default). `solve_autotuned()`: the same kernel under `triton.autotune` (BLOCK_SIZE 128..16384 x 1..16 warps, as in vector add). Times both, torch and a copy with `do_bench`. | done, autotuned, timed |
| torch baseline (in `relu_triton.py`) | `torch.clamp(input, min=0.0, out=output)`: writes into a preallocated output like `solve()` does (`torch.relu` has no `out=` argument and would allocate a new tensor every call). | done, timed |

## Benchmark shape

N = 25,000,000 floats (LeetGPU's): 100 MB in, 100 MB out, 200 MB moved per call.
`relu_float4.cu` runs N = 25,000,003 so its scalar tail is exercised (3 more floats, negligible).

## Performance

One row per revision; see the timing method in [`../README.md`](../README.md#timing-method).
Bytes moved = 2 x N x 4 B (each element read once, written once). Times are medians; each variant
was run 3 times and the runs agreed to within 1 us.

### Measured on

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, compute capability 8.9 (sm_89) |
| SMs, L2 | 128 SMs, 72 MB L2 |
| Memory | 24 GB GDDR6X, 384-bit bus, 10501 MHz |
| Theoretical peak DRAM BW | 10501 MHz x 2 x 48 B = **1,008 GB/s** |
| Practical peak (D2D `cudaMemcpy`) | **900 GB/s** (89.3% of spec; 0.2222 ms median for 25M floats) |
| Software | CUDA 12.8 (nvcc 12.8.93), driver 595.112.07, torch 2.8.0+cu128, Triton 3.4.0 |

| Variant (file @ commit) | Change | Launch config | N | Time (ms) | GB/s | % of spec | % of D2D copy | % of torch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `relu_naive.cu` @ `24c58bc` | first timed version | 1024 floats/block, 256 threads | 25,000,000 | 0.2212 | 904 | 89.7% | 100.5% | 101.4% |
| `relu_float4.cu` @ `24c58bc` | first timed version; `float4` loads/stores + scalar tail | 1024 floats/block, 256 threads, 1 `float4` per thread | 25,000,003 | 0.2202 | 908 | 90.1% | 100.9% | 101.9% |
| `relu_triton.py` @ `24c58bc` | fixed config | BLOCK_SIZE 1024, 4 warps | 25,000,000 | 0.2253 | 888 | 88.1% | 99.5% | 99.6% |
| `relu_triton.py` @ `24c58bc` | autotuned (40 configs) | best: BLOCK_SIZE 128, 4 warps (one run picked 256 / 8 warps, same time) | 25,000,000 | 0.2202 | 908 | 90.1% | 101.9% | 101.9% |
| torch `clamp(min=0, out=)` | baseline | - | 25,000,000 | 0.2243 | 892 | 88.5% | 100.0% | 100% |

- **% of torch** = torch time / variant time (above 100% = faster than torch).
- **% of D2D copy** uses each script's own copy: the CUDA C++ rows use `cudaMemcpy` timed with
  `time_gpu()` (900 GB/s); the Triton and torch rows use `output.copy_(input)` timed with
  `do_bench` (892 GB/s, L2 flushed). The two methods differ by about 1% here, so differences
  under ~1% between a CUDA C++ row and a Triton/torch row are within method noise.
- **Every variant runs at copy speed.** ReLU is a copy with a compare, so all of them, torch
  included, are limited by DRAM bandwidth (~89-90% of spec).
- **`float4` vs scalar is within noise on the RTX 4090** (0.2202 vs 0.2212 ms). The scalar loads
  are already coalesced, so wider accesses move the same bytes with fewer instructions, and that
  isn't the bottleneck at this size.
- The autotuned Triton kernel is ~2% faster than the fixed 1024 / 4-warp config; the autotuner
  prefers small blocks (128-256 elements per program).

## Open issues

- None for correctness. A grid-stride variant (a fixed number of blocks, each looping over many
  elements) is planned as a learning exercise when grid-stride loops are covered.

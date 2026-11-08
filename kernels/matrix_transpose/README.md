# matrix_transpose

Output = Input^T: an input of `rows x cols` 32-bit floats becomes an output of `cols x rows`, with
`output[j][i] = input[i][j]`.

Source: LeetGPU -- https://leetgpu.com/challenges/matrix-transpose

Memory-bound, with no arithmetic at all: every element is read once and written once. Compare
achieved bandwidth (GB/s). The interesting part is the access pattern: reading rows and writing
columns can't both be coalesced unless the tile is staged (in shared memory in CUDA C++, or by a
layout conversion in Triton).

## Variants

| File | Description | Status |
| --- | --- | --- |
| `matrix_transpose_naive.cu` | CUDA C++, one thread per element: reads are coalesced, writes are strided by `rows`. | done |
| `matrix_transpose_smem.cu` | CUDA C++, 64 x 64 tile staged in shared memory (padded +1 column against bank conflicts), 16 x 16 threads, each thread handles 4 x 4 elements; both reads and writes coalesced. | done |
| `matrix_transpose_triton.py` | 2-D grid of BLOCK_M x BLOCK_K tiles: load a tile with input strides, `tl.trans` it, store it with output strides. Masks handle edge tiles. `solve()`: fixed config (64 x 64, 4 warps by default). `solve_autotuned()`: the same kernel under `triton.autotune`. Times both, torch and a copy with `do_bench`. | done |
| torch baseline (in `matrix_transpose_triton.py`) | `output.copy_(input.transpose(0, 1))`, timed with `do_bench` in the Triton script; no separate torch file. | done |

## Benchmark shapes

LeetGPU's benchmark configuration: **rows = 7000, cols = 6000** (42M floats, 168 MB in, 168 MB out).
All timings below use this shape.

## Performance

One row per revision; see the timing method in [`../README.md`](../README.md#timing-method).
Bytes moved = 2 x rows x cols x 4 B (each element read once, written once).

### Measured on

| | |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB, compute capability 8.0 (sm_80) |
| SMs, L2 | 108 SMs, 40 MB L2 |
| Memory | 80 GB HBM2e, 5120-bit bus, 1593 MHz |
| Theoretical peak DRAM BW | 1593 MHz x 2 x 640 B = **2,039 GB/s** |
| Practical peak (D2D `cudaMemcpy`) | **1,667 GB/s** (81.8% of spec; 0.2015 ms median for 7000 x 6000) |
| Software | CUDA 12.8 (nvcc 12.8.93), driver 595.91.07, torch 2.8.0+cu128, Triton 3.4.0 |

| Variant (file @ commit) | Change | Launch config | rows x cols | Time (ms) | GB/s | % of spec | % of D2D copy | % of torch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `matrix_transpose_smem.cu` @ `106aff6` | first timed version; `cudaDeviceSynchronize()` moved out of `solve` so it isn't timed | 64 x 64 tile, 16 x 16 threads | 7000 x 6000 | 0.2340 | 1,436 | 70.4% | 86.2% | 212.6% |
| `matrix_transpose_triton.py` @ `6fed2da` | fixed config, same tile and thread count as the smem kernel | 64 x 64 tile, 8 warps (256 threads) | 7000 x 6000 | 0.2253 | 1,491 | 73.1% | 90.2% | 220.9% |
| `matrix_transpose_triton.py` @ `6fed2da` | autotuned (60 configs: tiles 16..128 per side, 1..16 warps, 4..64 elements per thread) | best: 32 x 128 tile, 16 warps | 7000 x 6000 | 0.2109 | 1,593 | 78.1% | 96.3% | 235.9% |
| torch `output.copy_(input.transpose(0, 1))` | baseline | - | 7000 x 6000 | 0.4976 | 675 | 33.1% | 40.8% | 100% |

- **% of torch** = torch time / variant time (above 100% = faster than torch).
- **% of D2D copy** uses each script's own copy: the CUDA C++ rows use `cudaMemcpy` timed with
  `time_gpu()` (1,667 GB/s); the Triton and torch rows use `output.copy_(input)` timed with
  `do_bench` (1,653 GB/s, L2 flushed). At 336 MB moved the difference is under 1%.
- The Triton kernel beats the smem CUDA C++ kernel at the same 64 x 64 tile and 256 threads (73.1%
  vs 70.4%), and the autotuned rectangular 32 x 128 tile gets within 4% of the copy.
- torch's `copy_` of a transposed view is far slower than any tiled kernel here (33% of spec).

### Why the D2D copy isn't an apples-to-apples target

The device-to-device `cudaMemcpy` moves the same number of bytes as a transpose, but it isn't a fair
ceiling for these kernels:

- **Access width.** `cudaMemcpy` is NVIDIA's tuned copy and moves data in wide chunks (up to 16 B per
  access). Our kernels move one 4-byte `float` per thread per access, so fewer bytes are in flight
  per memory instruction. Even a plain copy kernel with 4-byte accesses falls clearly short of
  `cudaMemcpy`; with 16-byte (`float4`) accesses it matches or beats it.
- **No reordering.** A copy reads and writes the same contiguous addresses. A transpose has to
  regroup the data (via shared memory, plus a `__syncthreads()` per tile), which costs a little
  more even when both reads and writes are coalesced.

So "% of D2D copy" is a useful reference, but 100% isn't reachable with 4-byte accesses. Closing
the gap means wider (`float4`) loads and stores, which would be a new variant.

## Open issues

*None yet.*

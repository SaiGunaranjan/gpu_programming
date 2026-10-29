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
| `matrix_transpose.cu` | CUDA C++ kernel. | todo |
| `matrix_transpose_triton.py` | 2-D grid of 64 x 64 tiles: load a tile with input strides, `tl.trans` it, store it with output strides. Masks handle edge tiles. | done |
| `matrix_transpose_torch.py` | torch baseline (`input.T.contiguous()`). | todo |

## Benchmark shapes

*To be filled in from the LeetGPU problem's benchmark configurations.*

## Performance

One row per revision; see the timing method in [`../README.md`](../README.md#timing-method).

| Variant (file @ commit) | Change | rows x cols | Time (ms) | GB/s | % of torch |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Open issues

*None yet.*

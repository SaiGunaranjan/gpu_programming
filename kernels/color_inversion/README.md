# color_inversion

Invert the colours of an RGBA image in place: each of R, G, B becomes `255 - value`, and alpha is
left unchanged. The image is `width x height` pixels, stored as one flat array of
`width * height * 4` `uint8` values (R, G, B, A, R, G, B, A, ...).

Source: LeetGPU -- https://leetgpu.com/challenges/color-inversion

Memory-bound: one byte read and one byte written per channel, one subtract. Compare achieved
bandwidth (GB/s), not FLOP/s.

## Variants

| File | Description | Status |
| --- | --- | --- |
| `color_inversion.cu` | CUDA C++ kernel. | todo |
| `color_inversion_triton.py` | 1-D grid; each program handles `BLOCK_SIZE=1024` pixels (4096 bytes), inverts every byte and keeps the alpha bytes (`offsets % 4 == 3`) with `tl.where`. | done |
| `color_inversion_torch.py` | torch baseline. | todo |

## Benchmark shapes

*To be filled in from the LeetGPU problem's benchmark configurations.*

## Performance

One row per revision; see the timing method in [`../README.md`](../README.md#timing-method).

| Variant (file @ commit) | Change | width x height | Time (ms) | GB/s | % of torch |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Open issues

- [ ] The script prints both images instead of asserting that they match.

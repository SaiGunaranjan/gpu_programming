# conv1d

1-D convolution with no padding ("valid" mode): for an input of `input_size` floats and a filter of
`kernel_size` floats, `output[i] = sum_j input[i + j] * kernel[j]` for
`i = 0 .. input_size - kernel_size`. (Strictly this is cross-correlation, as in `F.conv1d`: the
filter is not flipped.)

Source: LeetGPU -- https://leetgpu.com/challenges/1d-convolution

Each input element is used by up to `kernel_size` outputs, so with large filters (up to 2047 here)
this is compute-heavy rather than memory-bound: `2 * kernel_size` FLOPs per output. Reuse of the
input through shared memory is the main optimization lever.

## Variants

| File | Description | Status |
| --- | --- | --- |
| `conv1d_smem_input.cu` | One thread per output, 256 threads per block. The block's input tile (256 + `kernel_size` - 1 elements) is staged in **static** shared memory, sized for the maximum filter (`MAX_KERNEL_SIZE` = 2047); the filter is read from global memory. | done |
| `conv1d_smem_input_filter.cu` | Same mapping, but **dynamic** shared memory (`extern __shared__`, sized at launch) holds both the input tile **and** the filter, so the compute loop reads only shared memory. | done |
| `conv1d_triton_per_output.py` | Each program owns `BLOCK_SIZE=1024` outputs and loops over them one at a time: for each output it loads a 2048-wide input window, multiplies by the filter and `tl.sum`s it. Simple but not optimal: every output reloads its whole window. Filters are limited to 2048 taps (the `tl.arange(0, 2048)`). | done |
| `conv1d_triton_*.py` | An optimized Triton version. | todo |
| `conv1d_torch.py` | `F.conv1d` baseline. | todo |

## Benchmark shapes

*To be filled in from the LeetGPU problem's benchmark configurations.*

## Performance

One row per revision; see the timing method in [`../README.md`](../README.md#timing-method).

| Variant (file @ commit) | Change | input_size, kernel_size | Time (ms) | GFLOP/s | % of torch |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Open issues

- [ ] Both `.cu` files compare the GPU and CPU results with exact `!=`, so a normal build prints
  ~75 last-bit "mismatches" (FMA rounding). With `-fmad=false` there are 0, which shows the kernels
  are correct. A relative tolerance would make the default build quiet.
- [ ] `conv1d_triton_per_output.py`'s own `assert_close` against `F.conv1d` fails at the default
  float32 tolerance. That's torch's error, not Triton's: see
  [`LEARNINGS_TRITON.md`](LEARNINGS_TRITON.md). Compare against a float64 reference, or loosen the
  tolerance.

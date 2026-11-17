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
| CUDA C++ | Shared-memory tree, sequential addressing, warp shuffles, grid-stride with many elements per thread. | todo |
| torch baseline | `torch.sum(input)`. | todo |

## Benchmark shape

N = 4,194,304 (2^22) elements (LeetGPU's), float32: 16 MB read per call.

## Open issues / later

- [ ] Triton: replace the multi-pass `solve()`. Every pass allocates a new tensor with
  `torch.full` and launches a kernel from Python, so the host side costs more than the GPU work.
  Planned: a persistent (grid-stride) kernel where each program loops over many blocks into a
  vector accumulator and calls `tl.sum` once, then a fixed second pass over a preallocated
  partials buffer (2 launches, no allocation per call, deterministic).
- [ ] Triton alternative to compare: one launch with `tl.atomic_add` into `output[0]` (needs
  `output` zeroed first; the order of the adds, and so the last bits, can change run to run).
- [ ] CUDA C++ variants (see the Variants table).
- [ ] Timing: GPU time (`do_bench` / cudaEvents) and back-to-back wall time per call.

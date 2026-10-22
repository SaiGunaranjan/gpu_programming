# matmul

C = A @ B, with A (M x K) and B (K x N).

Source: LeetGPU -- https://leetgpu.com/challenges/matrix-multiplication

## Variants

| File | Description | Status |
| --- | --- | --- |
| `matmul_naive.cu` | One thread per output element, all reads from global memory. Each element of A is re-read N times and each element of B is re-read M times. | done |
| `matmul_tiled.cu` | Shared-memory tiling to cut the redundant global loads. | todo |
| `matmul_triton.py` | Triton kernel. | todo |
| `matmul_torch.py` | `torch.matmul` / cuBLAS baseline. | todo |

FLOP count for the comparison table: `2 * M * N * K`.

# vector_add

Elementwise C = A + B over a 1-D array.

Source: LeetGPU -- https://leetgpu.com/challenges/vector-addition

Memory-bound, same as `matrix_add` -- compare GB/s, not TFLOP/s.

## Variants

| File | Description | Status |
| --- | --- | --- |
| `vector_add.cu` | 1-D grid, one thread per element. | done |
| `vector_add_chatgpt.cu` | Generated reference version, kept for comparison. Untracked. | - |
| `vector_add_triton.py` | Triton kernel. | todo |
| `vector_add_torch.py` | `A + B` baseline. | todo |

# matrix_add

Elementwise C = A + B over a 2-D grid.

Source: LeetGPU -- https://leetgpu.com/challenges/matrix-addition

Memory-bound: 3 floats moved per output element, 1 add. The useful number here
is achieved bandwidth (GB/s) against the card's peak, not TFLOP/s.

## Variants

| File | Description | Status |
| --- | --- | --- |
| `matrix_add.cu` | 2-D grid, one thread per element. | done |
| `matrix_add_triton.py` | Triton kernel. | todo |
| `matrix_add_torch.py` | `A + B` baseline. | todo |

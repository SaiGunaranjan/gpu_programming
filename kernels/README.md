# kernels

One folder per operator. Every implementation of that operator -- CUDA C++,
Triton, and the PyTorch baseline -- lives together in its folder, alongside the
benchmark that times all three against each other.

```
kernels/<operator>/
    <operator>.cu           CUDA C++ (one file per variant: _naive, _tiled, ...)
    <operator>_triton.py    Triton kernel
    <operator>_torch.py     PyTorch / cuBLAS baseline
    bench.py                times every variant, prints the comparison table
    README.md               notes, and where the problem came from
```

The folder name is the operator, not where the problem came from. LeetGPU
problems get their URL recorded in the operator's README, so a kernel can later
grow variants that have nothing to do with the original exercise without
needing to move.

## Building the CUDA C++ kernels

Each `.cu` file is standalone and contains its own `main()`:

```
nvcc kernels/matmul/matmul_naive.cu -o matmul_naive
./matmul_naive
```

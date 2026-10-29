# kernels

One folder per operator. Every implementation of that operator (CUDA C++, Triton, and the
PyTorch baseline) lives together in its folder, alongside the benchmark that times them against
each other. Each problem is also a learning exercise: how it was coded, what went wrong, and what
it taught about the GPU.

## Operators

| operator | source | CUDA C++ | Triton | torch | branch |
| --- | --- | --- | --- | --- | --- |
| [`vector_add`](vector_add/) | [LeetGPU](https://leetgpu.com/challenges/vector-addition) | done | written, not in repo yet | todo | `18-vector-addition--triton-vs-cuda-cpp-vs-torch` |
| [`matrix_add`](matrix_add/) | [LeetGPU](https://leetgpu.com/challenges/matrix-addition) | done | written, not in repo yet | todo | `19-matrix-addition-triton-vs-cuda-cpp-vs-torch` |
| [`matmul`](matmul/) | [LeetGPU](https://leetgpu.com/challenges/matrix-multiplication) | naive done; 2D blocktiling in progress | written, not in repo yet | todo | `17-matmul-triton-vs-cuda-cpp-vs-torch` |
| `conv1d` | [LeetGPU](https://leetgpu.com/challenges/1d-convolution) | written, not in repo yet | written, not in repo yet | todo | `20-1d-convolution-triton-vs-cuda-cpp-vs-torch` |
| `matrix_transpose` | [LeetGPU](https://leetgpu.com/challenges/matrix-transpose) | todo | written, not in repo yet | todo | `21-matrix-transpose-triton-vs-cuda-cpp-vs-torch` |
| `color_inversion` | [LeetGPU](https://leetgpu.com/challenges/color-inversion) | todo | written, not in repo yet | todo | `22-color-inversion-triton-vs-cuda-cpp-vs-torch` |

Each operator is developed on its own branch and merged to `master` through a PR. Changes that
affect every operator (this file, `.gitignore`, [`../learnings/`](../learnings/)) go to `master`
first; operator branches pick them up with `git merge master`.

## Folder layout

```
kernels/<operator>/
    <operator>_<variant>.cu     CUDA C++, one file per variant (_naive, _2d_blocktiling, ...)
    <operator>_triton.py        Triton kernel
    <operator>_torch.py         PyTorch baseline (cuBLAS for matmul)
    bench.py                    times every variant, prints the comparison table
    README.md                   the problem and its performance
    LEARNINGS_CUDA_CPP.md       what this problem taught, CUDA C++ side
    LEARNINGS_TRITON.md         what this problem taught, Triton side
```

The folder name is the operator, not where the problem came from. LeetGPU problems get their URL
recorded in the operator's README, so a kernel can later grow variants that have nothing to do
with the original exercise without needing to move.

**Variants:** bug fixes and cleanups change the same file; git history keeps the old versions. A
new optimization technique (vectorized loads, warp tiling, double buffering, ...) gets a new file,
so `bench.py` can time every variant side by side and each file stays a readable snapshot of one
idea.

## Where things are written down

| file | contents |
| --- | --- |
| `kernels/README.md` (this file) | operator index, conventions, build and run, timing method |
| `kernels/<op>/README.md` | the problem and its LeetGPU link; the LeetGPU benchmark shapes; naming notes; the variants table; **a performance table with one row per revision** (file, commit, change, time, TFLOP/s or GB/s, % of torch); open issues |
| `kernels/<op>/LEARNINGS_CUDA_CPP.md`, `LEARNINGS_TRITON.md` | learnings for that problem: how it was coded, the bugs and challenges, GPU architecture lessons, profiling observations |
| [`learnings/CUDA_CPP_GENERAL.md`](../learnings/CUDA_CPP_GENERAL.md), [`learnings/TRITON_GENERAL.md`](../learnings/TRITON_GENERAL.md) | lessons that apply across problems |

## Build and run

Each `.cu` file is standalone and contains its own `main()`, which checks the GPU result against a
CPU reference. A file's header comment lists only the flags specific to that file; everything
general is here.

### Windows

1. From the Start menu, open **x64 Native Tools Command Prompt for VS**. (It sets up the MSVC
   compiler that `nvcc` needs.)
2. Compile and run:
   ```
   nvcc kernels\matmul\matmul_naive.cu -o matmul_naive
   matmul_naive
   ```

### Linux

```bash
nvcc -arch=native kernels/matmul/matmul_naive.cu -o matmul_naive
./matmul_naive
```

- **`-arch=native`** builds for the GPU in this machine. A wrong `-arch` (e.g. `sm_120` on an
  `sm_86` card) still compiles, but the kernel never runs, and the outputs are silently all
  zeros. To build for a specific card instead: `-arch=sm_86` (RTX A4000), `-arch=sm_120` (RTX
  PRO 6000 Blackwell). Check the card with `nvidia-smi --query-gpu=name,compute_cap --format=csv`.
- Binaries have no extension on Linux; `.gitignore` ignores extension-less files under `kernels/`.

### Debugging and inspection

| goal | command |
| --- | --- |
| out-of-bounds and misaligned accesses | `compute-sanitizer ./prog` (memcheck) |
| shared-memory races | `compute-sanitizer --tool racecheck ./prog` |
| barrier misuse | `compute-sanitizer --tool synccheck ./prog` |
| reads of uninitialized global memory | `compute-sanitizer --tool initcheck ./prog` |
| source lines in sanitizer reports | add `-lineinfo` to the `nvcc` command |
| registers, spills, shared memory per kernel | add `-Xptxas -v` |
| the machine code the GPU runs (SASS) | `nvcc -arch=native -cubin f.cu -o f.cubin && cuobjdump -sass f.cubin` |
| rule out FMA as the cause of CPU/GPU float differences | add `-fmad=false` (diagnosis only: it makes kernels slower) |

Why these matter: [`learnings/CUDA_CPP_GENERAL.md`](../learnings/CUDA_CPP_GENERAL.md) §1 (compiling)
and §5 (debugging).

### Triton and torch

```bash
python kernels/<operator>/<operator>_triton.py
python kernels/<operator>/bench.py
```

## Timing method

*To be written when the first `bench.py` lands.* It will record the GPU, the CUDA, driver, Triton
and torch versions, the warm-up and repeat counts, and the problem shapes (the LeetGPU benchmark
configurations, listed in each operator's README).

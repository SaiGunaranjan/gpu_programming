# Triton: general learnings

Lessons that apply across problems: the programming model, `tl.constexpr`, masking, autotuning,
reading the generated PTX, and so on. Lessons specific to one problem live next to its code, in
`kernels/<op>/LEARNINGS_TRITON.md`. CUDA C++ counterpart: [`CUDA_CPP_GENERAL.md`](CUDA_CPP_GENERAL.md).

Grouped by topic, not by date. Each entry says **when** it was added and **which kernel** it came
from.

*The matmul-centred Triton notes are in `kernels/matmul/LEARNINGS_TRITON.md`. Their general parts
get split out into this file later.*

---

## 1. Autotuning

### 1.1 Choose the tile size and `num_warps` together, not as a full cross product
*Added 2026-11-08, from `kernels/matrix_transpose/matrix_transpose_triton.py`*

**What went wrong.** The first transpose autotune list was every tile size (8x8 .. 1024x1024) x every
`num_warps` (1 .. 32), 42 configs. The script appeared to hang: GPU at 0%, while a `ptxas` process
sat at 100% CPU for minutes on a 6 MB PTX file for a 1-warp program. `@triton.autotune` must
**compile every config before it times any of them**, so one absurd config blocks the whole run.

**Why.** A Triton tile lives in the program's **registers**, spread over `num_warps x 32` threads
(shared memory is only used for layout changes such as `tl.trans`). So the number that matters is

**elements per thread = BLOCK_M x BLOCK_K / (num_warps x 32)** (1-D: `BLOCK_SIZE / (num_warps x 32)`)

| config | elements per thread | |
| --- | --- | --- |
| 64x64, 4 warps | 32 | sensible |
| 64x64, 8 warps | 16 | sensible |
| 8x8, 32 warps | 0.06 | most threads have nothing to do |
| 1024x1024, 1 warp | 32,768 | far beyond the 255 registers per thread: spills or a compile that never ends |

**Rule of thumb for memory-bound kernels (copy, transpose, elementwise):** aim for roughly
**4 to 64 elements per thread**, and treat anything above ~128 as a red flag. Below ~4, warps are
mostly idle or there are too few loads in flight per thread. Above ~64, register pressure grows
and compile time explodes. (Kernels with accumulators, like matmul, need their own budget: count
the accumulator, A and B fragments per thread instead.)

**How to build the list.** Pick tile sizes and warp counts **together**: either list the pairs by
hand, or generate the cross product and **filter it** by elements per thread, so the corners
(huge tile with 1 warp, tiny tile with 32 warps) never get compiled. That also cuts the tuning time.
For 2-D tiles, include some **rectangular** shapes (e.g. 32x128, 128x32), not only squares: in a
transpose the read side and the write side see different contiguous lengths, so the best shape
need not be square.

### 1.2 Other autotuning gotchas
*Added 2026-11-08, from `kernels/matrix_transpose/matrix_transpose_triton.py`*

- **`key=[...]` must name kernel arguments.** Names that aren't arguments (e.g. `M`, `K` when the
  kernel takes `rows`, `cols`) are **silently dropped** (Triton 3.4:
  `[_args[k] for k in self.keys if k in _args]`). The key is then empty, and the autotuner tunes
  once and reuses that config for every shape.
- **`TRITON_PRINT_AUTOTUNING=1`** prints the winning config and how long tuning took. Set it before
  the first autotuned call.
- **Hang or slow compile?** Check `nvidia-smi` (GPU utilization) and `ps aux | grep ptxas`. GPU at
  0% with `ptxas` busy means it is still compiling, not running.

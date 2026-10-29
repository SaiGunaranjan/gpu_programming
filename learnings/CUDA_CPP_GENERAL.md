# CUDA C++: general learnings

Lessons that apply across problems: compiling, shared memory, barriers, floating point, launch
configuration, debugging, memory spaces. Lessons specific to one problem live next to its code, in
`kernels/<op>/LEARNINGS_CUDA_CPP.md`. Triton counterpart: [`TRITON_GENERAL.md`](TRITON_GENERAL.md).

Grouped by topic, not by date. Each entry says **when** it was added and **which kernel** it came
from. Add new entries to the matching section, or start a new section. Build and run commands live
in [`kernels/README.md`](../kernels/README.md); this file explains *why* the flags matter.

**Topics to revisit**
- [ ] §4.1b, "guard each access, no `return`".

---

## 1. Compiling (`nvcc`)

### 1.1 `-arch` must match the GPU you run on, or use `-arch=native`
*Added 2026-10-21, from `1d_convolution_v1.cu`*

- `-arch=sm_XX` builds machine code (SASS) for one GPU generation.
  - RTX PRO 6000 Blackwell = `sm_120`
  - RTX A4000 (Ampere) = `sm_86`
- The pod's GPU can change between sessions. Check it with `nvidia-smi`, or let nvcc pick:
  ```bash
  nvcc -arch=native 1d_convolution_v1.cu -o 1d_convolution_v1
  ```
- **Wrong arch fails silently.** A binary built for `sm_120` and run on an `sm_86` card does not
  crash. The launch fails with `cudaErrorNoKernelImageForDevice` ("no kernel image is available"),
  the kernel never runs, and the output buffer stays all zeros. Only an error check or
  `compute-sanitizer` shows this. See §5.1.

### 1.2 `-c` compiles without linking
*Added 2026-10-21, from `1d_convolution_v1.cu` header comment*

`nvcc -c file.cu -o file.o` produces an object file. That is useful before `main()` exists, since
an executable needs a `main`. Once `main` is written, drop `-c` and output an executable.

### 1.3 `-fmad=false` turns off fused multiply-add
*Added 2026-10-21*

Use it **only** to check that CPU/GPU mismatches come from rounding. Never leave it in a real
build. See §3.

---

## 2. Shared memory

### 2.1 A static `__shared__` array needs a compile-time size
*Added 2026-10-21, from `1d_convolution_v1.cu` (the notes in its closing comment)*

This does **not** compile ("cannot have static storage duration"):
```cpp
int num_elements_max = blockDim.x + MAX_KERNEL_SIZE - 1;
__shared__ float shared_mem_array[num_elements_max];
```
- `MAX_KERNEL_SIZE` is a `#define`. The preprocessor pastes in the number, so it's a constant.
- `blockDim.x` is only known at **launch**. The same compiled kernel can be launched with 128,
  256 or 1024 threads. A runtime value plus a constant is still a runtime value. That makes this
  a variable-length array (VLA), and `__shared__` memory can't be a VLA.
- A plain `int` variable isn't a compile-time constant even when it's built only from constants.
  Use `constexpr`.

**Two fixes:**
1. **Compile-time block size** (what v1 does): `#define BLOCK_SIZE 256`, then use it both in the
   kernel and in the launch:
   ```cpp
   constexpr int num_elements_max = BLOCK_SIZE + MAX_KERNEL_SIZE - 1;
   __shared__ float shared_mem_array[num_elements_max];
   ```
2. **Dynamic shared memory**: declare `extern __shared__ float smem[];` and pass the size in
   bytes as the **third** launch parameter: `kernel<<<grid, block, smem_bytes>>>(...)`. The host
   can then size it from the real `kernel_size` instead of `MAX_KERNEL_SIZE`. This shows up
   everywhere in real kernels.

### 2.2 Tiled-load checklist (bugs found while writing v1)
*Added 2026-10-21, from `1d_convolution_v1.cu`. Status column checked against v1 on 2026-10-21.*

| # | pitfall | status in v1 |
|---|---|---|
| 1 | **Early `return` before `__syncthreads()`.** The CUDA guide only allows `__syncthreads()` inside a branch if every thread in the block takes the same branch. In practice, on current GPUs, exited threads stop counting and the barrier releases once the remaining threads arrive, so nothing hangs or errors. **The real danger:** exited threads skip their share of the tile load. Safer pattern: don't return; guard the load and the store with `if`s instead. | **present but harmless.** `if (global_tid > input_size) return;` does fire for `input_size=300, kernel_size=5` (threads 301–511), but those threads only own slots past the end of `input`. Results are correct and all 4 sanitizer tools report 0 errors. **Tested 2026-10-21:** changing the exit to `global_tid >= output_size` makes threads 296–299 skip loading `input[296..299]`, and outputs 292–295 come out wrong (e.g. 0.56 vs 0.001). The sanitizers still report 0 errors, because initcheck only tracks global memory, not shared. |
| 2 | **Global index for shared slot `i`.** Slot `local_tid + i` must hold `input[blockIdx.x*blockDim.x + local_tid + i]`. Adding the thread index twice (`global_tid + i` with `i` starting at `local_tid`) is wrong. | fixed (loop starts at `i = 0`) |
| 3 | **Bounds on the tile load.** The last block can read past `input_size`. Guard every global load. | fixed |
| 4 | **Off-by-one in the guard.** Valid indices are `0 … size-1`, so the out-of-range test is `>=`, not `>`. | `> input_size` remains, but loads and stores are guarded separately, so it's harmless |
| 5 | **Tile size vs. `kernel_size`.** Each block needs only `blockDim.x + kernel_size - 1` elements. Loading `blockDim.x + MAX_KERNEL_SIZE - 1` is correct but wastes loads. Dynamic shared memory (§2.1, fix 2) solves this. | still loads the max size |

### 2.3 How many `__syncthreads()`? One per write → read handoff, not one per load
*Added 2026-10-21, from `1d_convolution_v2.cu` (loads both the input tile and the filter into
shared memory)*

A barrier is needed only **between threads writing shared memory and other threads reading it**.
Two loads in a row that write *separate* regions, with no reads in between, need **one** barrier
after both:
```cpp
// load input tile → smem_input   (writes)
// load filter     → smem_kernel  (writes, different region)
__syncthreads();                  // one barrier: both regions complete before any read
// compute: reads smem_input and smem_kernel
```
A barrier between the two loads protects nothing. It only makes every thread wait for the slowest
one before starting its filter loads.

**Rule:** write → `__syncthreads()` → read. Place a barrier where reads of *other threads'* data
begin. (A tiled matmul loop needs a *second* barrier per iteration, for the opposite handoff:
everyone must finish **reading** a tile before anyone **overwrites** it with the next one.)

**Triton equivalent: none needed.** In Triton you work on whole tiles (`tl.load`, `tl.dot`) and
never write a shared-memory slot per thread. The compiler stages tiles through shared memory and
inserts the barriers itself (`cp.async.wait_group` + `bar.sync`, seen in the matmul PTX; see the
Triton notes Q1, Q18–Q20). The only manual one, `tl.debug_barrier()` (= `__syncthreads()`), is for
rare cases where threads of one program hand data to each other through **global** memory
(`tl.store` then `tl.load`), which the compiler doesn't track. *(Added 2026-10-22.)*

---

## 3. Floating point: CPU vs GPU results

### 3.1 Compare floats with a tolerance, not `!=`
*Added 2026-10-21, from `1d_convolution_v1.cu`*

**Symptom:** v1 reported 139 of 511 outputs "not matching", yet the printed values looked
identical. `cout` prints only 6 significant digits. At 9 digits:
```
output_cpu[0] = 0.492379606, output_gpu[0] = 0.492379636   // diff ≈ 3e-8 (~1 ULP)
```
**Check:**
```cpp
#include <cmath>
if (fabsf(cpu[i] - gpu[i]) > 1e-5f * fmaxf(1.0f, fabsf(cpu[i]))) { /* real mismatch */ }
```
To see more digits: `#include <iomanip>`, then `cout << std::setprecision(9)`.

### 3.2 FMA (fused multiply-add) is why the last bit differs
*Added 2026-10-21*

FMA computes `a*b + c` in **one instruction with one rounding**. Without it, the multiply and the
add are each rounded:
```
no FMA:  t = round(a*b);  acc = round(acc + t)    // 2 roundings (what the CPU code did)
FMA:     acc = round(acc + a*b)                    // 1 rounding  (what nvcc emits by default)
```
- nvcc fuses `*` then `+` into FMA by default (`-fmad=true`; shows up as `FFMA` in SASS).
- It's the GPU's basic arithmetic unit: same cost as one add, half the instructions. GPU FLOPS
  ratings count one FMA as 2 FLOPs.
- FMA results are usually *slightly more* accurate, not less.
- Proof for v1: `-fmad=true` gave 139 mismatches; `-fmad=false` gave **0**.
- Per-operation control in CUDA: `__fmaf_rn(a,b,c)` forces an FMA. `__fmul_rn` and `__fadd_rn`
  are never fused. On the CPU, `std::fma(a,b,c)`.
- Different summation order (e.g. parallel reductions) is the other main reason GPU and CPU
  results don't match bit for bit.

---

## 4. Launch configuration

### 4.1 Bounds checks: does every kernel need `if (tid >= n) return;`?
*Added 2026-10-21, from `1d_convolution_v1.cu`. ⏳ **To revisit:** the "guard each access" style
(4.1b) wasn't fully clear yet.*

**Short answer:** every kernel needs a **bounds check**, but it doesn't have to be a `return`.

**Why a check is needed.** The grid is rounded *up* so every element gets a thread:
```cpp
blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;   // n=300, 256 threads → 2 blocks = 512 threads
```
Threads 300–511 have no element. Without a check they read or write past the end of the arrays.
Sometimes that crashes; sometimes it silently corrupts other data.

#### 4.1a Style 1: early `return`

```cpp
__global__ void scale(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;          // extra threads leave immediately
    out[tid] = 2.0f * in[tid];
}
```
Fine when **each thread works alone**: it reads its own element and writes its own element, with
no `__syncthreads()` and no warp operations (`__shfl_sync`, ...) after the `return`.

#### 4.1b Style 2: guard each access, no `return`

**The problem with `return` in a tiled kernel.** In the convolution, a thread has **two jobs**:

1. **Helper job:** load its share of the input tile into shared memory, for *all* threads in the
   block to use.
2. **Own job:** compute one output element.

A thread with no output element (job 2 not needed) may still have job 1 to do. If it `return`s, its
part of the tile is never loaded, and **other** threads read garbage from shared memory. That's
exactly what happened in the `>= output_size` experiment (§2.2 #1): threads 296–299 returned,
`input[296..299]` was never loaded, and outputs 292–295 came out wrong.

**The fix: no thread leaves early.** Everyone runs the whole kernel. You only wrap each **memory
access** that could go out of range in its own `if`:

```cpp
__global__ void conv1d(const float* input, const float* kernel, float* output,
                       int input_size, int kernel_size) {
    int local_tid  = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;           // first input index of this block's tile
    int global_tid = block_start + local_tid;
    int output_size = input_size - kernel_size + 1;
    int tile_size  = blockDim.x + kernel_size - 1;        // elements this block needs
    __shared__ float tile[BLOCK_SIZE + MAX_KERNEL_SIZE - 1];

    // Job 1 (helper): every thread helps load the tile.
    for (int i = local_tid; i < tile_size; i += blockDim.x) {
        if (block_start + i < input_size)                 // guard #1: don't read past input
            tile[i] = input[block_start + i];
    }

    __syncthreads();                                      // every thread reaches this: none returned

    // Job 2 (own): only threads with a real output element compute and store.
    if (global_tid < output_size) {                       // guard #2: don't write past output
        float acc = 0.0f;
        for (int j = 0; j < kernel_size; j++)
            acc += tile[local_tid + j] * kernel[j];
        output[global_tid] = acc;
    }
}
```

Compared with an early `return`:

| | early `return` | guard each access |
|---|---|---|
| threads with no output | leave at the top | stay, help load the tile, reach the barrier, then skip the store |
| out-of-range read of `input` | prevented by leaving | prevented by guard #1 |
| out-of-range write of `output` | prevented by leaving | prevented by guard #2 |
| `__syncthreads()` | not every thread reaches it | **every** thread reaches it |
| tile fully loaded? | only if the exited threads happened to own no needed slot | yes, always |

**One-line version:** `return` asks "does this *thread* have work?"; guards ask "is this *memory
access* in range?". In a kernel where threads help each other, only the second question is safe.

#### 4.1c Style 3: grid-stride loop

```cpp
for (int i = tid; i < n; i += blockDim.x * gridDim.x) { out[i] = 2.0f * in[i]; }
```
Launch a fixed number of blocks (e.g. a few per SM). Each thread handles several elements, and the
loop condition *is* the check.

#### 4.1d Other things an early `return` breaks
- **Warp-level operations** like `__shfl_sync(0xffffffff, ...)` or `__reduce_add_sync`. The
  `0xffffffff` mask promises all 32 threads of the warp take part. If some have returned, the
  result is undefined. Common in reductions.
- **When can the check be dropped?** Only if the size is guaranteed to be a multiple of the block
  size (e.g. `assert(n % BLOCK_SIZE == 0)` on the host). Most people keep it anyway, because it
  costs almost nothing.

**Rule of thumb:** always check bounds. Use `return` only when nothing after it needs every thread
of the block or warp.

### 4.2 Kernel launches are asynchronous: when do you need `cudaDeviceSynchronize()`?
*Added 2026-10-22, from the question "does a Triton kernel also need one?"*

A launch (`kernel<<<...>>>` in CUDA, `kernel[grid](...)` in Triton) only **queues** the kernel on a
GPU stream and returns to the CPU immediately. Work on the same stream runs **in order**. An
explicit wait is only needed when the **CPU** must know the GPU has finished.

| situation | CUDA C++ | Triton / PyTorch |
|---|---|---|
| next GPU op uses the result (same stream) | ordered automatically | ordered automatically (Triton launches on PyTorch's current stream) |
| copy the result back to the CPU | `cudaMemcpy` device→host **waits for the kernel** | `.cpu()`, `.item()`, `print()`, `.tolist()` wait automatically |
| **timing** | sync needed (or CUDA events) | `torch.cuda.synchronize()`, or `triton.testing.do_bench` (syncs for you) |
| **report a kernel error at the right line** | `cudaDeviceSynchronize()` return value (§5.1) | `torch.cuda.synchronize()` right after the launch while debugging |

So the `cudaDeviceSynchronize()` in `solve()` (v1/v2) isn't needed for correct results, because
the `cudaMemcpy` back to the host already waits. It's still useful for timing and for checking
errors. Without a sync, a timer measures only how long it took to *queue* the launch, and a crash
inside the kernel is reported at some later API call.


### 4.3 `blockDim` = threads per block, never outputs per block
*Added 2026-10-24*

`blockDim` is exactly the 2nd `<<< >>>` argument: threads per block, per dimension.
`dim3 block(16,16)` → `blockDim.x = 16`, `blockDim.y = 16`; total = `blockDim.x*blockDim.y*blockDim.z`
= 256. CUDA doesn't know how many outputs a thread computes.

| variable | meaning | set by |
|---|---|---|
| `gridDim` | number of **blocks** in the grid | 1st launch argument |
| `blockDim` | number of **threads** per block | 2nd launch argument |
| `blockIdx` | this block's position, `0 … gridDim-1` | hardware |
| `threadIdx` | this thread's position in its block, `0 … blockDim-1` | hardware |

Threads per block equals outputs per block **only when each thread computes one output**, as in
the convolution kernels (256 = 256). That's why `blockIdx.x*blockDim.x + threadIdx.x` works as an
output index there. In the 4×4-per-thread matmul: 256 threads but 64×64 = 4,096 outputs per block,
so output indexing uses your own `BM/BN/TM/TN` ([matmul learnings §6](../kernels/matmul/LEARNINGS_CUDA_CPP.md)).

**Order and limits** *(added 2026-10-24)*: `kernel<<<grid, block>>>`, grid **first**. With
`grid(256,256)`, `block(16,16)`: `gridDim.x = 256`, `blockDim.x = 16`. Swapping them still compiles
but asks for 65,536 threads per block, so the launch fails (silently without an error check, §5.1).
Limits: **≤ 1,024 threads per block** in total (`x*y*z`); grid `x` ≤ 2³¹−1, `y`, `z` ≤ 65,535.

### 4.4 The execution model: every thread runs the same kernel (SPMD)
*Added 2026-10-27, from debugging the `A_smem` print in `matmul_v1.cu` (now `matmul_2d_blocktiling.cu`)*

Every thread of every block runs **the same kernel code**, top to bottom. Only the ID values
differ, and the code turns them into different behaviour:
- **which data**: index arithmetic (`row0 = threadIdx.y * 4`);
- **which code**: `if` conditions (`if (gr < M)`, a debug-print condition).

This is **SPMD** (single program, multiple data). Triton is the same one level up: every *program*
(`tl.program_id`) runs the kernel on a different tile.

- **An `if` only controls the code inside its braces.** Debug print: all threads copy (no condition)
  → `__syncthreads()` → *one* thread prints (a condition on `blockIdx` **and** `threadIdx`). A
  thread condition on the print doesn't take anyone out of the copy, and
  any thread can print the whole tile because shared memory is visible to the whole block.
  Without the thread condition, all 256 threads print it.
- **Keep `__syncthreads()` outside such `if`s:** every thread must reach it (§2.2 #1).
- **Blocks run in no guaranteed order**, not all at once (as many as fit on the SMs). No barrier
  across blocks inside a kernel; `__syncthreads()` is per block.
- **Warps execute each instruction together.** If threads of one warp take different sides of an
  `if`, the warp runs both sides one after the other (**divergence**). Harmless for a debug print;
  costly in hot loops.

**CUDA vs Triton, same model one level apart** *(added 2026-10-27)*:

| | CUDA C++ | Triton |
|---|---|---|
| you write the kernel for | one **thread** | one **program** (a tile) |
| IDs | `threadIdx` + `blockIdx` | `tl.program_id` only (= `blockIdx`; no `threadIdx`) |
| threads inside a program/block | you write the `threadIdx` math | compiler generates `num_warps × 32` threads and splits the tile by layout (Triton notes Q3/Q4) |

Underneath, every hardware thread runs the same compiled kernel in both. Triton just writes the
thread level for you, so load/compute mappings, coalescing and barriers are your job in CUDA C++
and compiler choices in Triton.
---

## 5. Debugging

### 5.1 Check errors, and use `compute-sanitizer`
*Added 2026-10-21*

CUDA API calls and kernel launches report errors by return code. Nothing is printed unless you
check:
```cpp
kernel<<<grid, block>>>(...);
cudaError_t err = cudaGetLastError();          // launch errors (bad config, wrong arch)
if (err != cudaSuccess) printf("launch: %s\n", cudaGetErrorString(err));
err = cudaDeviceSynchronize();                 // errors while the kernel ran
if (err != cudaSuccess) printf("run: %s\n", cudaGetErrorString(err));
```
`compute-sanitizer ./binary` catches API errors and out-of-bounds memory accesses, with no code
changes.

### 5.2 Read the SASS to see what the compiler actually did
*Added 2026-10-21*

SASS is the GPU's real machine code. It shows exactly which loads happen, where, and how the loops
were unrolled:
```bash
nvcc -arch=native -cubin 1d_convolution_v1.cu -o v1.cubin
cuobjdump -sass v1.cubin | grep -E 'LDG|LDS|LDC|FFMA'
```
| instruction | meaning |
|---|---|
| `LDG` | load from **global** memory (through L1/L2) |
| `LDS` / `STS` | load from / store to **shared** memory |
| `LDC` | load from **constant** memory |
| `FFMA` | fused multiply-add, FP32 (§3.2) |

Example from v1: the convolution loop was unrolled 16 at a time, with one `LDG` per `kernel[j]` and
one `LDS` per `shared_mem_array[local_tid+j]`. That's how §6.2 was confirmed.


### 5.3 Two errors seen while writing the tiled matmul
*Added 2026-10-27*

**`cudaErrorInvalidConfiguration` (error 9): the launch itself was rejected**, before any thread
ran. Usual causes: a **grid dimension of 0** (e.g. a ceiling-division formula that rounds down for
small sizes), more than **1,024 threads per block**, or more **shared memory** than the per-block
limit. Debug by printing `grid.x/.y` and `block.x/.y` just before the launch and checking them
against hand-computed values. Test any `cdiv` formula on a tiny case first (`cdiv(5, 2)` must be 3).

**`arr[a, b]` is not 2D indexing.** Inside `[ ]` the comma is C++'s **comma operator**: evaluate
`a`, discard it, keep `b`. So `A_smem[r, c]` = `A_smem[c]` (a whole row), and nvcc warns
`expression has no effect`. 2D arrays need `A_smem[r][c]`.
---

## 6. Memory spaces

### 6.1 Overview: where data can live
*Added 2026-10-21*

| space | declared as | scope / lifetime | size (typical) | speed | good for |
|---|---|---|---|---|---|
| registers | local variables (`float acc`) | one thread | 255 regs/thread max | fastest | accumulators, per-thread values |
| shared | `__shared__` | one block, for the kernel's duration | ~48–100 KB per block (depends on GPU) | ~L1 speed | tiles reused by many threads in a block (§2) |
| constant | `__constant__` at file scope | all threads, set from the host | **64 KB** total | fast **when a warp reads the same address** | small read-only data like convolution filters |
| global | `cudaMalloc` pointer | all threads, until `cudaFree` | GBs (DRAM) | slowest; cached in L2 (+ L1) | inputs/outputs |

### 6.2 Warp broadcast: same-address reads are cheap
*Added 2026-10-21, from `1d_convolution_v1.cu`*

In v1, every thread reads `kernel[j]` from **global** memory in its inner loop (confirmed in SASS:
one `LDG` per `j`, §5.2). That costs less than "256 threads × DRAM" because:

1. **Broadcast.** All 32 threads of a warp read the *same* `kernel[j]` at the same time. The
   hardware merges this into one request and sends the value to all 32 threads. So it's one load
   per warp per `j`, not one per thread.
2. **Caching.** The filter is at most 8 KB (2047 floats). After the first touch it stays in L2,
   usually in L1 too. DRAM is read about once.

What's left to pay: every warp still issues a load for every `j` and waits on L1/L2 latency.

The same reasoning applies to the input tile: neighbouring blocks' overlapping halos mostly come
from L2, not DRAM.

### 6.3 Options for a small read-only array that every thread reads (e.g. a convolution filter)
*Added 2026-10-21, from `1d_convolution_v1.cu`*

| option | how | notes |
|---|---|---|
| **`__constant__` memory** (best fit) | `__constant__ float kernel_c[MAX_KERNEL_SIZE];` at **file scope**; fill from the host with `cudaMemcpyToSymbol(kernel_c, kernel_h, bytes)`; read as `kernel_c[j]` in the kernel (no pointer argument) | The textbook choice for convolution filters. The constant cache is built for warp-uniform reads. **But:** if threads in a warp read *different* addresses, the reads are serialized, which is slow. 64 KB limit. Takes nothing from the shared-memory budget. |
| shared memory | a second `__shared__` array, filled by the block before `__syncthreads()` | Also fast for broadcast. Uses up to 8 KB more shared memory per block and needs a copy step. |
| `const float* __restrict__` | add `__restrict__` to the pointer parameter | Promises the compiler no other pointer writes to this data, which lets it use the read-only cache path. Smallest change, smallest gain. |

### 6.4 2D arrays vs 1D indexing
*Added 2026-10-23*

| memory | 2D indexing? | example |
|---|---|---|
| fixed-size local array (registers) | **yes** | `float acc[TM][TN]; acc[i][j] += a[i]*b[j];` |
| static shared array | **yes** | `__shared__ float As[BM][BK]; As[r][c]` |
| global memory (`cudaMalloc` pointer) | **no**, compute the offset | `A[row*K + col]`. A `float*` has no shape (like Triton's `stride_am`/`stride_an`) |
| dynamic shared (`extern __shared__`) | 1D, **or** cast to a 2D view if the column count is compile-time | `float (*As)[BK] = reinterpret_cast<float (*)[BK]>(smem); As[r][c]` |

- `acc[i][j]` is laid out row-major, identical to `acc[i*TN + j]`: same machine code.
- **No `std::vector` in device code.** It's host-only and allocates on the heap. Use plain arrays.
- **What keeps an array in registers is constant indices, not 1D vs 2D.** Registers can't be indexed
  by a runtime value, so keep loops over `i`, `j` unrolled (`#pragma unroll`, constant `TM`/`TN`).
  Otherwise the compiler may move the array to **local memory** (DRAM). For small arrays it can
  sometimes avoid that.
- **Check:** `nvcc -Xptxas -v` prints registers used, `bytes stack frame` and spills per kernel.
  Tested 2026-10-23: `float acc[4][4]` with unrolled loops gave 40 registers, 0 stack, 0 spills.
  Even a runtime index `acc[r%4][r%4]` stayed at 0 stack for this small array.

**Clarification: loop variables as indices are fine** *(added 2026-10-23, follow-up question "so
we can't use `acc[i][j]` with loop variables?")*

You can, and that's the normal way to write it. The requirement is that the loop gets
**unrolled**:
- **Registers have no addresses.** They're named in the instruction itself (`R4`, `R5`, ...),
  fixed at compile time. No instruction means "register number `i`" for a runtime `i`. (Shared
  and global memory are addressable, which is why runtime indices work there.)
- **Unrolling turns loop variables into constants.** With constant bounds, the compiler copies the
  body and puts the actual numbers in place of `i` and `j`:
  ```cpp
  acc[0][0] += a[0] * b[0];    // → FFMA R4,  R20, R24, R4
  acc[0][1] += a[0] * b[1];    // → FFMA R5,  R20, R25, R5
  ...                          // 16 copies; no i, no j left (register numbers illustrative)
  ```
  So "indices known at compile time" means known **after unrolling**.

| code | in registers? | why |
|---|---|---|
| `for (i < TM)`, `TM` compile-time | ✅ | unrolled, so `i` becomes 0..TM-1 |
| `for (i < tm)`, `tm` a runtime argument | ❌ likely not | can't unroll, so `acc[i]` needs a runtime index |
| `acc[threadIdx.x % 4]`, `acc[data[k]]` | ❌ maybe not | the index depends on runtime values |

❌ means the compiler may move `acc` to addressable **local memory** (DRAM, slow). Tiny arrays can
sometimes escape this. `#pragma unroll` makes sure the loop is unrolled; the compiler usually
unrolls small constant-bound loops anyway.

**One-line rule** *(added 2026-10-23)*: runtime loop bounds are fine **unless** the loop variable
indexes a **per-thread local array** you want in registers (`acc[]`, `a[]`, `b[]`). The loop
variable itself is always a register, and that's fine. Example: in `1d_convolution_v2.cu`, every
loop (`k < kernel_size`, ...) has runtime bounds and that's correct. Those loops only index shared
and global memory, and `acc` is a scalar `float` (one register, no index). The register-array
concern applies to kernels like matmul, where each thread keeps a small array (its `TM×TN` tile).

---

## 7. Matmul tiling

Matmul-specific notes (tile levels, the outer product, index mappings, bounds when loading tiles,
stress testing) are in [`kernels/matmul/LEARNINGS_CUDA_CPP.md`](../kernels/matmul/LEARNINGS_CUDA_CPP.md).

# matmul: CUDA C++ learnings

What writing the tiled matmul taught: the tiling concepts, the index mappings, the bugs, and the
stress test. Lessons that apply beyond matmul are in
[`learnings/CUDA_CPP_GENERAL.md`](../../learnings/CUDA_CPP_GENERAL.md) ("general §X.Y" below). The Triton side is in
[`LEARNINGS_TRITON.md`](LEARNINGS_TRITON.md); the problem, variants, performance and open issues
are in [`README.md`](README.md).

Naming follows LeetGPU: A is M×N, B is N×K, C is M×K, so **N is the inner dimension**.

---

## Bugs fixed while writing `matmul_2d_blocktiling.cu` (2026-10-27/28)

- Grid ceiling division written as `(a-b+1)/b` instead of `(a+b-1)/b`, which gave a grid dimension
  of 0 and `cudaErrorInvalidConfiguration` ([general §5.3](../../learnings/CUDA_CPP_GENERAL.md)). The same fix was needed in the
  per-thread pass counts.
- `arr[r, c]` instead of `arr[r][c]`: the comma operator ([general §5.3](../../learnings/CUDA_CPP_GENERAL.md)).
- `__synchthreads` typo.
- The debug print ran in every thread: it needs a condition on the block **and** the thread
  ([general §4.4](../../learnings/CUDA_CPP_GENERAL.md)).
- Missing barriers: one after both tile loads (before compute), and one after compute (before the
  next iteration overwrites the tiles). Debug prints of shared memory need a barrier before them.
- **Checking a whole tile instead of printing it:** recompute every slot's expected value (the
  DRAM value, or 0 outside the matrix) and compare. Exact `==` is fine for a plain copy. A check
  that prints only mismatches is silent when it passes, which looks the same as "never ran", so
  also print a count or a "tile OK" line. Comparing against the DRAM value has one blind spot: a
  wrong row stride written identically in the load and in the check. The final C-vs-CPU
  comparison covers it.

---

## 1. The three tile levels, and why the per-thread K step is 1
*Added 2026-10-23, from the question "do we need a separate inner-K per 4×4 thread tile?"*

In Triton you pick `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, and the compiler splits the work across threads.
In CUDA C++ (plain FMAs, no Tensor Cores) **you** pick every level:

| level | sizes you pick | controls |
|---|---|---|
| **block tile** | `BM × BN` (e.g. 64×64) | the piece of C per thread block, so `blocksPerGrid = (M/BM) × (N/BN)` |
| **K chunk** | `BK` (e.g. 8–32) | how much of A (`BM×BK`) and B (`BK×BN`) is in shared memory at once; outer loop runs `K/BK` times |
| **thread tile** | `TM × TN` (e.g. 4×4) | outputs per thread, in registers, so `threadsPerBlock = (BM/TM) × (BN/TN)` |

`threadsPerBlock` doesn't set the work per thread on its own. The tile sizes do, and the launch
must match them.

**There's no separate inner-K tile per thread.** Inside the shared-memory tile, each thread steps K
**by 1** and does an **outer product**:
```cpp
float acc[TM][TN] = {0};
for (int k = 0; k < BK; k++) {                              // step 1
    float a[TM], b[TN];
    for (int i = 0; i < TM; i++) a[i] = As[ty*TM + i][k];   // column of A tile
    for (int j = 0; j < TN; j++) b[j] = Bs[k][tx*TN + j];   // row of B tile
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            acc[i][j] += a[i] * b[j];                       // TM*TN FMAs
}
```
- Per `k`: `TM + TN` shared loads (8) for `TM × TN` FMAs (16). That reuse is why a thread tile
  helps, and it comes from the **output** tile's shape, not from any K size.
- `BK` and `TM`/`TN` are **independent** choices.
- Your Triton matmul shows the same thing: 32 outputs/thread × `BLOCK_SIZE_N`=32 k-steps =
  **1,024 `fma.rn.f32`** per loop iteration (Triton notes, Q1/Q3). The k loop was fully unrolled
  but still steps by 1. The `[4,4]` in `#blocked2` is the thread tile.

**Where a fixed inner K does exist: Tensor Cores.** With `mma.sync` / `wmma`, the **instruction**
sets the K step: e.g. `m16n8k8` (TF32) takes K in chunks of 8, and `m16n8k16` (FP16/BF16) in
chunks of 16. Advanced kernels also add a **warp tile** level between the block and thread levels.

## 2. "Outer product" = the same per-thread matmul, with the k loop on the outside
*Added 2026-10-23, follow-up to §1 ("each thread does a (4×K)·(K×4) matmul, so how is it an outer
product?")*

Both statements are true. The thread **does** compute the matmul `(TM×K)·(K×TN) → TM×TN`. "Outer
product" describes what **one k-step** of that matmul looks like when k is the **outer** loop:

```
inner-product view:  for each (i,j):  C[i][j] = Σ_k A[i][k]·B[k][j]          (one dot product per output)
outer-product view:  for each k:      C += A[:,k] (TM×1) × B[k,:] (1×TN)      (one outer product per k)
```
A column (TM×1) times a row (1×TN) is a TM×TN matrix: an **outer product**. So the matmul equals
the **sum of K outer products**. Same FMAs, same result, different loop order.

**Worked example** (2×2 output tile, K = 3):
```
A = | 1 2 3 |     B = | 1 0 |        C = A·B = | 4   5 |
    | 4 5 6 |         | 0 1 |                  | 10 11 |
                      | 1 1 |

k=0: [1,4]ᵀ×[1,0] = |1 0|   k=1: [2,5]ᵀ×[0,1] = |0 2|   k=2: [3,6]ᵀ×[1,1] = |3 3|
                    |4 0|                       |0 5|                       |6 6|
sum = | 4 5 | / | 10 11 |  ✓
```

**Why GPU code uses the outer-product order:**

| | outer-product order (k outer) | inner-product order (outputs outer) |
|---|---|---|
| loads per k-step | TM + TN = 8 | 2 per FMA |
| FMAs per load | 16 / 8 = **2** (each value reused 4×) | **0.5** (4× more traffic) |
| registers | 16 accumulators + 8 temporaries | to avoid reloads you'd need the whole TM×K and K×TN chunks in registers, which don't fit |

The thread tile only pays off because the k loop runs on the **outside**.

**Two clarifications** *(added 2026-10-23)*:

1. **K is the full inner dimension.** For `M×K · K×N`, each thread does **K outer-product steps in
   total**. Shared memory splits them into two loops: `K/BK` tile loads × `BK` steps per tile.
   ```cpp
   for (int k0 = 0; k0 < K; k0 += BK) {   // K/BK times
       // load next BM×BK (A) and BK×BN (B) tiles into shared (§4)
       __syncthreads();                   // tiles complete before anyone reads
       for (int k = 0; k < BK; k++) { ... one outer product ... }   // BK times
       __syncthreads();                   // everyone done reading before the next load overwrites
   }                                      // total = (K/BK) × BK = K steps
   ```
   (In the worked example above, K = 3 was the whole matmul, with no BK split.)

2. **It's 16 FMAs per k-step, not 16 bare multiplies.** A standalone outer product is only
   multiplies, but the code never builds it separately. Each product is added straight into the
   running total: `acc[i][j] += a[i] * b[j]` = multiply + add, fused into one FMA. That add **is**
   the "sum the outer products" step.
   Check against the standard count: `M·N/16` threads × `16·K` FMAs = `M·N·K` FMAs = **`2·M·N·K`
   FLOPs**, the usual matmul total. With no adds it would be only `MNK`.

## 3. Where the tile sizes live: launch vs. kernel, and why they're compile-time constants
*Added 2026-10-23*

`BM`/`BN` aren't passed *as* the grid. The launch dimensions are **calculated from** the tile sizes:
```cpp
dim3 block((BM/TM) * (BN/TN));                 // threads per block, e.g. 16*16 = 256
dim3 grid((N + BN - 1)/BN, (M + BM - 1)/BM);   // one block per BM×BN tile of C
```
**`BK` isn't in the launch at all.** It's used only inside the kernel (shared tile sizes and the
k-loop step). The launch says how many threads exist; how they walk K is the kernel's business.

**All of `BM, BN, BK, TM, TN` should be compile-time constants**, because:
1. static `__shared__ float As[BM][BK];` needs a constant size ([general §2.1](../../learnings/CUDA_CPP_GENERAL.md));
2. `float acc[TM][TN];` stays in **registers** only if its size is constant and all indices are
   known at compile time (fully unrolled loops). Otherwise it spills to slow **local memory**,
   which lives in DRAM;
3. a constant `BK` lets the k loop fully unroll (as Triton did: 1,024 straight FMAs).

| way | example | notes |
|---|---|---|
| `#define` | `#define BK 8` | simple; fixed for the whole file |
| `constexpr` | `constexpr int BK = 8;` | typed and scoped; preferred in modern C++ |
| **template parameter** | `template <int BM, int BN, int BK, int TM, int TN> __global__ void matmul(...)`, launched as `matmul<64,64,8,4,4><<<grid, block>>>(...)` | **= Triton's `tl.constexpr`.** Each `<...>` is a separately compiled kernel. Autotuning = instantiating a few and timing them (what `@triton.autotune` automates) |

A runtime `int BK` argument plus dynamic shared memory would work, but it blocks full unrolling and
tends to spill `acc` out of registers. **Rule:** problem sizes (`M, N, K`) are runtime arguments;
tile sizes are compile-time constants.

## 4. Splitting the tile *load* across threads: flat, coalesced, not 4×4 squares
*Added 2026-10-23. Example: BM = BN = BK = 64, 16×16 = 256 threads, each computing a 4×4 tile*

**Count:** 64×64 = 4,096 elements per tile ÷ 256 threads = **16 per thread**, for A and again for B.

**Don't copy in 4×4 squares** (matching the compute tile). The load mapping and the compute mapping
are **separate choices** (like Triton's separate load and compute layouts, Triton notes Q3d). For
loads the goal is **coalescing**: the 32 threads of a warp, in one load instruction, should read
**32 consecutive floats** (one 128-byte memory piece). With 4×4 squares, threads 0, 1, 2, ... read
columns 0, 4, 8, ... (stride 4), so a warp's request spans 4× more memory.

**Standard mapping:** flat thread id. Each pass copies 256 consecutive elements (4 rows of 64);
16 passes cover the tile. **One loop per tile, not two nested row/col loops:**
```cpp
int tid = threadIdx.y * blockDim.x + threadIdx.x;   // 0..255
for (int p = 0; p < (BM * BK) / 256; p++) {         // 16 passes
    int idx = p * 256 + tid;
    int r = idx / BK, c = idx % BK;                 // pass p, thread t → row 4p + t/64, col t%64
    As[r][c] = A[(blockIdx.y * BM + r) * K + (k0 + c)];
}
// B: Bs[r][c] = B[(k0 + r) * N + (blockIdx.x * BN + c)];
```
⚠️ This pass loop has **no tile check**, so it's correct only when `BM*BK` divides evenly by the
thread count (here 4,096 = 16 × 256). For other sizes use a ceiling pass count plus
`if (idx < BM*BK)`, or the index loop of §7 (see §10). *(added 2026-10-26)*

**Whole loop structure.** The load and compute loops are **siblings** inside `k0`, not nested in
each other *(clarified 2026-10-23)*:
```
for k0 = 0 … K step BK                  ← outer: walks along K (not over output tiles)
├── for p = 0 … 15                      ← load: 16 passes (A and B can share this loop)
│       As[r][c] = A[...];  Bs[r][c] = B[...];
├── __syncthreads();
├── for k = 0 … BK-1                    ← compute: 64 outer-product steps
│   ├── for i < 4: a[i] = As[..][k];    for j < 4: b[j] = Bs[k][..];   (unrolled)
│   └── for i < 4, for j < 4: acc[i][j] += a[i] * b[j];               (unrolled)
└── __syncthreads();
// after k0 finishes: write acc[4][4] to C
```
- **The block's output tile is fixed by `blockIdx` and never changes.** Each `k0` iteration loads a
  new **slice along K**: columns `k0…k0+BK-1` of A's row band and rows `k0…k0+BK-1` of B's column
  band.
- Depth: load = `k0`→`p` (2 levels); compute = `k0`→`k`→`i`→`j` (4 levels, with `i`/`j`
  unrolled into 16 straight FMAs, so only `k0` and `k` remain as real loops).
- In Triton you write only the `k0` loop. `p`, the barriers and `k`/`i`/`j` are generated.

- **Shared memory:** BK = 64 → 2 × 64×64 × 4 B = **32 KB/block**. Under the 48 KB static limit,
  but it limits how many blocks fit on one SM.
- **Edges:** if M, N, K aren't multiples of 64, guard each load and write 0 when out of range
  ([general §4.1b](../../learnings/CUDA_CPP_GENERAL.md) style, not `return`).
- **Next optimisation:** `float4` loads (4 consecutive floats per instruction) → 16 threads per
  64-wide row, 4 passes instead of 16. Needs 16-byte-aligned addresses.

## 5. What the inner-product version looks like (for comparison)
*Added 2026-10-24*

Inside each tile it's **three** loops: `i`, `j` (bound 4) and `k` (bound `BK`) **innermost**:
```cpp
for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < BK; k++)
            acc[i][j] += As[ty*4 + i][k] * Bs[k][tx*4 + j];
```
Same three loops as the outer-product version, in a different order:

| | order inside a tile | shared loads per FMA |
|---|---|---|
| inner product | `i` → `j` → **`k`** | **2** (row `i` of `As` re-read for every `j`) |
| outer product (§2) | **`k`** → `i` → `j` | **0.5** (8 loads feed 16 FMAs) |

Same FMAs, same result. The compiler won't swap the loop order for you.

**`acc[4][4]` is still needed** in the inner-product version. Because `k0` is the outermost loop,
each tile adds only *part* of every output's sum, so all 16 partial sums must survive across `k0`
iterations. A single `float sum` per output would only work without shared-memory tiling (`k` over
the whole of K).

## 6. Indexing when each thread computes a TM×TN tile: `blockDim` doesn't know about it
*Added 2026-10-24*

`blockDim.x` is **exactly what you launched with** (16), never "16 × 4". CUDA doesn't know a thread
computes 4×4 outputs; only your code does. So the **block** offset must step by the block's tile
width `BN` (= 64), not by `blockDim.x`:
```cpp
constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
int row0 = blockIdx.y * BM + threadIdx.y * TM;   // = blockIdx.y * (blockDim.y*TM) + threadIdx.y*TM
int col0 = blockIdx.x * BN + threadIdx.x * TN;
// thread owns C[row0 + i][col0 + j], i < TM, j < TN
```
❌ `blockIdx.x * blockDim.x + threadIdx.x * 4`: block 0 covers columns 0–63, but block 1 starts at 16
(should be 64), so blocks overlap and much of C is never written.

It's no longer a "global thread id". It's the **first output row/column** the thread owns. Name it
that way.

**Launch:** `dim3` uses `.x`/`.y` (not `[0]`); **x = columns, y = rows** (neighbouring threads in a
warp get neighbouring columns); ceiling division:
```cpp
dim3 block(16, 16);
dim3 grid((n_cols + BN - 1) / BN, (M + BM - 1) / BM);
```
(`n_cols` = C's column count: `K` in the Triton kernel's naming A=M×N, B=N×K; `N` in these notes'
naming A=M×K, B=K×N.)

**"The grid already uses 64, so why repeat 64 in the kernel?"** *(added 2026-10-24, from Triton
notes Q35, `matmul_4x4`)* The grid size only says **how many** blocks to launch. Inside the kernel
you get `gridDim` (e.g. 4), `blockDim` (16), `blockIdx` and `threadIdx`. **Nothing carries "each
block covers 64 columns."** The `64` in `cdiv(N, 64)` is used on the host to calculate the block
count and is then gone. The host and kernel are two halves of one agreement that CUDA doesn't link.
Change one alone and it still runs, just computing the wrong parts of C. (`N / gridDim.x` only
recovers 64 when N is an exact multiple.) **Fix:** define `BM, BN, TM, TN` once (`constexpr` or
template, §3) and derive `block`, `grid` and `row0/col0` all from them.

**The index formula sets the start; the loops set the count** *(added 2026-10-24)*. Writing
`col0 = blockIdx.x*(blockDim.x*16) + threadIdx.x*16` alone doesn't make a thread compute 16×16. Three
things must agree: (1) `row0`/`col0` spaced by `TM`/`TN`; (2) loops computing `acc[TM][TN]` for all
`i < TM`, `j < TN`; (3) grid = `cdiv(N, blockDim.x*TN)`, `cdiv(M, blockDim.y*TM)`. Spacing by 16 but
computing 4×4 leaves gaps in C.
**Size limit:** TM = TN = 16 means 256 accumulators, but a thread has at most **255 registers**
(and also needs `a[]`, `b[]`, indices), so values spill to local memory (check with `-Xptxas -v`,
[general §6.4](../../learnings/CUDA_CPP_GENERAL.md)). **Measured 2026-10-24 (sm_86, simple outer-product test kernel):** 4×4 → 40 regs, 0 spills;
8×8 → 96 regs, 0 spills; **16×16 → 255 regs (the cap), 80 bytes spilled.** Hitting 255 registers
also means at most 65,536 / 255 ≈ 256 threads per SM, i.e. one 256-thread block, so there's little
room to hide latency. Typical SIMT thread tiles are 4×4 to 8×8 (16–64 accumulators).

**Summary: the "outputs per thread" contract** *(added 2026-10-24)*. CUDA checks none of it:

| where | must say | 4×4 per thread, 16×16 threads |
|---|---|---|
| 1. launch (host) | grid = outputs ÷ (threads × outputs per thread) | `grid(cdiv(N,64), cdiv(M,64))` |
| 2. index math (kernel) | thread starts spaced `TM`/`TN`, block starts `BM`/`BN` | `col0 = blockIdx.x*64 + threadIdx.x*4` |
| 3. compute + store (kernel) | loops produce all `TM×TN` outputs from that start | `acc[i][j]`, `i,j < 4` → `C[row0+i][col0+j]` |
| (+ with shared memory) | tile sizes and load loop from the same constants | `As[BM][BK]`, 16 load passes |

Break it and the kernel still compiles and runs, leaving gaps, overlaps or wrong values. Guard: one
set of constants for all places (§3), and always compare against a CPU reference ([general §3.1](../../learnings/CUDA_CPP_GENERAL.md)).

## 7. Two index mappings in one kernel: flat for loading, spaced for computing (full tiled matmul)
*Added 2026-10-24. Tested on sm_86: 5 sizes incl. 1000×513×77 and 1×1×1, 0 mismatches vs CPU;
memcheck/racecheck/synccheck 0 errors; 60 registers, 0 spills, 32 KB smem per block.*

The **same 256 threads** use two different formulas, in phases separated by `__syncthreads()`:

| phase | index | thread `t` does | why |
|---|---|---|---|
| **load** | flat `tid = threadIdx.y*16 + threadIdx.x` | copies tile elements `tid`, `tid+256`, … (16 each) | neighbouring threads → neighbouring columns → **coalesced** |
| **compute + store** | `row0 = threadIdx.y*TM`, `col0 = threadIdx.x*TN` | its own 4×4 patch, from shared | register reuse (§2) |

After the barrier the whole tile is in shared memory, readable by every thread in the block, so
**it doesn't matter which thread loaded which element.** The load mapping is chosen for DRAM, the
compute mapping for registers.

```cpp
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);   // 256

    int block_row = blockIdx.y * BM;          // this block's tile of C
    int block_col = blockIdx.x * BN;

    // Mapping 1 (LOAD): flat id, neighbouring threads -> neighbouring columns (coalesced)
    int tid = threadIdx.y * blockDim.x + threadIdx.x;   // 0..255

    // Mapping 2 (COMPUTE): each thread owns a TM x TN patch, spaced TM/TN apart (inside the tile)
    int row0 = threadIdx.y * TM;
    int col0 = threadIdx.x * TN;

    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- load phase: mapping 1 ----
        for (int idx = tid; idx < BM * BK; idx += NUM_THREADS) {
            int r = idx / BK, c = idx % BK;
            int gr = block_row + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int idx = tid; idx < BK * BN; idx += NUM_THREADS) {
            int r = idx / BN, c = idx % BN;
            int gr = k0 + r, gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        // ---- compute phase: mapping 2 ----
        for (int k = 0; k < BK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[row0 + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][col0 + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    // ---- store: mapping 2 ----
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gr = block_row + row0 + i, gc = block_col + col0 + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

// host
constexpr int BM = 64, BN = 64, BK = 64, TM = 4, TN = 4;
dim3 block(BN / TN, BM / TM);                              // 16 x 16
dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
matmul_tiled<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
```
- Load loop `for (idx = tid; idx < BM*BK; idx += NUM_THREADS)` = the 16 passes of §4, for any
  tile size.
- Edges: out-of-range elements load as `0.0f` (they add nothing); the store is guarded; no `return`,
  so all threads reach both barriers ([general §4.1b](../../learnings/CUDA_CPP_GENERAL.md)).
- **Next optimisation: bank conflicts** (both 2-way, checked 2026-10-24). In the compute phase, one
  warp = 16 `tx` × 2 `ty`:
  - `As[row0+i][k]`: 2 addresses, 4 rows = 256 floats apart → **same bank**.
  - `Bs[k][col0+j]`: 16 addresses `tx*4`; `tx` and `tx+8` are 32 floats apart → **same bank**.
  A bank serves one address per cycle, so these reads take 2 rounds instead of 1. For `As`,
  padding rows (`As[BM][BK+1]`, so rows no longer start in the same bank) fixes it. For `Bs` the
  clash is *within* a row, so padding doesn't help; use a different column mapping, `float4` loads,
  or swizzling (Triton notes Q20/Q21). Speed only; results are correct.

## 8. How to picture it: one launch shape, relabelled for each job
*Added 2026-10-25, from "the launch has one blockDim but the A load seems to need another"*

There's only **one** `blockDim` (16×16). The load doesn't need a different one: it **relabels the
same 256 threads with arithmetic**. The launch shape is just how threads get their ID badges.

**Step 1, launch:** 16×16 badges `(tx, ty)`.

**Step 2, flatten:** `tid = ty*16 + tx`, read row by row like text → a line 0…255
(warp 0 = tid 0–31 = ty 0–1).

**Step 3, fold into the load shape:** `r = tid/64`, `c = tid%64` → a **virtual 4×64 block** (your
own invention; CUDA doesn't know it):
```
virtual r=0  [ tid   0 …  63 ]  ← ty 0–3
        r=1  [ tid  64 … 127 ]  ← ty 4–7
        r=2  [ tid 128 … 191 ]  ← ty 8–11
        r=3  [ tid 192 … 255 ]  ← ty 12–15
```
It covers a 64-wide row exactly. 16 passes stack it down the 64×64 tile (pass `p` → tile rows
`4p…4p+3`). Warp 0 → row `4p`, cols 0–31 = 32 consecutive floats = **coalesced**.

**Step 4, place the tile in A:**
```
                 A  (M × K)
          k0=0   k0=64  k0=128 ...      ← k0 loop slides right
        ┌──────┬──────┬──────┬────
by=0    │      │      │      │          rows 0–63
        ├──────┼──────┼──────┼────
by=1    │ tile │ tile │ tile │ →        rows 64–127   (block_row = blockIdx.y*64)
        ├──────┼──────┼──────┼────
global A row = blockIdx.y*64 + (4p + r),   global A col = k0 + c
```
`blockIdx.x` doesn't appear: every block in the same block-row loads the **same** A tiles. For B
it's the reverse (`blockIdx.x` picks the column band; `blockIdx.y` doesn't matter).

**One thread, both jobs:** thread `(tx=5, ty=6)`, block `(bx=2, by=1)`, `k0 = 128`:

| | formula | value |
|---|---|---|
| flat id | `ty*16 + tx` | 101 |
| load: virtual position | `r = 101/64`, `c = 101%64` | r = 1, c = 37 |
| load: A elements (16 passes) | row `64 + 4p + 1`, col `128 + 37` | A[65][165], A[69][165], …, A[125][165] |
| compute: C patch | `row0 = 6*4`, `col0 = 5*4` in the tile | **C[88…91][148…151]** |

It *loads* one tile column (16 scattered elements) but *computes* a 4×4 patch somewhere else.
Two jobs, connected only through shared memory and the barrier.

**Mental model:** the launch hands out 256 badges arranged 16×16. For each job, **flatten the badge
and fold it into the shape that suits the job**: 4×64 for loading (contiguous warp reads), 4×4
patches for computing (register reuse). `blockIdx` and `k0` only say **where** in A, B, C the tile
is.

**Clarification: how the global positions are computed** *(added 2026-10-25)*

**global = (blockIdx × tile size) + (local position in the tile)**

- **C patch** of block `(bx=2, by=1)`: tile starts at row `1×64 = 64`, col `2×64 = 128`. Local
  patch rows 24–27, cols 20–23 → global rows `64+24…64+27` = **88…91**, cols `128+20…128+23` =
  **148…151**.
- **A elements:** the A tile starts at row `by×64 = 64` (the A rows matching the block's C rows) and
  column `k0 = 128` (current K slice). Thread 101's local position: column 37, row `4p+1`:

  | pass p | local row | global row (64 + local) | global col (128 + 37) |
  |---|---|---|---|
  | 0 | 1 | 65 | 165 |
  | 1 | 5 | 69 | 165 |
  | … | … | … | … |
  | 15 | 61 | 125 | 165 |

  Over 16 passes thread 101 copies local column 37, every 4th row. The other threads fill the rest.

**`gridDim` isn't used inside this kernel, and that's normal.** A block needs *which* block it is
(`blockIdx`), not *how many* exist. Edges are handled by comparing against `M, N, K`. `gridDim` was
used on the **host** (`grid(cdiv(N,64), cdiv(M,64))`). Inside a kernel it's needed when work
depends on the **total** thread count, mainly the grid-stride loop ([general §4.1c](../../learnings/CUDA_CPP_GENERAL.md)):
`for (i = tid; i < n; i += blockDim.x * gridDim.x)`.

**Flattening for global memory** *(added 2026-10-25)*: the 2D picture gives `(gr, gc)`; global
memory is flat row-major, so flatten at the access: **`X[gr * (columns of X) + gc]`**. The row
stride is the column count of **that** matrix:

| | CUDA notes naming (A M×K, B K×N, C M×N) | Triton kernel naming (A M×N, B N×K, C M×K) |
|---|---|---|
| A | `A[gr*K + gc]`: K = inner dim | `A[gr*N + gc]`: N = inner dim (= `stride_am`) |
| B | `B[gr*N + gc]`: B's column count | `B[gr*K + gc]` |
| C | `C[gr*N + gc]` | `C[gr*K + gc]` |

Only A's stride is the inner dimension. Example: A[65][165] with K = 512 → `65*512 + 165 = 33,445`.
Shared memory stays 2D (`As[r][c]`, fixed compile-time size, [general §6.4](../../learnings/CUDA_CPP_GENERAL.md)).

**Summary in your words** *(added 2026-10-25)*: the launch gives a fixed number of workers in fixed
groups; index arithmetic assigns each worker its job, and the job can change per phase (flat 4×64
for loading A/B, spaced 4×4 patches for computing C). The grouping has three hardware rules the
relabelling must respect:
1. **Block** = the team that shares shared memory and barriers. Data hand-offs only happen inside a
   block (hence one block = one C tile).
2. **Warp** = 32 *consecutive* flat ids that execute together. Map consecutive `tid` → consecutive
   addresses for loads (coalescing).
3. **Registers are private** to each thread (`acc` can't be read by other threads); only shared
   memory is shared.

## 9. Do the tile widths have to divide the thread count?
*Added 2026-10-25*

**Indices are remappable, including `blockIdx`.** Advanced matmuls reorder *blocks* too, so that
blocks running at the same time work on nearby C tiles and reuse A/B from L2 (Triton's
`GROUP_SIZE_M`).

**Naming:** the A tile's load width is **`BK`** (K-chunk); `BN` is the B tile's width. In the
example both are 64.

**Dividing the thread count is convenient, not required.** The flat load loop
(`for idx = tid; idx < BM*BK; idx += NUM_THREADS`, `r = idx/BK`, `c = idx%BK`) works for any sizes.

| property | condition | if not met |
|---|---|---|
| no partial last pass | `BM×BK` divisible by the thread count | last pass has idle threads (the loop handles it) |
| each warp reads one 128-byte piece | row width (`BK` for A, `BN` for B) a **multiple of 32** | a warp spans several short rows → several smaller pieces; correct but slower |

Example: `BK = 8` (common, saves shared memory): A tile 64×8 = 512 → 2 passes, but a warp covers 4
rows × 8 → 4 separate 32-byte pieces. Fixed in real kernels with `float4` loads or a transposed A
load.

**Hard constraints** on tile sizes instead: `(BM/TM)×(BN/TN)` threads ≤ 1,024; shared memory
`(BM×BK + BK×BN)×4 B` within the per-block limit; `TM×TN` accumulators without spilling (§6).

## 10. Bounds when loading tiles: matrix edges need `if`s, tile edges don't
*Added 2026-10-26*

| boundary | how it's handled |
|---|---|
| **tile** (stay inside `As`/`Bs`) | the **loop**: counter starts at `tid`, condition `idx < BM*BK` → `r = idx/BK < BM`, `c = idx%BK < BK` always. No `if`. |
| **matrix** (stay inside A, B) | `if` on global `gr`/`gc`: A against M, K; B against K, N. **Write 0** when out of range. |
| **store to C** | `if (gr < M && gc < N)`: skip the write |

**Why conv v1/v2 needed a tile check:** their loop counter starts at 0 (`for i = 0; i < n; i += blockDim.x`)
but the index is `local_tid + i`, so the loop condition doesn't protect the index and
`local_tid + i < n` is needed. Starting the counter at the thread's index
(`for idx = local_tid; idx < n; idx += blockDim.x`) makes the loop condition the tile check.

**Why matmul must write 0, not skip:** shared memory isn't cleared between `k0` iterations, so a
skipped slot keeps the **previous tile's value**, and the compute loop reads every slot, adding the
stale value into real outputs. Zero adds nothing. (In the convolution, skipping was fine: slots past
the end of `input` are only read by threads whose output is never stored.)

**Pass loop vs. index loop** *(added 2026-10-26)*: the tile check is needed if the loop counts
**passes**:
```cpp
constexpr int TILE = BM * BK, PASSES = (TILE + NUM_THREADS - 1) / NUM_THREADS;   // ceiling
for (int p = 0; p < PASSES; p++) {
    int idx = p * NUM_THREADS + tid;
    if (idx < TILE) { /* r = idx/BK, c = idx%BK; matrix check + load */ }      // tile check
}
// equivalent, check built into the loop condition:
for (int idx = tid; idx < TILE; idx += NUM_THREADS) { ... }
```
The last pass doesn't need all threads, only as many as there are elements left; the rest skip.
E.g. BM = 64, BK = 10: 640 elements / 256 threads → passes of 256 + 256 + **128**; threads
128–255 skip pass 3. With an even split (4,096 = 16 × 256) every pass is full and the check never
fails.

## 11. 2D or 1D indexing? Same speed; think in 2D, flatten only where required
*Added 2026-10-26*

`As[r][c]` and `As[r*BK + c]` compile to the same address math ([general §6.4](../../learnings/CUDA_CPP_GENERAL.md)). Choose for clarity:

| part | use | why |
|---|---|---|
| giving out the load work | 1D (`tid`, `idx`) | it's a line: 4,096 elements over 256 threads |
| tile / matrix positions | 2D (`r, c`, `gr, gc`) | the bounds checks and the picture are 2D |
| shared arrays | 2D (`As[BM][BK]`) | readable; the compiler knows the shape |
| global memory | 1D (`A[gr*K + gc]`) | required: a pointer has no shape |
| compute patch | 2D (`acc[i][j]`, `row0+i`) | matches the 4×4 patch |

**The real cost: `/` and `%`** in `r = idx / BK`, `c = idx % BK`. With a compile-time power-of-2
`BK` they become a shift and a mask (free). With a runtime or non-power-of-2 value they take dozens
of instructions, inside a loop that runs every tile. **Equivalent 2D load loop** that divides once:
```cpp
constexpr int ROWS_PER_PASS = NUM_THREADS / BK;     // 256/64 = 4 (BK must divide the thread count)
int r0 = tid / BK, c = tid % BK;                    // once
for (int r = r0; r < BM; r += ROWS_PER_PASS)        // 16 iterations, no division inside
    As[r][c] = (block_row + r < M && k0 + c < K) ? A[(block_row + r) * K + (k0 + c)] : 0.0f;
```
Same elements as the §7 loop (thread 101: column 37, rows 1, 5, 9, …).

---

## 12. Stress test of `matmul_2d_blocktiling.cu` (functional correctness)
*Added 2026-10-28. The harness (`stress.cu`, not in the repo) `#include`s the kernel file unchanged
(with `#define main user_main`), so it tests the real kernel.*

**How it checks:** random floats in [-1, 1]; CPU reference in `double`; pass if
`|gpu - ref| ≤ 1e-5·Σ|a·b| + 1e-6` (an error budget that scales with the sum's size). One extra
run per config with small integers must match **exactly**. C is pre-filled with NaN
(`cudaMemset(0xFF)`), so any output the kernel never writes is caught. Launch:
`block(BK/TK, BM/TM)`, `grid(cdiv(K,BK), cdiv(M,BM))`, as in `solve()`.

**Shapes (per config):** 1×1×1, 1×1×7, 7×1×1, 1×300×1, 2×3×5, exactly one tile, one tile −1, one
tile +1, multi-tile, 103×167×129, 97×13×211, 17×1000×33, 500×7×300, K=1, M=1, 255×257×253,
333×129×65, plus 1024³ and 1000×2049×777.

**Results:**

| tile configs (BM, BN, BK, TM, TK) | threads | result |
|---|---|---|
| 64,64,64,4,4 (yours) · 32,32,32,4,4 · 64,16,64,4,4 · 128,8,128,8,8 · 64,32,32,8,2 · 16,16,16,1,1 · 32,64,64,2,8 | divide both tiles | **140/140 pass**, worst relative error ~1e-7 (float rounding). memcheck, racecheck, synccheck, initcheck: **0 errors** |
| 48,16,40,4,4 (120 threads) · 24,8,20,4,4 (30 threads) | **don't** divide the tiles | intermittent wrong C (0–2 failing tests per run); racecheck: write–write races between A's zero-fill (`A_smem[...] = 0.0f`) and B's load (`B_smem[...] = B[...]`). Open issue in [README.md](README.md#open-issues) |

**Power-of-2 tiles don't avoid it** *(added 2026-10-29)*. Triton requires power-of-2 block shapes
(`tl.arange(0, N)` needs N to be a power of 2). CUDA C++ has no such rule: `__shared__ float
A_smem[48][16]` is legal. Powers of 2 are still the usual choice (cheap `/` and `%`, aligned and
vector loads, thread counts that are whole warps; 120 threads = 3.75 warps). With power-of-2 tiles
**and** threads, the thread count either divides the tile or is **bigger** than it. The second
case also overshoots: **BM=16, BN=8, BK=16, TM=TK=1** → 256 threads, A and B tiles of 128
elements, so one pass covers 256 slots (128 extra). Confirmed: wrong C on some runs (M500 N7
K300), racecheck reports the same races.

**Takeaways**
- A race shows up as **intermittent** failures: the same binary passed under memcheck and failed
  on some plain runs. One clean run proves nothing about races; `racecheck` does.
- `memcheck` didn't flag the overflow. `A_smem` and `B_smem` sit next to each other, so writing
  past `A_smem` lands inside `B_smem`: still valid shared memory, just the wrong array.

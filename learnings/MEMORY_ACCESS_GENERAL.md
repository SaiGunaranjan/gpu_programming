# Shared Memory Banks, Bank Conflicts and DRAM Coalescing — Q&A Notes

*Context: NVIDIA Tesla T4 (Turing, sm_75), CUDA C++ and Triton kernels.*

*Added 2026-11-06: Q&A notes from a separate Claude session, brought in unchanged. The bank rules
(32 banks × 4 bytes) and 32-byte DRAM sectors are the same on Ampere, Ada and Blackwell, so it also
applies to the RTX 4090 (sm_89) and RTX PRO 6000 (sm_120). Related: shared memory **size** limits
(per block vs per SM, 48 KB static cap, opt-in) are in [`CUDA_CPP_GENERAL.md` §2.4](CUDA_CPP_GENERAL.md#24-shared-memory-limits-per-block-vs-per-sm-and-why-static-arrays-stop-at-48-kb);
broadcast of same-address global reads is in [§6.2](CUDA_CPP_GENERAL.md#62-warp-broadcast-same-address-reads-are-cheap).*

---

## Key takeaways (read this first)

> **1. Shared memory = 32 banks, each 4 bytes wide.** Word `w` lives in bank `w % 32`.
> The **address** a thread accesses decides the bank, not the thread's lane number.

> **2. Cost of one warp's shared-memory access = the largest number of *different words* any single bank must serve.**
> - 32 different words in 32 different banks → **1 pass**
> - Many threads reading the **same word** → **broadcast**, no conflict
> - `n` different words in the same bank → **n-way bank conflict → n passes** (still correct, just slower)
> - Writes follow the same rule.

> **3. Fixes for bank conflicts:** **padding** (`tile[32][33]`) or **swizzling** (store `(r, c)` at column `(c + r) % 32` or `c ^ r`).
> Libraries (CUTLASS, Triton) use **XOR swizzles on 16-byte chunks**: no wasted memory, and rows stay 16-byte aligned for `float4` / `ldmatrix`.

> **4. Warps are formed by flattening the block:** `linear_id = x + y·Dx + z·Dx·Dy`, then cut into groups of 32.
> In a block with `blockDim.x = 32`, a warp has **one fixed `threadIdx.y`** and `threadIdx.x = 0…31`.

> **5. How to analyze any memory access for performance:** freeze everything that's constant within a warp, let the
> warp's 32 consecutive linear IDs vary, and look at the 32 addresses produced.
> **What gets computed is per thread; how fast it runs is per warp.**

> **6. DRAM is fetched in 32-byte sectors.** Cost of one warp's global access = **number of distinct sectors touched**.
> 32 consecutive aligned floats = 128 bytes = 4 sectors = 100% efficient. A column read = 32 sectors = 12.5% efficient.

> **7. DRAM wants a warp's addresses packed close together (few sectors); shared memory wants them spread across banks.**
> Contiguous rows satisfy both. Fix DRAM coalescing first: it usually costs far more than bank conflicts.

> **8. The real DRAM limit is shared bandwidth, not per-warp latency.** On the T4, 320 GB/s ≈ 200 bytes per core clock
> for the *whole GPU* ≈ 5 bytes per clock per SM. Every wasted byte in a sector spends that shared budget.

> **9. Loading a 2-D tile:** a warp stays one contiguous run only if the tile width is a multiple of 32; otherwise
> some warps straddle two tile rows (2 runs, more sectors). Jumps *between* warps are free. Idle lanes come from
> `BM*BK % threads` (last pass) and from the bounds check at the matrix edge. Details and a summary table:
> [`CUDA_CPP_GENERAL.md` §9](CUDA_CPP_GENERAL.md#9-matrices-in-memory-1-d-storage-2-d-view) *(added 2026-11-10)*.

---

## Q1. I understand that DRAM reads/writes have to be coalesced so that a warp of threads reads contiguous memory locations. I want to understand reads and writes to/from shared memory in more detail. I believe shared memory is divided into 32 banks, and the 32 threads of a warp write to these 32 different banks.

**Answer.** The picture is right in outline, but one detail changes how you reason about it: **the address each thread accesses decides which bank it hits, not its lane number.** Lane 5 has no fixed bank.

### How banks work

Shared memory is split into **32 banks, each 4 bytes wide**. Consecutive 4-byte words go to consecutive banks, wrapping around:

```
bank = (byte_address / 4) % 32
```

So `float` index 0 → bank 0, index 31 → bank 31, index 32 → bank 0 again. Each bank can serve **one 4-byte word per cycle**. When a warp issues a shared-memory load or store, the hardware looks at all 32 addresses at once:

| Situation | Result |
|---|---|
| All 32 addresses in different banks | **1 pass** (full speed). Order doesn't matter, addresses needn't be contiguous. |
| Several *different* addresses in the same bank | **Bank conflict**: n-way conflict → n passes |
| Several threads reading the *same* word | **Broadcast**: one read delivered to all, no conflict |

### Common access patterns (from the interactive widget)

| Pattern (thread `t` reads float index…) | Passes | Why |
|---|---|---|
| `t` (a row) | 1 | 32 consecutive words → 32 banks |
| `2·t` (stride 2) | 2 | only even banks used, 2 words each |
| `t·32` (column of `[32][32]`) | **32** | every address is a multiple of 32 → all bank 0 |
| `t·33` (column of `[32][33]`) | 1 | padding shifts each row by one bank → bank `t` |
| `t·32 + (t ^ row)` (XOR-swizzled column) | 1 | data stored scrambled so a column spans all banks |
| `5` (everyone reads one word) | 1 | broadcast |
| Matmul `As[ty][kk]` (16×16 block) | 1 | 2 distinct words in 2 banks, each broadcast to 16 threads |
| Matmul `Bs[kk][tx]` (16×16 block) | 1 | 16 distinct words in 16 banks, each broadcast to 2 threads |

> **Key conclusion:** the tiled matmul's shared-memory reads are conflict-free: `As[ty][kk]` is a broadcast and `Bs[kk][tx]` reads consecutive words.

### Extra details

- **Writes** follow the same rule. If several threads write *different values* to the same word, one wins (unspecified which). No broadcast for writes.
- **Wider accesses** (`float2` = 8 B, `float4` = 16 B): a warp asks for 256 or 512 bytes, more than the 128 bytes the banks deliver per cycle, so the hardware splits it into half- or quarter-warps automatically. Contiguous chunks stay conflict-free.
- **In Triton**, you never choose the shared layout. When `tl.dot` stages tiles through shared memory, the compiler picks a **swizzled** layout. In TTGIR it appears as `#shared<{vec = …, perPhase = …, maxPhase = …}>`.
- **Measuring conflicts** with Nsight Compute:

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum python your_script.py
```

---

## Q2. I didn't follow completely. Show me an example, say the transpose of a matrix, where we load the tile from DRAM using contiguous addresses and then write to shared memory. Then we also need to read from shared memory into registers.

**Answer.** The transpose has five steps. (The interactive stepper used an 8×8 tile, 8 threads, 8 banks; the real hardware is 32×32 with 32 threads and 32 banks, same logic.)

| Step | Operation | Access pattern | Cost |
|---|---|---|---|
| 1 | DRAM → registers: thread `t` reads `A[r][t]` | contiguous row | coalesced |
| 2 | registers → shared: thread `t` writes `tile[r][t]` | row of tile → 32 banks | 1 pass |
| 3 | Repeat for all rows, then `__syncthreads()` | — | — |
| 4 | shared → registers: thread `t` reads `tile[t][r]` | **column** of tile | 32 passes without padding, **1 pass with padding** |
| 5 | registers → DRAM: thread `t` writes `B[r][t]` | contiguous row | coalesced |

### CUDA code (32×32 tile, `blockDim = (32, 32)`)

Each warp is one value of `threadIdx.y`, so each warp handles one row of the tile.

```cpp
__global__ void transpose(const float* in, float* out, int N) {
    __shared__ float tile[32][32 + 1];          // +1 = the padding column

    int x = blockIdx.x * 32 + threadIdx.x;      // column in A
    int y = blockIdx.y * 32 + threadIdx.y;      // row in A

    // Steps 1+2: DRAM → register → shared
    //   in[y*N + x]: warp reads 32 neighbouring floats → coalesced
    //   tile[ty][tx]: word ty*33 + tx → all banks different → 1 pass
    tile[threadIdx.y][threadIdx.x] = in[y * N + x];

    __syncthreads();                            // Step 3: whole tile loaded

    // swap block coordinates: this block writes the mirrored tile of B
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // Step 4: shared → register, reading a COLUMN of the tile
    //   tile[tx][ty]: word tx*33 + ty → bank (tx + ty) % 32 → all different → 1 pass
    //   (with tile[32][32]: word tx*32 + ty → same bank for all 32 → 32 passes)
    float v = tile[threadIdx.x][threadIdx.y];

    // Step 5: register → DRAM, 32 neighbouring floats → coalesced
    out[y * N + x] = v;
}
```

*(Bounds checks omitted. Add `if (x < N && y < N)` around the load and store when N isn't a multiple of 32.)*

> **Key conclusion:** a transpose has to read a column *somewhere*. Reading a column from DRAM ruins coalescing (expensive).
> So do every DRAM access as a contiguous row, and move the column access into shared memory, where it's only a
> bank-conflict problem that one padding float per row fixes.

**Why padding works:** without padding, `tile[t][0]` is word `32·t` → bank 0 for every `t`. With 33 floats per row, `tile[t][0]` is word `33·t`, and `33·t % 32 = t` → thread `t` lands in bank `t`.

---

## Q3. So padding a float per row is *one of the methods* to avoid bank conflicts? Could we use a circular shift by 1 per row, etc.?

**Answer.** Yes. Your circular shift works too. The general name is **swizzling**: store the tile in a scrambled order so that both rows and columns spread across all banks.

### Circular shift (rotation)

Store logical `(r, c)` at physical column `(c + r) % 32`:

```cpp
__shared__ float tile[32][32];                                          // no padding
tile[threadIdx.y][(threadIdx.x + threadIdx.y) & 31] = in[y * N + x];   // store row
__syncthreads();
float v = tile[threadIdx.x][(threadIdx.y + threadIdx.x) & 31];         // read column
```

Both accesses land in bank `(tx + ty) & 31`: all different within a warp. No wasted memory.

### XOR swizzle (what libraries use)

Store `(r, c)` at physical column `c ^ r`:

```cpp
tile[threadIdx.y][threadIdx.x ^ threadIdx.y] = in[y * N + x];
float v = tile[threadIdx.x][threadIdx.y ^ threadIdx.x];
```

4-bank illustration (each cell = which logical column is stored in that bank):

```
   Rotation (c + r) % 4        XOR  c ^ r
   bank: 0  1  2  3            bank: 0  1  2  3
row 0:   0  1  2  3            row 0:   0  1  2  3
row 1:   3  0  1  2            row 1:   1  0  3  2
row 2:   2  3  0  1            row 2:   2  3  0  1
row 3:   1  2  3  0            row 3:   3  2  1  0
```

Every logical column appears once in each bank column → column reads are conflict-free, and each row is a permutation of banks → row reads are conflict-free.

### Padding vs swizzling

| | Padding `[32][33]` | Rotation / XOR swizzle |
|---|---|---|
| Wasted shared memory | 1/32 ≈ 3% | none |
| Index math | none | one add/XOR per access |
| Rows stay 16-byte aligned | **no** (33 floats = 132 bytes) | yes |
| Works with `float4` / `ldmatrix` | awkward | yes (swizzle 16-byte chunks) |

> **Key conclusion:** padding is the simple fix; swizzling is what fast kernels use, mainly because it keeps rows 16-byte
> aligned for vectorized loads. XOR is preferred because it's easy to apply at chunk granularity and is its own inverse.
> CUTLASS's `Swizzle<B, M, S>` and Triton's `vec` / `perPhase` / `maxPhase` are XOR schemes.

**Other ways around bank conflicts:** change the access pattern so warps read rows; use warp shuffles (`__shfl_sync`) instead of shared memory for small transposes; or use `ldmatrix.trans` (available on sm_75) to transpose 8×8 tiles on the way into registers.

---

## Q4. If we have a 64×64 tile, how is it allocated across the banks without swizzling? The first 32 columns of row 0 go to the 32 different banks. What happens to the next 32 columns of row 0? And what happens with swizzling?

**Answer.** Shared memory has no rows or columns. It's **one long line of 4-byte words**, and the banks take turns. The 2D tile is just a numbering of those words.

### Without swizzling

Row-major `float tile[64][64]`: element `(r, c)` is word `64·r + c`, so

```
bank = (64·r + c) % 32 = c % 32      (64·r is a multiple of 32)
```

- Row 0, cols 0–31 → banks 0–31 (first sweep)
- Row 0, cols 32–63 → banks 0–31 again (second sweep)
- Row 1 starts at word 64 = bank 0, and repeats

> **Key conclusion:** every row takes exactly two sweeps, and **the bank depends only on the column**.

Scaled-down view (4 banks, 8-wide tile):

```
                  bank 0  bank 1  bank 2  bank 3
row 0, cols 0–3:   c0      c1      c2      c3
row 0, cols 4–7:   c4      c5      c6      c7
row 1, cols 0–3:   c0      c1      c2      c3
row 1, cols 4–7:   c4      c5      c6      c7
```

For a warp:
- 32 consecutive elements of a row → 1 pass
- All 64 elements of a row (e.g. `float2` per thread) → 256 bytes → 2 passes. **Not a conflict**: 128 bytes per pass is simply the banks' maximum.
- A column, `tile[t][c]` for t = 0..31 → all in bank `c % 32` → **32-way conflict**

### With an XOR swizzle

Store `(r, c)` at physical column `c ^ (r % 32)`. XOR only changes the low 5 bits of `c`, so each element stays in its own half (0–31 or 32–63) but gets shuffled within it:

```
bank = (c % 32) ^ (r % 32)
```

```
          bank 0  bank 1  bank 2  bank 3
row 0:     c0      c1      c2      c3     |  c4  c5  c6  c7
row 1:     c1      c0      c3      c2     |  c5  c4  c7  c6
row 2:     c2      c3      c0      c1     |  c6  c7  c4  c5
row 3:     c3      c2      c1      c0     |  c7  c6  c5  c4
```

A warp reading column `c` for rows `t = 0..31` hits banks `(c % 32) ^ t`: 32 different banks.

```cpp
__shared__ float tile[64][64];
tile[r][c ^ (r & 31)] = value;        // write
float v = tile[r][c ^ (r & 31)];      // read: same formula both ways
```

With padding instead, `tile[64][65]`: `bank = (65·r + c) % 32 = (r + c) % 32`, which also spreads a column across banks.

**Chunk-level swizzling in real kernels:** a 64-float row is 16 chunks of 4 floats (16 bytes); the 32 banks hold 8 chunks per sweep, so `physical_chunk = chunk ^ (r % 8)`. Each chunk stays contiguous, so `float4` / `ldmatrix` still work. That's Triton's `vec` parameter.

---

## Q5. For the "pad by 1 float" method, are you saying we populate the 32×32 tile from DRAM into a 32×33 matrix in shared memory?

**Answer.** Yes, exactly.

```cpp
__shared__ float tile[32][33];   // 32 rows × 33 columns in shared memory
tile[threadIdx.y][threadIdx.x] = in[y * N + x];   // threadIdx.x is 0..31, column 32 never touched
```

> **Your understanding (confirmed):** the 32×32 data goes into columns 0–31 of a 32×33 shared array.
> Column 32 is **never written and never read**. It holds whatever junk was there.

- Nothing changes in DRAM; no extra data is loaded.
- You index normally. The compiler computes `row * 33 + col` instead of `row * 32 + col`.
- Cost: 32 × 33 × 4 = 4,224 bytes instead of 4,096 → 128 bytes (≈3%) wasted per tile.

---

## Q6. You said a 4×4 tile stored as `tile[4][5]`, but the table shows a 5×4 tile?

**Answer.** The table wasn't the tile's shape. It showed the same 20 words **arranged by bank**.

The **logical** tile is 4 rows × 5 columns:

```
         col 0   col 1   col 2   col 3   col 4
row 0:   (0,0)   (0,1)   (0,2)   (0,3)    pad
row 1:   (1,0)   (1,1)   (1,2)   (1,3)    pad
row 2:   (2,0)   (2,1)   (2,2)   (2,3)    pad
row 3:   (3,0)   (3,1)   (3,2)   (3,3)    pad
```

**In memory** it's one line of 20 words:

```
word:   0     1     2     3     4    5     6     7     8     9    10    ...
       (0,0) (0,1) (0,2) (0,3) pad  (1,0) (1,1) (1,2) (1,3) pad  (2,0)  ...
```

**By bank** (4 banks), cut every 4 words → 20 / 4 = 5 sweeps:

```
                 bank 0   bank 1   bank 2   bank 3
words 0–3:       (0,0)    (0,1)    (0,2)    (0,3)
words 4–7:        pad     (1,0)    (1,1)    (1,2)
words 8–11:      (1,3)     pad     (2,0)    (2,1)
words 12–15:     (2,2)    (2,3)     pad     (3,0)
words 16–19:     (3,1)    (3,2)    (3,3)     pad
```

> **Key conclusion:** each line of the bank table is one **sweep across the banks**, not a row of the tile.
> Because tile rows (5 words) don't line up with sweeps (4 words), each row's pad pushes the next row one bank
> further, which spreads every column across all banks.
> For any element: `word = 5·r + c`, `bank = word % 4`.

---

## Q7. So we load the DRAM 4×4 tile into the shared 4×5 tile, with the last column as some random value. How do we access column 0 for all rows, i.e. `tile_shared[t][0]`? Do we just do `tile_shared[threadIdx.y][0]`?

**Answer.** You index normally (the compiler handles the padding), but for a **warp** to read a column, the thread index in the **row** slot must be the one that changes **within a warp**. With `blockDim = (32, 32)`, that's `threadIdx.x`:

```cpp
float v = tile[threadIdx.x][0];   // thread t reads tile[t][0] → column 0
```

| Code | What one warp reads | Banks |
|---|---|---|
| `tile[threadIdx.x][0]` | rows 0–31 of column 0 (a column) | 32 different banks thanks to the pad → 1 pass |
| `tile[threadIdx.y][0]` | the same single element 32 times | one word, broadcast → 1 pass, but not a column |
| `tile[threadIdx.y][threadIdx.x]` | one row | 32 different banks → 1 pass |

In the transpose, `tile[threadIdx.x][threadIdx.y]`: within a warp `threadIdx.y` is fixed (picks the column) and `threadIdx.x` sweeps the rows.

---

## Q8. For a 32×32 thread block, both `threadIdx.x` and `threadIdx.y` vary from 0 to 31, so why use `threadIdx.x` to sample the rows of a column and not `threadIdx.y`? And for a 16×16 block, how are warps defined: is the first warp `threadIdx.y = 0, 1` with `threadIdx.x = 0…15`, the second warp `threadIdx.y = 2, 3`, and so on?

**Answer.** Yes, your 16×16 understanding is exactly right.

### How warps are formed

CUDA flattens the block into a linear ID, **x changing fastest**, then cuts it into groups of 32:

```
linear_id = threadIdx.x + threadIdx.y * blockDim.x  (+ threadIdx.z * blockDim.x * blockDim.y)
```

**16×16 block:** `linear_id = x + 16·y`

```
            x = 0  1  2  ...  15
y = 0  →  IDs   0  1  2  ...  15  ┐ warp 0
y = 1  →  IDs  16 17 18  ...  31  ┘
y = 2  →  IDs  32 33 34  ...  47  ┐ warp 1
y = 3  →  IDs  48 49 50  ...  63  ┘
  ...
y = 14, 15                          warp 7
```

**32×32 block:** `linear_id = x + 32·y` → each warp is exactly one value of `y`, with `x = 0…31`.

> **Your understanding (confirmed):** in a 16×16 block, warp 0 = `y ∈ {0, 1}` × `x ∈ 0..15`, warp 1 = `y ∈ {2, 3}`, etc.

### Why `threadIdx.x`

Across the whole block both indices take every value, but **bank conflicts and coalescing are decided per warp instruction**. Inside a warp of a 32×32 block, `threadIdx.y` is fixed and only `threadIdx.x` changes. So whatever `threadIdx.x` indexes is the direction the warp spreads across memory. `threadIdx.y` still matters: it picks *which* column each warp reads, so all 32 warps together cover the tile.

### Caution for 16×16 blocks

A warp now spans two values of `y`. For a 16×16 transpose with `tile[16][17]`, the read `tile[threadIdx.x][threadIdx.y]` has one warp reading two columns at once; banks are `(17·x + y) % 32`, and `(x=0, y=0)` and `(x=15, y=1)` both land in bank 0 → a **2-way conflict**.

> **Key conclusion:** the padding trick was worked out for a warp that is exactly one 32-wide row. Whenever `blockDim.x`
> isn't 32, recompute the banks for the actual 32 threads in a warp.

---

## Q9. When we write `tile[threadIdx.y][0]`, this operation happens for all threads in parallel, right? Every thread runs the same kernel at its own thread ID. So doesn't it mean `threadIdx.y = 0` samples `tile[0][0]`, `threadIdx.y = 1` samples `tile[1][0]`, …, `threadIdx.y = 31` samples `tile[31][0]`? Am I completely wrong, or does this only matter at the time of reads/writes?

**Answer.** You're not wrong. Your reading is correct **at the block level**: across the whole block, all of column 0 gets read. The table was describing **one warp**.

The hardware runs 32 warps, each issuing one instruction for its 32 threads together. In a 32×32 block those threads share one `threadIdx.y`:

```
v = tile[threadIdx.y][0];

warp 0  (y = 0, x = 0..31):  all 32 threads → tile[0][0]    one word, broadcast
warp 1  (y = 1, x = 0..31):  all 32 threads → tile[1][0]    one word, broadcast
...
warp 31 (y = 31, x = 0..31): all 32 threads → tile[31][0]   one word, broadcast
```

The column is read, but **spread over 32 separate warp instructions**, one element each.

```
v = tile[threadIdx.x][0];

warp 0  (y = 0, x = 0..31):  thread x → tile[x][0]          the whole column in one instruction
warp 1  (y = 1, x = 0..31):  thread x → tile[x][0]          the whole column again
...
```

> **Key conclusion — two levels:**
> - **What gets computed** is per thread. Your mental model is the right one for correctness.
> - **How fast it runs** is per warp. *Every* instruction (not just reads/writes) is issued for a whole warp;
>   memory is just where the grouping shows up in performance.
>
> For performance, ask: *for the 32 threads that share a warp, which 32 addresses does this line produce?*

Also, "all threads run in parallel" is only roughly true: each SM partition issues one warp instruction per clock, so warps take turns and overlap.

---

## Q10. So in both cases, `v = tile[threadIdx.y][0]` and `v = tile[threadIdx.x][0]`, there is wastage? In the first, each warp reads only 1 element, so we need all 32 warps to read the full column. In the second, each warp reads the full column, so the other 31 warps are redundant. Am I right? And is this true whether the tile is in shared memory or DRAM?

**Answer.** Partly. Both lines read only 32 distinct values with 1,024 threads (each value 32 times), but that's because **each expression uses only one of the two thread indices**, not because of warps. Warps only decide how the duplicates are grouped:

- `tile[threadIdx.y][0]`: duplicates **inside** each warp
- `tile[threadIdx.x][0]`: duplicates **across** warps

A real kernel uses both indices, e.g. `tile[threadIdx.x][threadIdx.y]`, so every element is read exactly once.

> **Key conclusion:** duplicate reads aren't always waste. If every thread needs the value, a broadcast is the point:
> `As[ty][kk]` in the tiled matmul is exactly this pattern, and it's how tiling reuses data.

### Cost in shared memory

| | Per warp instruction | Whole block |
|---|---|---|
| `tile[threadIdx.y][0]` | 1 word, broadcast → 1 pass | 32 instructions × 1 pass |
| `tile[threadIdx.x][0]`, `[32][33]` | 32 banks → 1 pass | 32 instructions × 1 pass |
| `tile[threadIdx.x][0]`, `[32][32]` | all bank 0 → 32 passes | 32 × 32 passes |

### Cost in DRAM

| | Per warp instruction |
|---|---|
| `in[threadIdx.y * N + 0]` | one address → one 32-byte sector, broadcast. Cheap. |
| `in[threadIdx.x * N + 0]` | 32 different rows → 32 sectors, 4 useful bytes of every 32 → uncoalesced |

> **Key conclusion:** in shared memory, both cost the same (with padding). In DRAM, the column read is the expensive one,
> and padding can't fix it. Repeats across warps are mostly caught by L1/L2, but the first read is still uncoalesced.

---

## Q11. For `tile[threadIdx.y][0]`, all threads of the warp compete for the same value, which is in the same bank. Why is there no bank conflict? Also, I'm confused now: I was using `threadIdx.x` for columns and `threadIdx.y` for rows, but the flattening of threads into warps has confused me.

### Why no conflict

A bank conflict means **different words in the same bank**, not "many threads in the same bank". A bank delivers one word per cycle but can hand that word to any number of threads (the crossbar copies it out).

Librarian analogy (each bank = a librarian fetching one book per trip):
- 32 people want **the same book** → one trip, everyone gets a copy (`tile[threadIdx.y][0]`)
- 32 people want **32 different books on her shelf** → 32 trips (`tile[threadIdx.x][0]` with `[32][32]`)
- 32 people want books on **32 different shelves** → all librarians go at once, 1 trip (padded case)

> **Key conclusion:** cost = the largest number of **different words** any one bank must supply.
> Threads asking for the same word don't add to it.

### Untangling `threadIdx.x` vs `threadIdx.y`

They don't mean "column" and "row". They're two counters, and **you** decide which matrix dimension each indexes.

- **Correctness:** does every element get handled exactly once? Any mapping can work.
- **Speed:** within one warp, which addresses come out? With `blockDim.x = 32`, `threadIdx.x` is the index that spreads a warp across memory.

The habit "x on columns" comes from DRAM: neighbouring columns are neighbouring addresses, so it makes a warp's reads coalesced. It's a convention for speed, not a CUDA rule.

**The check for any line:** freeze `threadIdx.y`, let `threadIdx.x` run 0–31, list the 32 addresses.

| Transpose line | `tx` sits on… | One warp touches | Result |
|---|---|---|---|
| `in[y*N + x]` | columns of A | 32 neighbouring floats in a row of A | coalesced |
| `tile[ty][tx] = …` | columns of the tile | a row of the tile | 32 banks, 1 pass |
| `v = tile[tx][ty]` | **rows** of the tile | a column of the tile | 32 banks with padding, 1 pass |
| `out[y'*N + x']` | columns of B | 32 neighbouring floats in a row of B | coalesced |

> **Key conclusion:** in the transpose, `threadIdx.x` indexes columns everywhere except the shared-memory read,
> where it deliberately indexes rows. **That one swap is the transpose.**

---

## Q12. So for performance optimization we need to check: (1) when reading/writing DRAM, is every thread of a warp accessing contiguous memory (which it does if we index as `in[y*N + x]`); (2) when reading shared memory, if threads of a warp access different words (like rows of a tile for a given column), those words should be in different banks — it still works if they're in the same bank, but not optimally. Am I right?

**Answer.** Yes, these are the two right checks.

> **Your understanding (confirmed), with refinements:**
>
> **1. DRAM — count sectors, not contiguity.** Cost = number of 32-byte sectors the warp's addresses touch.
> - Contiguous and aligned → 4 sectors for 32 floats. Perfect.
> - Order within that range doesn't matter.
> - Misaligned start (e.g. N = 2003, rows not on 128-byte boundaries) → usually 5 sectors. Minor.
> - Strided/scattered (column reads) → up to 32 sectors. Avoid.
> - Applies to **writes** too.
>
> **2. Shared memory — cost = most different words any one bank serves.**
> - Same word from many threads → broadcast, free.
> - n different words in one bank → n passes. Correct but n× slower.
> - Applies to **writes** too: in the transpose, both `tile[ty][tx]` and `tile[tx][ty]` must be checked.

**Priority:** a badly uncoalesced DRAM access usually costs far more than a bank conflict (DRAM is hundreds of cycles away and limited to 320 GB/s on the T4). **Fix coalescing first.**

**Beyond these two:** data reuse (arithmetic intensity), occupancy, register spills (`n_spills`), and instruction choice (`float4`, tensor cores). Nsight Compute reports all of them.

---

## Q13. I follow shared memory access now, but not DRAM access fully. I assumed a warp must access 32 consecutive locations, like a row of a matrix. What is this 32-byte sector? Each thread accesses a float and 32 threads access 32 consecutive floats — is that not right?

**Answer.** Your picture is right for the ideal case. What's missing is how the hardware fetches.

The memory system never fetches 4 bytes on its own. Data moves between DRAM and L2 in **32-byte sectors** (8 floats). When a warp executes a load, the hardware:

1. collects the warp's 32 addresses,
2. works out which 32-byte sectors contain them,
3. fetches each of those sectors **once**,
4. gives each thread its 4 bytes.

> **Key conclusion:** the true cost of a warp's DRAM access is **the number of distinct sectors it touches**.
> "32 consecutive floats" is the best case: 128 bytes = exactly 4 sectors, every byte used.

### Access patterns (from the interactive widget)

| Pattern (thread `t` reads float…) | Sectors | Bytes fetched / used | Efficiency |
|---|---|---|---|
| `t` (row, aligned) | 4 | 128 / 128 | 100% |
| `perm(t)` (row, shuffled) | 4 | 128 / 128 | 100% |
| `t + 3` (row, misaligned) | 5 | 160 / 128 | 80% |
| `2·t` (stride 2) | 8 | 256 / 128 | 50% |
| `t·64` (column of a 64-wide matrix) | 32 | 1,024 / 128 | 12.5% |
| `5` (all same float) | 1 | 32 / 4 | broadcast, cheap |

**Thinking in sectors covers more cases:**
- 32 threads × `half` (2 bytes) = 64 bytes = 2 sectors, perfect.
- 32 threads × `float4` (16 bytes) = 512 bytes = 16 sectors, all used → why vector loads are efficient.
- Partial sectors on writes still cost a full sector's transfer.

### Shared memory vs DRAM

| | Shared memory | DRAM (global memory) |
|---|---|---|
| Unit the hardware works in | 4-byte word per bank | 32-byte sector |
| Cost of one warp access | most different words in any one bank | number of distinct sectors touched |
| Same address from many threads | broadcast, free | one sector, free |
| What hurts | different words in the same bank | addresses spread over many sectors |

> **Key conclusion:** DRAM wants a warp's addresses **packed close together**; shared memory wants them **spread across banks**.
> Contiguous rows satisfy both, which is why row access is the safe default everywhere.

---

## Q14. My understanding: DRAM operates on a 256-bit (32-byte) bus, moving 32 bytes per clock, and the more data threads request, the more clock cycles to fetch it. So 32 threads × 4 bytes = 128 bytes → 4 trips of 32 bytes. 2 bytes per thread → 64 bytes → 2 trips. A `float4` read → 512 bytes → 16 trips. Is that it?

**Answer.** Your counts are right (4, 2 and 16 sectors), and "more sectors = more data on the bus" is right. Two details are off:

### 1. A sector isn't one clock on the whole 256-bit bus

The T4's 256-bit bus isn't one wide pipe; it's **16 independent narrow channels**:

```
8 GDDR6 chips × 2 channels each = 16 channels, each 16 bits wide
16 channels × 16 bits = 256-bit bus
```

A sector comes from **one** channel, which always sends a burst of 16 transfers: 16 × 16 bits = 256 bits = **32 bytes**. That burst is why a sector is 32 bytes. Addresses are interleaved across channels, so a warp's 4 sectors usually come back **in parallel**, not as 4 sequential trips.

### 2. The cost is shared bandwidth, not cycles per warp

From one warp's view, 1 or 4 sectors take about the same time: ~400+ cycles of latency, mostly waiting. The real limit is that **all 40 SMs share 320 GB/s**:

```
320 GB/s ÷ 1.59 GHz ≈ 200 bytes per core clock for the whole GPU (~6 sectors/clock)
÷ 40 SMs            ≈ 5 bytes per clock per SM  (barely one float, for 64 FP32 cores)
```

| Pattern | Useful bytes | Bytes moved | Bandwidth used usefully |
|---|---|---|---|
| 32 consecutive floats | 128 | 128 (4 sectors) | 100% |
| stride 2 | 128 | 256 (8 sectors) | 50% |
| column read | 128 | 1,024 (32 sectors) | 12.5% |

> **Key conclusion — the corrected mental model:**
> - **Latency** (hundreds of cycles) is hidden by having many warps.
> - **Bandwidth** (320 GB/s shared by everyone) is the hard limit, measured in whole sectors.
> - **Coalescing** is about not spending that limit on bytes nobody uses.
>
> A column-read kernel doesn't make one warp wait 8× longer; it makes the **whole GPU** run out of DRAM bandwidth 8× sooner.

---

## Q15. Say I have a 16×16 thread block, but the tile is 64×64 (`A_smem[64][65]` after padding). If I do `A_smem[threadIdx.x][threadIdx.y]`, I only access 16 rows for each of 16 columns, but I need all 64 rows of each column. Do I flatten the 256 threads and reshape them to 4×64? Is this correct?

```cpp
lin_tid = threadIdx.y * blockDim.x + threadIdx.x;
row = lin_tid / 64;
col = lin_tid % 64;
for (i = 0; i < 16; i++)
{
    A_smem[col][row + 4*i];
}
```

**And do we have to manage warps manually, or just focus on thread/block indices and let the compiler map to warps?**

**Answer.** Yes, it's correct. Flattening and reshaping is the standard way to handle a tile bigger than the block. (Note: `row + 4*i`, not `row+4i`.)

> **Your understanding (confirmed):** flatten the 256 threads into a linear ID, then reshape it to whatever shape the
> access needs.

### Check 1: every element read exactly once

`col` covers 0–63, `row` covers 0–3, and over `i = 0…15` the second index `row + 4*i` covers 0–63 → all 64×64 elements, once each.

### Check 2: what one warp does

Warp 0 = `lin_tid` 0–31 → `row = 0` (fixed), `col = 0…31` → `A_smem[0..31][4*i]` = 32 rows of one column. Banks:

```
word = 65·col + c   →   bank = (col + c) % 32   → 32 different banks, 1 pass
```

Warp 1 (`lin_tid` 32–63) gets `col = 32…63`, also conflict-free. All 8 warps are conflict-free.

> **Key conclusion:** whatever you compute with `% width` is the index that varies fastest **within a warp**.
> Put it on the dimension you want the warp to spread across.

Clearer naming (your `col` actually indexes the **rows** of `A_smem`):

```cpp
int tid  = threadIdx.y * blockDim.x + threadIdx.x;   // 0..255
int r    = tid % 64;          // which row of the tile: varies fastest within a warp
int c0   = tid / 64;          // starting column: 0..3

for (int c = c0; c < 64; c += 4)        // 256 threads / 64 rows = 4 columns per pass
    v = A_smem[r][c];                    // warp reads 32 rows of column c
```

### Same trick for loading the tile from DRAM

Here put the fastest-varying index on **columns**, so each warp reads a contiguous run of a DRAM row:

```cpp
int c  = tid % 64;            // column: varies fastest → coalesced DRAM read
int r0 = tid / 64;            // 0..3

for (int r = r0; r < 64; r += 4)
    A_smem[r][c] = A[(tile_row + r) * N + (tile_col + c)];
```

Warp 0 reads row `r`, columns 0–31 (128 contiguous bytes = 4 sectors) and writes `A_smem[r][0..31]` → banks `(r + c) % 32`, all different.

> **Key conclusion:** decide **separately for each access** which dimension gets `tid % width`:
> columns for DRAM traffic, whichever direction you need for shared memory.

### Do you manage warps yourself?

- **For correctness, no.** Write per-thread code with `threadIdx` / `blockIdx`. The hardware forms warps with a fixed rule (`x + y·Dx + z·Dx·Dy`, groups of 32); the compiler doesn't rearrange it.
- **For performance, you must know that rule**, as above.

Places where warps appear in the code itself:
- **Warp-level operations:** `__shfl_sync`, `__ballot_sync`, warp reductions, `wmma` / `mma` tensor-core instructions.
- **Divergence:** threads in one warp taking different sides of an `if` run both sides one after the other. Branching on `tid / 32` (warp ID) is free; branching on `tid % 2` splits every warp.
- **Don't assume lockstep:** since Volta, lanes can be scheduled independently. Use `__syncwarp()` (or `__syncthreads()`) when lanes exchange data through shared memory.

---

## One-page checklist

| Check | Question to ask | Good | Bad |
|---|---|---|---|
| DRAM coalescing | How many 32-byte sectors do the warp's 32 addresses touch? | 4 for 32 floats (contiguous, aligned) | up to 32 (column / scattered) |
| Shared-memory banks | What's the most *different* words any one bank must serve? | 1 | n → n-way conflict |
| Warp membership | Which 32 threads share a warp? | 32 consecutive linear IDs, `x` fastest | assuming `threadIdx.y` varies within a warp of a 32-wide block |
| Index mapping | Which index gets `tid % width`? | columns for DRAM; the needed direction for shared memory | same mapping everywhere without checking |
| Bank-conflict fix | Padding or swizzle? | `[32][33]` for simple kernels; XOR swizzle for `float4` / `ldmatrix` | unpadded column reads |
| Priority | What to fix first? | DRAM coalescing, then bank conflicts | — |

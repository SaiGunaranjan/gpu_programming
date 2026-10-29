# matmul: Triton learnings

**Date:** 2026-10-15
**Kernel studied:** [`matmul_triton_tiled.py`](matmul_triton_tiled.py) (originally `matrix_multiplication_tile.py`)
**Compiler output studied:** the kernel's PTX (`print(k.asm['ptx'])`) and TTGIR
(`print(k.asm['ttgir'])`), dumped to a text file (not in the repo).

The CUDA C++ side is in [`LEARNINGS_CUDA_CPP.md`](LEARNINGS_CUDA_CPP.md); cross-problem Triton
lessons go in [`learnings/TRITON_GENERAL.md`](../../learnings/TRITON_GENERAL.md).

---

## Summary

### Key facts so far

| Fact | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0 (`sm_120a`), **188 SMs** |
| Per-SM limits | 32 blocks max, 1,536 threads (48 warps), 65,536 registers, 100 KB shared memory |
| Triton version | 3.4.0 |
| Problem size | A = 8001×2003, B = 2003×4017, C = 8001×4017, all FP32 |
| Tile config | `BLOCK_SIZE_C_ROWS=64`, `BLOCK_SIZE_C_COLS=64`, `BLOCK_SIZE_N=32`, `num_warps=4` (default) |
| Grid | (126, 63) = **7,938 pids / thread blocks** |
| Threads per block | 4 warps × 32 = **128** |
| Outputs per thread | 64×64 / 128 = **32** |
| Shared memory per block | 2 buffers × (64×32 + 32×64) × 4 B = **32 KB** |
| Blocks per SM | **2** (limited by registers: `k.n_regs`=196; shared memory alone would allow 3), see Q9 |
| Tensor Cores used? | **No.** `allow_tf32=False` means FP32 FMA on CUDA cores (1,024 `fma.rn.f32`, 0 `mma`) |
| Correctness | `OK  max abs diff = 0.0` vs `torch.matmul` |

### Open experiments (suggested, not yet done)

1. ~~Register count~~ done: `n_regs = 196` → 2 blocks/SM (Q9). Still check `k.n_spills`.
2. Set `allow_tf32=True` and look for a `#mma` / `nvidia_mma` layout in TTGIR and `mma.sync` in
   PTX. `assert_close` may need a looser tolerance.
3. Use sizes that are multiples of 16 (e.g. 8192 / 6144 / 4096) and check whether the load layouts
   become `sizePerThread=[1,4]` with 16-byte `cp.async`.
4. Try `num_warps=8` and see how the `#blocked2` accumulator layout changes.
5. Add `@triton.autotune` (configs in Q13) and compare the winner's time, `n_regs` and `n_spills` with the 64×64 baseline.

---

## The kernel (as studied)

```python
@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    BLOCK_SIZE_C_ROWS: tl.constexpr,
    BLOCK_SIZE_C_COLS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)

    offsets_a_rows = (pid_y*BLOCK_SIZE_C_ROWS + tl.arange(0,BLOCK_SIZE_C_ROWS))
    abs_offsets_a_rows = offsets_a_rows*stride_am

    offsets_b_cols = (pid_x*BLOCK_SIZE_C_COLS + tl.arange(0,BLOCK_SIZE_C_COLS))
    abs_offsets_b_cols = offsets_b_cols*stride_bk

    acc = tl.zeros((BLOCK_SIZE_C_ROWS,BLOCK_SIZE_C_COLS),dtype=tl.float32)

    for i in range(0,tl.cdiv(N,BLOCK_SIZE_N)):
        offsets_a_cols_b_rows = i*BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
        offsets_a_cols =  offsets_a_cols_b_rows*stride_an
        offsets_a = abs_offsets_a_rows[:,None] + offsets_a_cols[None,:]
        mask_a = (offsets_a_rows[:,None] < M) & (offsets_a_cols_b_rows[None,:] < N)

        offsets_b_rows = offsets_a_cols_b_rows*stride_bn
        offsets_b = offsets_b_rows[:,None] + abs_offsets_b_cols[None,:]
        mask_b = (offsets_a_cols_b_rows[:,None] < N ) & (offsets_b_cols[None,:] < K)

        a_ = tl.load(a+offsets_a,mask=mask_a,other=0)
        b_ = tl.load(b+offsets_b,mask=mask_b,other=0)

        acc = tl.dot(a_,b_,acc,allow_tf32=False)

    offsets_c = offsets_a_rows[:,None]*stride_cm + offsets_b_cols[None,:]*stride_ck
    mask_c = (offsets_a_rows[:,None] < M) & (offsets_b_cols[None,:] < K)
    tl.store(c+offsets_c,acc,mask=mask_c)

# launch
grid = lambda meta: (triton.cdiv(M,meta["BLOCK_SIZE_C_ROWS"]), triton.cdiv(K,meta["BLOCK_SIZE_C_COLS"]))
k = matrix_multiplication_kernel[grid](..., BLOCK_SIZE_C_ROWS=64, BLOCK_SIZE_C_COLS=64, BLOCK_SIZE_N=32)
```

The naming is unusual. **`N` is the inner (reduction) dimension and `K` is C's column count.**

---

## Q1. Explain the PTX output (`print(k.asm['ptx'])`)

The file is the **PTX** Triton generated from the kernel. PTX is NVIDIA's GPU "assembly
language". The last line is the correctness check.

### File structure (3,164 lines)

| Lines | What it is | Need to read it? |
|---|---|---|
| 1–30 | Header and kernel inputs | Briefly |
| 30–2978 | **The kernel code** | Yes |
| 2981–3161 | Debug info (`.debug_abbrev`, `.b8 …`) | No. It only maps instructions to `.py` lines |
| 3164 | `OK  max abs diff = 0.0` | Yes: the kernel matches `torch.matmul` |

### Header

```
.target sm_120a                    → compiled for Blackwell (RTX PRO 6000)
.extern .shared ... global_smem[]  → uses shared memory
.reqntid 128                       → 128 threads per block (4 warps)
```
`param_0 … param_9` are the kernel arguments. Strides equal to 1 were specialised away, so there
are fewer params than arguments.

`.loc 1 47 21  // matrix_multiplication_tile.py:47:21` means "the following instructions come from
line 47 of the Python file".

### The kernel in 4 stages

**① Setup (lines ~30–850): "which tile am I?"**
- `mov.u32 %r366, %ctaid.x` is `tl.program_id(axis=0)`.
- `shl.b32 %r1, %r366, 6` is ×64 (`pid_y * BLOCK_SIZE_C_ROWS`, with 64 baked in).
- `%tid.x` is the thread ID within the block (0–127).
- Many `setp.lt.s32` instructions are the **masks**.
- 4× `cp.async.commit_group`: the first tiles are **prefetched** before the loop starts.

**② Main loop (lines 1011–2545): the `for i` loop**
```
$L__BB0_3:                 ← loop starts
  cp.async.wait_group 2    ← wait for the oldest prefetched tile
  bar.sync 0               ← all 128 threads sync
  ld.shared.v4.b32 ...     ← shared memory → registers (96 per iteration)
  fma.rn.f32 ...           ← acc += a*b (1,024 per iteration)
  cp.async ... commit      ← start fetching the NEXT tile
@%p91 bra $L__BB0_3        ← loop back
```
This is **software pipelining**: tile *i* is computed while tiles *i+1* and *i+2* load. Triton adds
this automatically.

**③ After the loop:** `cp.async.wait_group 0` drains the remaining loads.

**④ Store:** a shuffle through shared memory (`st.shared`/`ld.shared`), then 32×
`@%p92 st.global.b32` (the `@%p` predicate is `mask_c`).

### Main finding: no Tensor Cores

| Instruction | Count | Meaning |
|---|---|---|
| `fma.rn.f32` | **1,024** | FP32 multiply-add on CUDA cores |
| `mma.sync` | **0** | Tensor Core matmul |

1,024 = 32 outputs per thread × `BLOCK_SIZE_N`=32. The cause is `allow_tf32=False`: full FP32
precision can't use Tensor Cores. The result is exact but slower.

---

## Q2. Explain the TTGIR output (`print(k.asm['ttgir'])`)

### What TTGIR is

Triton lowers code in stages:
```
Python (@triton.jit) → TTIR → TTGIR → LLVM IR → PTX → cubin
```
- **TTIR**: tensor ops, no GPU details.
- **TTGIR** ("Triton GPU IR"): the same ops plus **how they map onto the GPU**: which thread owns
  which element, what goes in shared memory, pipelining, and whether `tl.dot` uses Tensor Cores.

It is still readable as whole tensors (`tensor<64x32xf32>`), so it's the best place to see
Triton's decisions.

### File structure (231 lines)

| Lines | Part |
|---|---|
| 1–6 | **Layout definitions** (`#blocked`, `#shared`) |
| 7–8 | Module/function header (`num-warps = 4`, `threads-per-warp = 32`, `target = "cuda:120"`) |
| 9–98 | Setup: offsets, masks, shared memory alloc, **prefetch of 2 tiles** |
| 99–145 | **Main loop** (`scf.for`) |
| 146–167 | Store to C |
| 168–230 | `#locN = loc(...)` → maps ops to lines in the `.py` file |
| 231 | `OK max abs diff = 0.0` |

### Reading one line

```
%11 = arith.muli %8, %10 : tensor<64xi32, #ttg.slice<...>> loc(#loc7)
result  op      inputs     type (64 ints + layout)           source line (#loc7 → line 25)
```

### Layouts (lines 1–6)

```
#blocked  = sizePerThread=[1,1], threadsPerWarp=[1,32], warpsPerCTA=[4,1]   → A tile load
#blocked1 = sizePerThread=[1,1], threadsPerWarp=[1,32], warpsPerCTA=[2,2]   → B tile load + C store
#blocked2 = sizePerThread=[4,4], threadsPerWarp=[2,16], warpsPerCTA=[4,1]   → accumulator acc
#shared   = swizzled_shared vec=1, maxPhase=1                               → plain smem, no swizzle
```

### Setup ops

| TTGIR op | Python |
|---|---|
| `tt.get_program_id x/y` | `tl.program_id(axis=0/1)` |
| `tt.make_range {0..64}` | `tl.arange(0,64)` |
| `tt.splat` | scalar → tensor of copies |
| `tt.expand_dims` + `tt.broadcast` | `[:, None]` / `[None, :]` |
| `arith.cmpi slt` + `arith.andi` | masks and `&` |
| `arith.addi %arg4, 31` → `divsi 32` | `tl.cdiv(N, 32)` |

- **Only 9 args (`%arg0`–`%arg8`)**: `stride_an`, `stride_bk` and `stride_ck` are 1, so they were
  baked in. `%arg3`=M, `%arg4`=N, `%arg5`=K, `%arg6`=stride_am, `%arg7`=stride_bn,
  `%arg8`=stride_cm. `tt.divisibility = 16` on the pointers means they are 16-byte aligned.
- `ttg.local_alloc : memdesc<2x64x32xf32>` and `<2x32x64xf32>`: **2 shared-memory slots each for
  A and B**. `ttg.async_copy_global_to_local` (→ `cp.async`) starts loading tiles 0 and 1 before
  the loop.

### Main loop

```
scf.for ... iter_args(acc, slot indices, 4 async tokens)
  async_wait {num = 2}   ← wait until tile i has arrived
  local_load → A, B      ← smem → registers in the dot_op layout
  tt.dot A, B, acc
  async_copy tile i+2    ← into the slot just freed
  scf.yield ...
```
This is **double buffering**. The slot index cycles 0, 1, 0, 1, ….

**Tensor Core check:** `tt.dot ... #ttg.dot_op<{opIdx = 0, parent = #blocked2}>`. The parent is a
plain `#blocked` layout, which means **FMA on CUDA cores**. With Tensor Cores the parent would be
`#ttg.nvidia_mma<...>` (`#mma`).

### Store

```
ttg.async_wait {num = 0}; ttg.local_dealloc ...
ttg.convert_layout acc: #blocked2 → #blocked1   ← reshuffle so writes are coalesced
tt.store ptrs, acc, mask
```

### What to look for in TTGIR

| Search for | Tells you |
|---|---|
| `#mma` / `nvidia_mma` | Tensor Cores used? (here: no) |
| `local_alloc` | Shared-memory buffers (here: 2 each for A and B) |
| `async_copy` / `async_wait` | Pipelining (here: yes) |
| `convert_layout` | Thread-to-thread data shuffles (these cost time) |
| Number of `%argN` | Which args were specialised away |

---

## Q3. Layout details: how are loads and compute distributed across threads?

### Q3a. "My pid tile is 64x64, so each thread block generates a 64x64 tile of C. Am I right?"

**Yes.** grid = `(cdiv(M,64), cdiv(K,64))`, so each program (= thread block / CTA) owns one 64×64
tile of C. A block has 4 warps × 32 = 128 threads.

### Q3b. "We load a 64x32 tile of A and a 32x64 tile of B. How is loading distributed across threads/warps?"

How to read a `#blocked` layout: it defines a **footprint**, the patch that all 128 threads cover
together. The footprint repeats until the tile is filled.
```
footprint rows = sizePerThread[0] × threadsPerWarp[0] × warpsPerCTA[0]
footprint cols = sizePerThread[1] × threadsPerWarp[1] × warpsPerCTA[1]
```

**A tile (64×32), `#blocked`:** footprint = 4×32, repeated 16 times down the rows.
```
                      col: 0  1  2 ... 31
row 0  → warp 0          L0 L1 L2 ... L31
row 1  → warp 1
row 2  → warp 2
row 3  → warp 3
row 4  → warp 0   (repeats)
...
row 63 → warp 3
```
Warp w loads rows w, w+4, …, w+60. Lane L loads column L. **16 elements per thread.**

**B tile (32×64), `#blocked1`:** footprint = 2×64, repeated 16 times.
```
          cols 0–31   cols 32–63
row 0      warp 0      warp 1
row 1      warp 2      warp 3
row 2      warp 0      warp 1   (repeats)
```
**16 elements per thread.**

The PTX confirms it: 16 + 16 = 32 `cp.async` per tile pair, × 3 copies of the code (2 prologue +
1 loop) = 96 `cp.async`.

Why this shape: neighbouring threads read neighbouring addresses, so a warp's reads merge into a
few wide memory transactions (**coalescing**). It is good for loading, not for computing.

### Q3c. "How many output elements does each thread compute? Is it 32 because 64×64/(4×32)?"

**Yes.** 4096 ÷ 128 = **32 outputs per thread.** Which 32 is set by `#blocked2`:
footprint = (4·2·4) × (4·16·1) = 32×64, repeated 2 times, so each thread owns **two 4×4 blocks**.
```
C tile (64×64); each cell = one thread's 4×4 block
             cols 0-3   4-7   ...  60-63    (16 lanes across)
rows  0-3    w0:L0      L1    ...  L15
rows  4-7    w0:L16     L17   ...  L31
rows  8-15   warp 1
rows 16-23   warp 2
rows 24-31   warp 3
---- repeats ----
rows 32-63   each thread's second 4×4 block
```
Example: warp 0, lane 0 owns C[0–3][0–3] and C[32–35][0–3].

Work per thread per loop step: 32 outputs × 32 k = **1,024 FMAs**. It needs 8 A-rows × 32 = 256
values plus 32 × 4 B-cols = 128 values = 384 floats = **96 `ld.shared.v4`**. Both numbers match
the PTX.

### Q3d. Why does the load layout differ from the compute layout?

The thread that loads `A[5][7]` (warp 1, lane 7) is not the thread that needs it. All 16 threads
covering row 5 need it. So shared memory sits in the middle:
```
global ──(#blocked/#blocked1, coalesced)──► shared memory
shared ──(local_load, dot_op layout)──► registers
registers ──(fma, #blocked2)──► acc
acc ──(convert_layout → #blocked1)──► coalesced tl.store
```
Correction: the 32 `st.global` stores per thread happen after `convert_layout` to `#blocked1`.
The count is the same (32), but they are different elements from the ones the thread computed.

---

## Q4. Why is the accumulator layout `[4,4]/[2,16]/[4,1]`? Who decides it? Could it be different?

**Who:** the Triton compiler, automatically, during the TTIR → TTGIR conversion (the `tt.dot`
conversion pattern in Triton's source, `TritonToTritonGPU`). You don't write layouts in standard
Triton.

**How (rule):**
1. Outputs per thread = 4096/128 = 32. If ≥ 16, use `sizePerThread=[4,4]`; if ≥ 4, use `[2,2]`;
   otherwise `[1,1]`.
2. Spread the 32 lanes along columns first: 64/4 = 16 lanes. The remaining 2 go on rows, giving
   `threadsPerWarp=[2,16]`.
3. Columns are full (16×4 = 64), so the warps go on rows: `warpsPerCTA=[4,1]`.

**Why 4×4:** register reuse. Each k step, a thread reads 4 A values + 4 B values and does **16 FMAs**.
With `[1,1]` it would need 2 reads per FMA. The cost is more registers.

**Can it differ?** Yes. It's a heuristic. It changes with:

| Change | Effect |
|---|---|
| `num_warps` | Different `warpsPerCTA`, fewer outputs per thread |
| Block sizes | Different per-thread count and footprint |
| `allow_tf32=True` | Tensor Core `#mma` layout (rules set by the hardware) |

For direct control, use **Gluon** (`triton.experimental.gluon`, in Triton 3.4).

**Why the load layouts use `[1,1]`:** the Coalesce pass tries vector loads (4 floats = 16 B). The
row strides 2003 and 4017 are odd, so it can't prove 16-byte alignment and falls back to 1 float
per thread (4-byte `cp.async`). Multiples of 16 would likely give `[1,4]` and 16-byte copies.

---

## Q5. How many thread blocks does an SM hold on this Blackwell GPU?

It's not fixed. The **hardware max is 32**, but the actual number is set by resources.

| Per-SM resource | Limit |
|---|---|
| Thread blocks | 32 (NVIDIA CUDA docs, CC 12.0) |
| Threads | 1,536 (48 warps) |
| Registers | 65,536 |
| Shared memory | 100 KB |

For this kernel:

| Limit | Per block | Blocks that fit |
|---|---|---|
| Threads | 128 | 12 |
| Shared memory | 32 KB | **3** |
| Registers | unknown | probably ≥ 3 |

So about **3 blocks per SM**. 188 × 3 = **564 blocks at once**, and 7,938 / 564 ≈ **14 waves**.
Confirm with `print(k.n_regs, k.n_spills)`.

---

## Q6. How many threads/warps form a thread block?

| | Value |
|---|---|
| Threads per warp | **Always 32** (hardware) |
| Warps per block | You choose: 1–32 |
| Threads per block | warps × 32, max **1,024** (CUDA limit) |

In Triton: `num_warps`, **default 4**, e.g. `kernel[grid](..., num_warps=8)`.
This kernel has 4 warps = 128 threads (PTX `.reqntid 128`; TTGIR `"ttg.num-warps" = 4`).

---

## Q7. "Can we choose blocks per SM? Each SM has its own shared memory, and blocks per SM sets how many threads use that SM's shared memory. Right or wrong?"

**Partly right, partly wrong.**

- **Wrong: you don't choose blocks per SM.** You choose the grid size and the per-block resources
  (threads, shared memory, registers). The **hardware scheduler** puts as many blocks on each SM as
  fit.
- **Wrong: blocks don't share shared memory.** Each block gets its own **private slice**:
  ```
  SM shared memory (100 KB)
  ├── Block 0: 32 KB  ← only Block 0's 128 threads see this
  ├── Block 1: 32 KB
  ├── Block 2: 32 KB
  └── ~4 KB unused
  ```
- **Right:** the SM's shared memory is a fixed pool divided among resident blocks, so each block's
  request indirectly limits how many blocks fit.

| Lever | Blocks per SM |
|---|---|
| Smaller `BLOCK_SIZE_*` | More |
| Fewer `num_stages` (pipeline buffers) | More |
| Larger `num_warps` | Fewer |

More blocks per SM is not automatically faster, because bigger tiles reuse more data. Autotuning
finds the balance.

---

## Q8. Diagram: warps, thread blocks, SMs and pids when a Triton kernel launches

```
═══════════════════════════════════════════════════════════════════════════
 ① WHAT YOU LAUNCH (software view)
═══════════════════════════════════════════════════════════════════════════
 grid = (cdiv(8001,64), cdiv(4017,64)) = (126, 63)  →  7,938 programs

            pid_x=0   pid_x=1   pid_x=2        pid_x=62
          ┌─────────┬─────────┬─────────┬ ··· ┬─────────┐
 pid_y=0  │ (0,0)   │ (0,1)   │ (0,2)   │     │ (0,62)  │
          ├─────────┼─────────┼─────────┼ ··· ┼─────────┤
 pid_y=1  │ (1,0)   │ (1,1)   │ (1,2)   │     │ (1,62)  │
          ├─────────┼─────────┼─────────┼ ··· ┼─────────┤
 pid_y=125│(125,0)  │(125,1)  │         │     │(125,62) │
          └─────────┴─────────┴─────────┴ ··· ┴─────────┘
   1 pid (Triton "program")  ==  1 thread block (CUDA "CTA")

═══════════════════════════════════════════════════════════════════════════
 ② INSIDE ONE PID / THREAD BLOCK
═══════════════════════════════════════════════════════════════════════════
 Thread block for pid (5, 7)  →  computes C[320:384, 448:512]
 ┌───────────────────────────────────────────────────────────────┐
 │  num_warps = 4   →   4 warps × 32 threads = 128 threads       │
 │  Warp 0: [t0  ... t31 ]   ┐                                   │
 │  Warp 1: [t32 ... t63 ]   │ each thread holds 32 of the       │
 │  Warp 2: [t64 ... t95 ]   │ 64×64 outputs (#blocked2)         │
 │  Warp 3: [t96 ... t127]   ┘                                   │
 │  Private shared memory: 32 KB (2 A buffers + 2 B buffers)     │
 └───────────────────────────────────────────────────────────────┘
 A warp = 32 threads executing the SAME instruction together.
 It is the unit the hardware schedules.

═══════════════════════════════════════════════════════════════════════════
 ③ WHERE BLOCKS RUN (hardware view): 188 SMs
═══════════════════════════════════════════════════════════════════════════
 ┌──────── SM 0 ─────────┐ ┌──────── SM 1 ─────────┐       ┌─────── SM 187 ────────┐
 │ Shared mem (100 KB)   │ │ Shared mem (100 KB)   │       │                       │
 │ ┌─────┬─────┬─────┬─┐ │ │ ┌─────┬─────┬─────┬─┐ │  ...  │   3 blocks each       │
 │ │Blk A│Blk B│Blk C│ │ │ │ │Blk D│Blk E│Blk F│ │ │       │                       │
 │ └─────┴─────┴─────┴─┘ │ │ └─────┴─────┴─────┴─┘ │       │                       │
 │ 3 blocks × 4 warps    │ │                       │       │                       │
 │ = 12 warps resident   │ │                       │       │                       │
 │ 4 warp schedulers:    │ │                       │       │                       │
 │ ┌────┬────┬────┬────┐ │ │                       │       │                       │
 │ │ S0 │ S1 │ S2 │ S3 │ │ │                       │       │                       │
 │ │3 wp│3 wp│3 wp│3 wp│ │ │                       │       │                       │
 │ └────┴────┴────┴────┘ │ │                       │       │                       │
 └───────────────────────┘ └───────────────────────┘       └───────────────────────┘
 • A block always stays on ONE SM.
 • When a warp waits on memory, the scheduler switches to another ready
   warp (latency hiding).

═══════════════════════════════════════════════════════════════════════════
 ④ OVER TIME: "waves"
═══════════════════════════════════════════════════════════════════════════
 Running at once = 188 × 3 = 564 blocks;  7,938 / 564 ≈ 14.1 waves
 time ──────────────────────────────────────────────────────────►
 │ wave 1: pids 0–563 │ wave 2: 564–1127 │ ... │ wave 14 │ tail │
                                                       (66 blocks, most SMs idle)
 There are no hard wave boundaries: a finished block's slot is
 immediately given to the next waiting pid.
```

**Chain:** 1 launch → 7,938 pids (= thread blocks) → 4 warps each → 32 threads each; blocks are
placed on 188 SMs, ~3 at a time per SM (limited by 32 KB shared memory each).

| Level | Who decides | This kernel |
|---|---|---|
| Number of pids (blocks) | You (`grid`) | 7,938 |
| Warps per block | You (`num_warps`) | 4 |
| Threads per warp | Hardware | 32 |
| Which SM a block runs on | Hardware scheduler | Automatic |
| Blocks running per SM | Resources (smem, regs, threads) | ~3 |

---

## Q9. Check my understanding of launch + pipelining. And `k.n_regs = 196`: what does it mean?

**User's understanding:** launch = grid (number of pids) + warps per pid + pipeline depth. With
double buffering, 2 A/B tile pairs sit in shared memory while the 3rd pair is computed in
registers. Each block needs 32 KB, so 100 KB / 32 KB ≈ 3 blocks per SM. Registers, threads etc.
may take up shared memory and lower that number.

**Verdict: mostly right, with 3 fixes.**

1. **Right:** you choose grid, `num_warps` and `num_stages`. This kernel used the default
   `num_stages=3`, which means **3 − 1 = 2 shared-memory buffers** for A and for B. Each block
   needs 2 × (8 KB A + 8 KB B) = 32 KB.
2. **Fix (pipeline picture):** the "3rd pair" is not a separate copy. In iteration i, threads read
   tile i out of its smem slot into registers (each thread reads only the 384 floats it needs).
   Then the copy of tile i+2 is started **into that same slot**. At steady state: tile i is in
   registers being computed, tile i+1 is arriving in slot B, tile i+2 is arriving in slot A. The
   copies are *in flight* (async), not necessarily "already there".
3. **Fix (resources):** registers do **not** take up shared memory. They are separate resources,
   each with its own per-SM limit. A block fits only if **all** limits allow it, and the tightest
   one wins. (CUDA also reserves ~1 KB of smem per block, so 3 × 33 KB = 99 KB ≤ 100 KB: smem
   allows 3.)

**`n_regs = 196`** = **196 32-bit registers per thread** (the hardware max is 255).
- Why so many: 32 FP32 accumulators + A/B values loaded from smem (preloaded ahead to hide latency)
  + 64-bit pointers/masks for 32 `cp.async` per tile + loop counters.
- Registers are allocated in steps of 8 per thread, so 196 rounds up to 200.
  200 × 128 threads = **25,600 registers per block**.
- 65,536 / 25,600 = 2.56 → **only 2 blocks per SM. Registers, not shared memory, are the real
  limiter.**

| Limit | Blocks per SM |
|---|---|
| Threads (1536/128) | 12 |
| Shared memory (100 KB / 33 KB) | 3 |
| **Registers (65,536 / 25,600)** | **2** ← wins |

**Corrected numbers:** 2 blocks × 4 warps = 8 of 48 warps resident (**~17% occupancy**).
188 × 2 = **376 blocks at once**; 7,938 / 376 ≈ **21.1 → 22 waves**. This replaces the "~3 per SM,
~14 waves" estimates in Q5 and Q8.

To fit 3 blocks: registers per thread must be ≤ 168 (65,536 / 3 / 128 ≈ 170, rounded down to a
multiple of 8). The levers are smaller tiles, `num_warps=8` (fewer outputs per thread), or Triton's
`maxnreg` launch option (risks spills; check `k.n_spills`).

---

## Q10. Where do 65,536 and 1,536 come from?

They're fixed hardware limits **per SM**, read from the GPU:
```python
p = torch.cuda.get_device_properties(0)
p.max_threads_per_multi_processor   # 1536  (= 48 warps × 32)
p.regs_per_multiprocessor           # 65536 (32-bit each = 256 KB register file)
```
They're also listed in NVIDIA's CUDA Programming Guide (compute capability 12.0 table).

## Q11. Max is 1,024 threads per block, so why 1,536? Is the 256 KB of registers "split" across threads?

- **Two different limits:** max 1,024 threads per **block**; max 1,536 threads per **SM**. An SM runs
  several blocks at once. 128-thread blocks → 12 fit by threads; 512 → 3; 1,024 → only 1 (512
  slots wasted).
- **Registers are split, not shared.** Each thread gets its own private `n_regs` (196 → 200).
  The SM hands out 200 per thread until 65,536 run out: 2 blocks × 25,600 = 51,200 used; the
  remaining 14,336 aren't enough for a 3rd block, so they sit unused.
- **Shared memory** is what's actually shared, among the threads of one block.

## Q12. Is an SM a hardware resource that the GPU allocates when a kernel launches? How do block/warp limits map to SMs?

**Yes.** An SM is a physical processor (like a CPU core, but very wide). The kernel is software
that describes work; the GPU's hardware **block scheduler** places it onto SMs.

Lifecycle:
1. **Compile:** each block's needs are fixed: 128 threads (4 warps), 196 → 200 regs/thread,
   32 KB smem.
2. **Launch:** the grid becomes a queue of 7,938 blocks. Per-**block** limits are checked here
   (≤ 1,024 threads, ≤ ~99 KB smem); breaking them = launch error.
3. **Placement:** the scheduler puts a block on an SM only if that SM has enough **free** warp
   slots, registers and smem. Those are **reserved** for the block's whole life.
4. **Run:** the block stays on that SM until done (it never moves). The SM's warp schedulers run
   its warps.
5. **Finish:** the resources are freed and the next queued block takes the slot.

Two kinds of limits:

| | Per block | Per SM |
|---|---|---|
| What it is | Rule on how big one block may be | Capacity of the hardware |
| Threads | 1,024 (32 warps) | 1,536 (48 warps) |
| Blocks | — | 32 |
| Registers | — | 65,536 |
| Shared memory | ~99 KB | 100 KB |
| When checked | At launch (error if exceeded) | Continuously (decides how many blocks run at once) |

Inside an SM there are **4 sub-partitions**, each with its own warp scheduler and 16,384
registers. A block's 4 warps are spread across them.

---

## Q13. What are the tradeoffs (tile size, num_warps, num_stages)? What does autotune do?

**User's understanding:** larger blocks (64×64 → 128×128) mean fewer pids and fewer waves, so it
finishes faster, but put pressure on shared memory and registers. The tradeoffs are between grid
size (affects smem and registers), num_warps (limits blocks) and num_stages (smem).

**Verdict: right direction, two cause-and-effect links are off.**

- **"Larger blocks → fewer pids → faster": not by itself.** Total work is identical. Fewer blocks
  means more work per block. The real win is **data reuse → less global-memory traffic**:

  | Tile | Multiply-adds per element loaded | A re-read | B re-read |
  |---|---|---|---|
  | 64×64 | 32 | 63× | 126× |
  | 128×128 | **64** | **32×** | **63×** |

  Fewer waves is a side effect, not the goal.
- **"Grid size affects smem/registers": reversed.** **Tile size** sets smem and registers per block.
  Grid size is a consequence of tile size, and it decides **whether there are enough blocks to fill
  188 SMs**.
- **num_warps** decides how many threads share one tile: more warps → fewer outputs and registers
  per thread, but more threads per block.

### Tradeoff table

| Knob | Increase it → gain | Cost |
|---|---|---|
| Tile size (BLOCK_M × BLOCK_N) | More reuse, less memory traffic | More smem and registers per block (risk of **spills**); fewer blocks per SM; fewer blocks in total (may not fill SMs); more wasted work on edge tiles |
| BLOCK_K (`BLOCK_SIZE_N` here) | Fewer loop iterations | More smem per stage, more registers |
| num_warps | Fewer registers per thread | More threads per block; more coordination overhead |
| num_stages | Deeper prefetch, better latency hiding | +1 A+B tile of smem per stage → fewer blocks per SM |

### Non-obvious effects

1. **Low occupancy isn't automatically bad.** This kernel runs 8/48 warps per SM (17%), but each warp
   has 1,024 independent FMAs per step. Latency is hidden either by many warps or by lots of
   independent work per warp (ILP); fast matmuls choose ILP.
2. **Tail effect.** 128×128 → 63 × 32 = 2,016 blocks. At 1 block/SM that's 10.7 waves, and the last
   wave uses only 70% of the GPU. Tiles that are too large on small matrices leave SMs idle.
3. **Tensor Cores (`allow_tf32=True`) are a bigger lever than any of these knobs.**

### Autotune

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C_ROWS': 64,  'BLOCK_SIZE_C_COLS': 64,  'BLOCK_SIZE_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_C_ROWS': 128, 'BLOCK_SIZE_C_COLS': 64,  'BLOCK_SIZE_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_C_ROWS': 128, 'BLOCK_SIZE_C_COLS': 128, 'BLOCK_SIZE_N': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_C_ROWS': 64,  'BLOCK_SIZE_C_COLS': 128, 'BLOCK_SIZE_N': 16}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],   # re-tune when these change
)
@triton.jit
def matrix_multiplication_kernel(...): ...

# launch WITHOUT block sizes; the grid lambda already reads them from `meta`
matrix_multiplication_kernel[grid](a, b, c, M, N, K, ...strides...)
```

How it works:
1. The first call with a new `(M, N, K)` compiles every config.
2. It times each one (several runs, real inputs).
3. It caches the winner for that key; later calls use it directly.

Caveats:
- The first call is slow (compiling and benchmarking every config).
- The winner is specific to the GPU and shape.
- Kernels that *accumulate* into the output need `reset_to_zero=['c']`. This kernel overwrites `c`,
  so it's safe.
- It only searches the configs you list.
- After tuning, inspect the winner's `n_regs` / `n_spills` to understand why it won.

---

## Q14. Can some warps load tiles while others compute?

**Yes, as a technique: it's called warp specialization (producer/consumer warps). But your kernel
doesn't do it.** In your kernel all 4 warps do everything: each issues its share of the copies, then
each does its share of the FMAs.

**Overlap still happens in your kernel, in two ways:**
1. **Async copies.** `cp.async` is fire-and-forget. A warp spends a few instructions issuing the copy;
   the memory system moves the data in the background while the same warp goes back to FMAs. So
   "loading" costs warps very little time; the waiting is done by hardware, not by warps.
2. **Other blocks on the same SM.** With 2 blocks per SM, while block 0's warps wait at
   `cp.async.wait_group`, the warp schedulers run block 1's warps, which may be computing.

```
Your kernel (all warps do both):
  warp 0-3: [issue copy i+2][FMA on tile i ...........][wait][issue copy i+3][FMA on tile i+1 ...]
  memory  :      └──── tile i+2 arriving in background ────┘

Warp specialization:
  producer warp : [copy i+2][copy i+3][copy i+4] ... (few registers)
  consumer warps: [FMA tile i][FMA tile i+1] ...      (get the registers the producer doesn't use)
                 ↑ they hand off through shared memory + barriers ("tile ready" / "slot free")
```

**Why bother, if async copies already overlap?** The payoff shows up with Hopper/Blackwell hardware:
- **TMA** (a hardware copy engine): one thread can copy a whole tile, so one small producer warp is
  enough.
- **Register rebalancing** (`setmaxnreg`): producer warps give up registers so consumer warps can
  hold bigger accumulators.
- Async Tensor Core instructions (`wgmma` on Hopper, `tcgen05` on datacenter Blackwell) let
  consumers keep the math units busy while waiting is handled elsewhere.

**In Triton 3.4 (installed here):** `for i in tl.range(0, n, warp_specialize=True):` asks the
compiler to split the loop into producer/consumer warps. The NVIDIA backend
(`triton/backends/nvidia/compiler.py`) runs this pass for capability ≥ 10.0, which includes your
SM120. But the pass is built around datacenter-Blackwell features (TMA descriptors, `tcgen05`,
tensor memory) that SM120 lacks or that your FP32 FMA kernel doesn't use, so it will probably have
no effect here. Verify: add it, print TTGIR and look for `ttg.warp_specialize`. For full manual
control, Gluon lets you write the partitions yourself.

---

## Q15. So threads only *issue* the copy, and the memory system does the actual copying?

**Yes.** The threads' part of the work is:
1. **Address and mask math.** Each thread computes the address and mask for each of its 32
   elements per A+B tile pair. Most of the `add`/`setp`/`selp` instructions in the PTX are this.
2. **Issuing** one `cp.async` per element (4 bytes each here; 16 bytes if vectorized).

From then on, the **memory system** (load/store unit → L1 → L2 → DRAM) moves the data **straight
into shared memory**, without passing through registers. The thread is free right after issuing.

Threads only wait at two points:
- `cp.async.wait_group N`: "stop until all but my N newest copy groups have landed".
- `bar.sync`: "stop until all threads in the block are here", so everyone's copies are in before
  anyone reads the tile.

The **shared → registers** step is different. `ld.shared` (96 per iteration) is done by the
threads, and their FMAs depend on it, so it costs real time inside the compute phase.

```
thread:  [address math][issue cp.async ×32] ──► back to FMAs ... [wait_group][bar.sync][ld.shared ×96][FMA ×1024]
memory:                 └──────── global → shared, in the background ────────┘
```

Compared with a plain `ld.global` (global → **register**): that is also non-blocking until the
value is used, but the destination register stays reserved while the data is in flight, and the
data must then be written to shared memory by the thread. `cp.async` skips both.

---

## Q16. So cp.async (chosen automatically by Triton) moves data from DRAM to shared memory, bypassing registers?

**Yes, bypassing registers. Two refinements:**

1. **Not "straight from DRAM".** Every global read goes through the L2 cache:
   ```
   DRAM → L2 → (L1) → shared memory        registers never touched
   ```
   - If the data is already in L2, DRAM isn't touched at all. That's common here: A is re-read 63×
     and B 126×, and many of those reads hit L2.
   - Your PTX has `cp.async.ca` = "cache in L1 too". The other form, `cp.async.cg`, skips L1 but
     only works for 16-byte copies. Your copies are 4 bytes (odd strides, see Q4), so `.ca` is the
     only option.
2. **Triton picks it only when it pipelines the loop.** It uses `cp.async` when:
   - the GPU is Ampere (sm_80) or newer (yours is sm_120), and
   - `num_stages > 1` and the load feeds a loop it can pipeline (here: loads feeding `tl.dot`).

   Otherwise (e.g. `num_stages=1`) it emits plain `ld.global` into registers. If the data then needs
   to go to shared memory (as `tl.dot` operands do), threads write it there with `st.shared`.

Check it yourself: run with `num_stages=1`, print the PTX, and `cp.async` should be gone,
replaced by `ld.global`.

---

## Q17. Why "(L1)" in brackets? What's the difference between shared memory and L1?

**Brackets = optional step.** `cp.async.ca` keeps a copy in L1 on the way; `cp.async.cg` skips L1.
L2 is never skipped.

**Shared memory and L1 are the same physical memory on each SM, split into two uses.** Each SM has
one block of fast on-chip SRAM (128 KB on this GPU architecture, per NVIDIA's docs). Part of it is
used as shared memory and the rest as L1 cache. The split is called the **carveout**. Up to 100 KB
can go to shared memory, which is the limit PyTorch reported.

| | Shared memory | L1 cache |
|---|---|---|
| Managed by | **You / the compiler**: explicit addresses (`local_alloc`, `st.shared`, `ld.shared`) | **Hardware**, automatically |
| What's in it | Exactly what you put there | Recently used global-memory data, chosen by the hardware |
| How long data stays | Until you overwrite it or the block ends | Until the hardware evicts it (you can't control this) |
| Who can see it | All threads of **one block** | Used by everything running on the SM |
| Guaranteed fast? | Yes, once the data is there | Only if the data happens to still be there (a cache hit) |
| Speed | About the same (same SRAM) | About the same |

**Why matmul uses shared memory, not just L1:** the tile must definitely be there when 128 threads
read it, many times, in a layout the compiler chose. A cache can't guarantee that. Shared memory is a
scratchpad you control.

**Tradeoff:** shared memory and L1 come out of the same pool. More shared memory per SM (3 blocks ×
33 KB) leaves less L1. The driver sets the split based on how much shared memory the kernel asks
for.

---

## Q18. CUDA C++ vs Triton: what do you control in each?

**Core difference:** in CUDA you program **one thread**; in Triton you program **one block (a
tile)**, and the compiler decides what each thread does.

**Blocks per SM: neither lets you set it directly.** The hardware scheduler decides it in both
(Q12). Both let you *influence* it through resources:
- CUDA: `__launch_bounds__(maxThreads, minBlocksPerSM)` makes the compiler cap registers so that
  many blocks fit.
- Triton: `maxnreg=` launch option (present in the installed 3.4 compiler) caps registers per
  thread. Plus `num_warps` and `num_stages`.

| What | CUDA C++ | Triton |
|---|---|---|
| Grid size | You | You |
| Threads per block | You, any count up to 1,024 | Only `num_warps` (× 32) |
| Work per block (tile size) | You | You (`BLOCK_*` constexprs) |
| **Which thread handles which element (layout)** | **You**, by hand via `threadIdx` | **Compiler** (`#blocked`, `#mma`, …). You can't set it. |
| **Shared memory: allocate, size, layout, swizzle** | **You** (`__shared__`, padding, swizzle by hand) | **Compiler** (`local_alloc`, `#shared`). You influence size via tiles and stages. |
| Global → shared copies (`cp.async`, TMA) | You write them | Compiler, when it pipelines (Q16) |
| Pipelining (number of buffers, when to wait) | You write it | `num_stages` (kernel- or loop-level via `tl.range(..., num_stages=)`) |
| Barriers (`__syncthreads`, `bar.sync`) | You (and you can get them wrong) | Compiler inserts them |
| Registers per thread | Compiler, capped by `__launch_bounds__` / `-maxrregcount` | Compiler, capped by `maxnreg` |
| Blocks per SM | Indirect (resources, `__launch_bounds__`) | Indirect (resources, `maxnreg`) |
| Tensor Core use | You call `wmma` / `mma` / inline PTX | Compiler, from `tl.dot` + dtype + `input_precision` |
| Vectorized (16-byte) loads | You (`float4`, alignment) | Compiler, if it can prove alignment; hints `tl.multiple_of`, `tl.max_contiguous` |
| Cache behaviour | PTX qualifiers, `__ldg`, etc. | `cache_modifier=`, `eviction_policy=` on `tl.load`/`tl.store` |
| L1 / shared memory split (carveout) | `cudaFuncSetAttribute(...PreferredSharedMemoryCarveout)` | No knob |
| Warp-level tricks (`__shfl_sync`, votes) | You | Compiler uses them inside `tl.sum`, `tl.max`, etc. You can't call them. |
| Warp specialization | You write it | `tl.range(..., warp_specialize=True)` (limited, Q14) |
| Thread-block clusters (sm_90+) | You | `num_ctas` |
| Masks / bounds checks | You write `if (i < M)` | You write `mask=` |
| Inline assembly | Full inline PTX | `tl.inline_asm_elementwise` (element-wise only) |

**Middle ground: Gluon** (`triton.experimental.gluon`, in Triton 3.4). Python like Triton, but you
set layouts, allocate shared memory, issue async copies and barriers, and write warp-specialized
partitions yourself.

**Rule of thumb:** Triton handles the hard-to-get-right parts (layouts, swizzles, barriers,
pipelining) and usually gets close to hand-written CUDA. CUDA (or Gluon) is for squeezing out the
last percent, or for patterns Triton can't express.

---

## Q19. What are barriers, async copies and swizzles?

### Async copy: "start the copy now, check on it later"
A normal load makes the thread wait (at the first use) for the data. An async copy (`cp.async`)
only *starts* the transfer from global memory into shared memory, and the thread keeps working.
Later, the thread checks that it has arrived with `cp.async.wait_group N`.

Analogy: ordering a delivery instead of driving to the shop. You keep working until you need it.

In your kernel: the copy of tile i+2 is started, tile i is computed, then the kernel waits for
tile i+1 (Q15, Q16).

### Barrier: "nobody continues until everyone in the block gets here"
`bar.sync 0` (CUDA: `__syncthreads()`) stops each thread at that line until **all 128 threads**
of the block arrive.

Why it's needed: threads share a tile in shared memory, but each thread copied only **its own part**
of it. Without a barrier, a fast thread could start reading the tile while a slow thread's part is
still missing. Your kernel needs two kinds of safety:
1. **Before reading** a tile: everyone's copies have landed (`wait_group` + `bar.sync`).
2. **Before overwriting** a slot with a new tile: everyone has finished reading the old one.

Get a barrier wrong in CUDA and you get wrong results only *sometimes* (a race condition). Triton
inserts them for you. Hopper/Blackwell add finer-grained **mbarriers** (wait for a specific copy to
arrive, not for all threads), used with TMA and warp specialization.

### Swizzle: rearranging where data sits in shared memory to avoid bank conflicts
Shared memory is split into **32 banks**, each 4 bytes wide. Address → bank = (address / 4) % 32.
A warp's 32 threads can read in **one step** only if they hit **32 different banks** (or the same
4-byte word, which is broadcast). Two different addresses in the same bank = a **bank conflict**:
the reads happen one after another.

Example: a 64×32 FP32 tile stored row by row. Each row is 32 floats = 128 bytes, so element
`[r][c]` is always in **bank c**. If a warp reads one **column** (32 rows, same c), all 32 reads
hit the same bank → **32 steps instead of 1**.

```
plain:     row r, col c → bank c                 column read → all in bank c (32-way conflict)
swizzled:  row r, col c → stored at col (c XOR r) → column read → 32 different banks (no conflict)
```

**Swizzling** = store `[r][c]` at a scrambled column (usually `c XOR (some bits of r)`). Each row
gets a different shuffle, so a column's elements land in different banks. Readers apply the same
formula, so nothing is lost. (The older fix is **padding**: make each row 33 floats wide. That
wastes a bit of memory.)

In your TTGIR: `#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, ...}>`.
`maxPhase = 1` means **no swizzling**. Triton chose a plain layout for this FP32 FMA path. With
Tensor Cores (`#mma`), you'd typically see larger `vec`/`maxPhase` values.

---

## Q20. "Wait for tile i+1": isn't it already in shared memory? And swizzling, explained slowly

### Part 1: the loop, step by step

**User's model:** threads start the copy of tile i+2, don't wait, compute tile i, and since tile i+1
is already in shared memory, they can read it straight away.

**Mostly right, with two corrections.**

**Correction 1: tile i+1 is *probably* there, but not guaranteed.** Its copy was started one
iteration earlier, so it has had a whole iteration of compute time to arrive. Usually it has. But
memory can be slow, so the threads must **check** before reading it. That check is
`cp.async.wait_group`. If the tile has landed, the check passes immediately. If not, the threads
stall until it does. "Wait for tile i+1" means "confirm it arrived", not "start fetching it".

**Correction 2: the order inside one iteration.** In the generated code (TTGIR and PTX), the copy of
tile i+2 is started *after* computing tile i, not before. Tile i+2 goes into **the slot tile i
used**, and that slot can only be overwritten once every thread has finished reading tile i.

```
2 slots: S0, S1

before loop:  start copy tile 0 → S0,  start copy tile 1 → S1

iteration i (tile i is in slot S[i%2]):
  1. wait_group  → my copies for tile i have landed   (tile i+1 may still be arriving)
  2. bar.sync    → everyone's copies for tile i have landed
  3. ld.shared   → read my part of tile i: S[i%2] → registers
  4. FMA × 1024  → compute
  5. start copy tile i+2 → S[i%2]   (the slot tile i just vacated)

time ──────────────────────────────────────────────────────────────────►
iter 0: [wait t0][read S0][compute t0][copy t2→S0 ...........]
iter 1:                                [wait t1][read S1][compute t1][copy t3→S1 ......]
iter 2:                                                   [wait t2][read S0][compute t2]...
memory: t1 arriving in S1 ─────────────┘        t2 arriving in S0 ─┘
```

**Barrier nuance:** `bar.sync` does **not** wait for async copies by itself; it only waits for
threads to *arrive*. That's why it's paired with `wait_group`:
1. Each thread waits for **its own** copies (`wait_group`).
2. Then it waits for **every other thread** to have done the same (`bar.sync`).

Together: "the whole tile has landed".

### Part 2: swizzling, slowly

**Shared memory is like 32 checkout counters (banks).** Each counter serves one 4-byte value per
step. When a warp's 32 threads all read at once:
- 32 threads at 32 **different** counters → served in **1 step**.
- several threads at the **same** counter (for different values) → they **queue**. That's a bank
  conflict.

The bank is decided by the position in memory: consecutive 4-byte values go to bank 0, 1, 2, … 31,
then wrap back to bank 0.

**Small example: 4 banks, a 4×4 tile.** Rows are `r`, columns are `c`. Element `[r,c]` = row r,
column c. Each row is 4 values, so it fills exactly banks 0–3:

```
Plain layout (bank = c)
            bank0  bank1  bank2  bank3
row 0       [0,0]  [0,1]  [0,2]  [0,3]
row 1       [1,0]  [1,1]  [1,2]  [1,3]
row 2       [2,0]  [2,1]  [2,2]  [2,3]
row 3       [3,0]  [3,1]  [3,2]  [3,3]
```
- 4 threads read **row 0** → banks 0, 1, 2, 3 → all different → **1 step**. ✓
- 4 threads read **column 0** ([0,0], [1,0], [2,0], [3,0]) → **all bank 0** → **4 steps**. ✗

**The fix: shift each row by a different amount, so a column is spread across banks.** The simplest
version is a rotation: row r is shifted right by r.
```
Rotated layout (bank = (c + r) % 4)
            bank0  bank1  bank2  bank3
row 0       [0,0]  [0,1]  [0,2]  [0,3]
row 1       [1,3]  [1,0]  [1,1]  [1,2]
row 2       [2,2]  [2,3]  [2,0]  [2,1]
row 3       [3,1]  [3,2]  [3,3]  [3,0]
```
Now column 0 is [0,0]→bank0, [1,0]→bank1, [2,0]→bank2, [3,0]→bank3 → **1 step**. ✓ Rows are
still 1 step too, because each row is just reordered.

**XOR is the same idea with a different shuffle.** XOR compares two numbers bit by bit: 1 where the
bits differ, 0 where they match. E.g. `1 XOR 2` = `01 XOR 10` = `11` = 3.

Swizzled rule: element `[r,c]` goes to bank `c XOR r`.
```
c XOR r table             Swizzled layout (bank = c XOR r)
      c=0 c=1 c=2 c=3                  bank0  bank1  bank2  bank3
r=0:   0   1   2   3       row 0       [0,0]  [0,1]  [0,2]  [0,3]
r=1:   1   0   3   2       row 1       [1,1]  [1,0]  [1,3]  [1,2]
r=2:   2   3   0   1       row 2       [2,2]  [2,3]  [2,0]  [2,1]
r=3:   3   2   1   0       row 3       [3,3]  [3,2]  [3,1]  [3,0]
```
Column 0: [0,0]→bank 0, [1,0]→bank 1, [2,0]→bank 2, [3,0]→bank 3 → **1 step**. ✓

Every row and every column of the XOR table contains each bank exactly once, so both row reads and
column reads are conflict-free.

**Why XOR instead of rotation?** Same goal. XOR is a single, very cheap bit operation, it never
"wraps around" past the last bank (for power-of-2 sizes), and it's easy to apply to groups of
elements (e.g. 4 floats at a time). That's what `vec`, `perPhase` and `maxPhase` in `#shared`
control.

**Nothing is lost.** Writers and readers both use the same formula, so a thread that wants `[2,0]`
computes `0 XOR 2 = 2` and reads bank 2.

**Real size:** 32 banks, and a 32-float row exactly fills banks 0–31. That's the 64×32 example from
Q19: a plain layout puts a whole column in one bank (32 steps), and swizzling spreads it over 32
banks (1 step). Your kernel uses `maxPhase = 1` = no swizzle.

---

## Q21. Who controls swizzling: Triton vs CUDA C++?

**Triton: the compiler, automatically.** When it creates a shared-memory buffer, it looks at how the
data will be **read** (which layout `tl.dot` needs, the data type, Tensor Cores or not) and picks
a swizzle. The choice shows up in TTGIR as the `#shared` encoding:
```
#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>   ← yours: no swizzle
```
- `vec`: how many elements move together as one group (e.g. 4 floats = 16 bytes)
- `perPhase`: how many rows share the same shuffle
- `maxPhase`: how many different shuffles there are (1 = no swizzle)

With Tensor Cores you'd see larger values, or a `#ttg.nvmma_shared` encoding. You **cannot** set it
in standard Triton. **Gluon** lets you choose the shared layout yourself.

**CUDA C++: the developer, explicitly.** `__shared__ float tile[64][32];` is a plain array, so the
hardware stores it in plain row order. To avoid conflicts you choose one of:
1. **Write the XOR yourself** in every shared-memory read and write:
   `tile[r][c ^ (r % 32)]`. Easy to get wrong: every reader and writer must use the same formula.
2. **Padding:** `__shared__ float tile[64][33];`. Simple, but wastes memory.
3. **A library:** CUTLASS / CuTe have swizzle layouts (`Swizzle<B, M, S>`), so you declare the
   layout once and the library does the index math.
4. **Let the copy hardware do it:** with TMA (Hopper and newer), you pick a swizzle mode
   (32 / 64 / 128-byte) when you describe the tile, and the hardware writes the data already
   swizzled. Tensor Core instructions that read shared memory understand those modes.

Even with options 3 and 4, *you* decide which swizzle to use. In Triton, the compiler decides.

---

## Q22. Are CUTLASS / CuTe CUDA C++ libraries? What do the names stand for?

**Yes, both are NVIDIA C++ libraries for writing CUDA kernels.** They are open source
(github.com/NVIDIA/cutlass) and **not** part of the CUDA Toolkit install; you download and
`#include` them. They are header-only templates.

- **CUTLASS** = **CU**DA **T**emplates for **L**inear **A**lgebra **S**ubroutines. Building blocks
  (tiles, copies, pipelines, Tensor Core calls, epilogues) that you assemble into your own fast
  matmul or convolution kernels.
- **CuTe** = the layout layer inside CUTLASS (since CUTLASS 3.0). It describes "which thread / which
  memory slot holds which element" as composable layouts, including swizzles. NVIDIA doesn't
  officially expand the name; it's usually read as "CUDA Tensors". Conceptually it's close to
  Triton's `#blocked` / `#shared` layouts, but you write them yourself.
- Recent CUTLASS versions also include a **Python** version of CuTe ("CuTe DSL").

**How they compare with other NVIDIA libraries:**

| Library | What you get | Write your own kernel? |
|---|---|---|
| cuBLAS | Ready-made matmul and other routines; closed source; just call a function | No |
| CUTLASS / CuTe | Templates to *build* matmul-like kernels | Yes, from building blocks |
| CUB / Thrust | Ready-made parallel building blocks (sort, scan, reduce); ship with the toolkit | Partly |
| Triton | Python language; the compiler generates the kernel | Yes, at tile level |

`torch.matmul` usually calls cuBLAS. Kernels like FlashAttention 2/3 are built on CUTLASS / CuTe.

---

## Q23. What's the compile flow for CUDA C++? Does a `.cu` file compile to the cubin?

**Yes, broadly.** You write a `.cu` file; `nvcc` compiles it, and the GPU code ends up as a cubin.
Two additions: a `.cu` file holds **both CPU and GPU code**, and the cubin gets **packed inside your
program** rather than sitting alone.

```
                 kernel.cu  (CPU code + __global__ GPU kernels)
                      │ nvcc splits it
          ┌───────────┴────────────┐
     CPU (host) code          GPU (device) code
          │                        ↓  NVIDIA's front end (cicc)
     gcc / clang              NVVM IR  (NVIDIA's flavour of LLVM IR)
          │                        ↓
          │                       PTX      (virtual GPU assembly)
          │                        ↓  ptxas
          │                      cubin     (SASS = real machine code for one GPU, e.g. sm_120)
          │                        ↓
          │                    fatbinary   (cubins for several GPUs + PTX, bundled)
          └──────────┬─────────────┘
               program / .so  (fatbinary embedded inside)
```

**Side by side:**

| | Triton | CUDA C++ |
|---|---|---|
| You write | Python, per **tile** | C++, per **thread** |
| Top of the pipeline | TTIR → **TTGIR** (compiler chooses layouts, shared memory, pipelining) | None. Those decisions are already in your source. |
| Bottom of the pipeline | LLVM IR → PTX → `ptxas` → cubin | NVVM IR → PTX → `ptxas` → cubin |
| When it compiles | **At runtime**, on the first call (JIT), for exactly your GPU and argument values | **Before running** (`nvcc`, ahead of time) |
| Uses actual values | Yes: constexprs, stride == 1, 16-byte divisibility (why you saw only 9 kernel args) | Only what's known at compile time (templates, constants) |
| Targets | The GPU you're running on | Any list you choose (`-gencode` for sm_80, sm_90, sm_120 …) |

**The bottom half is shared.** Both produce PTX, and both use NVIDIA's `ptxas` to turn it into the
cubin.

**At runtime (CUDA C++):** the driver picks the cubin that matches your GPU. If there isn't one,
it **JIT-compiles the embedded PTX**. That's why shipping PTX keeps a program working on future
GPUs.

**Looking at the stages:**
- Triton: `k.asm` has keys `'ttir'`, `'ttgir'`, `'llir'`, `'ptx'`, `'cubin'`.
- CUDA: `nvcc -ptx kernel.cu` (PTX only); `nvcc --keep` (keep all intermediate files).
- Real machine code (SASS) for either: `cuobjdump -sass <binary>` or `nvdisasm <file.cubin>`.

---

## Q24. What are sm_80, sm_90, …?

**`sm_XY` is the name of a GPU generation's architecture.** "SM" is the same Streaming
Multiprocessor from Q5–Q12. `XY` is its **compute capability** X.Y (major.minor version).
Each generation has a different instruction set, so it needs its own machine code.

| Name | Compute capability | Generation | Example GPUs |
|---|---|---|---|
| sm_70 | 7.0 | Volta | V100 |
| sm_75 | 7.5 | Turing | T4, RTX 20-series |
| sm_80 | 8.0 | Ampere | A100 |
| sm_86 | 8.6 | Ampere | RTX 30-series, RTX A6000 |
| sm_89 | 8.9 | Ada Lovelace | L4, L40, RTX 40-series |
| sm_90 | 9.0 | Hopper | H100, H200 |
| sm_100 | 10.0 | Blackwell (datacenter) | B200, GB200 |
| **sm_120** | **12.0** | **Blackwell (workstation / consumer)** | **RTX PRO 6000 (yours)**, RTX 50-series |

Your PTX said `.target sm_120a`. The `a` means "use features specific to this exact architecture".
Code built that way runs only on sm_120. Also, sm_120 is **not** a newer sm_100: it lacks some
datacenter features (e.g. `tcgen05` / tensor memory, Q14).

**Why it matters for compiling:**
- **cubin (machine code)** runs only on its own generation. An sm_80 cubin also runs on sm_86/sm_89
  (same major version 8), but **never** on sm_90 or sm_120.
- **PTX** is forward-compatible: the driver can compile sm_80 PTX for a newer GPU at runtime.

**What the table row meant:**
- **CUDA C++** compiles *before* you know which GPU will run it. So you can build one program
  containing several cubins, e.g. `nvcc -gencode arch=compute_80,code=sm_80
  -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 kernel.cu`. The same
  file then runs on an A100, an H100 and your RTX PRO 6000.
- **Triton** compiles *at runtime*, when the GPU is already known. It just builds for that one
  (`sm_120a` here). On an H100 it would build `sm_90a` instead.

Check any GPU's value: `torch.cuda.get_device_capability()` → `(12, 0)` here.

---

## Q25. If FLOPs are fixed and only memory traffic depends on tile size, aren't bigger tiles always better? Why do more blocks per SM help?

**User's reasoning:** FLOPs are the same for any tile size, so compute time is the same. Only memory
traffic changes, and bigger tiles cut it. So bigger tiles should be better as long as they fit, and
idle SMs in the tail shouldn't matter, because DRAM, not compute, is the bottleneck.

**Two hidden assumptions are wrong.**

### 1. "Same FLOPs → same compute time": only if the math units stay busy
```
compute time = FLOPs ÷ (FLOPs per second actually achieved)
```
The achieved rate depends on:
- **how many SMs have work** (an idle SM contributes 0 FLOPs/s), and
- **what fraction of the time each SM's math units are busy** (stalled warps contribute 0).

Idle time adds to the total even though the FLOP count doesn't change.

### 2. "DRAM is the bottleneck": for matmul it's usually the opposite (numbers for this kernel)

| | Value |
|---|---|
| Work | 2 × 8001 × 2003 × 4017 = **128.8 GFLOP** |
| FP32 peak (188 SMs × 128 FP32 lanes × 2 × 2.43 GHz) | **~117 TFLOPS** → **≥ 1.10 ms** of pure math |
| Unavoidable DRAM traffic (read A + B once, write C once) | **225 MB** → at ~1.8 TB/s (NVIDIA spec) **≈ 0.13 ms** |

The math takes ~9× longer than the unavoidable memory traffic. **Matmul is compute-bound** if the
tiles give enough reuse.

**Where your point is right:** tile *re-reads*. Every block loads its own A and B tiles:

| Tile | Tile loads (whole kernel) | If all from DRAM | FLOPs per byte loaded |
|---|---|---|---|
| 64×64 | 8.2 GB | 4.6 ms | 16 |
| 128×128 | 4.2 GB | 2.3 ms | 32 |
| 256×256 | 2.2 GB | 1.2 ms | 64 |

The GPU can do ~65 FLOPs per byte of DRAM bandwidth (117 TFLOPS ÷ 1.8 TB/s). With 64×64 tiles, if
every re-read hit DRAM, the kernel would be memory-bound (16 < 65). In practice most re-reads hit
the **L2 cache**, which is much faster. **So bigger tiles help, up to the point where memory stops
being the bottleneck. After that, more reduction buys nothing**, and the costs of big tiles take
over.

### 3. Why one block per SM can hurt
1. **Nothing to switch to.** Warps of one block stop *together* at `wait_group` + `bar.sync`. With
   1 block per SM, the SM's math units sit idle during those waits. With 2+ blocks, the blocks are
   at different points in their loops, so one computes while the other waits:
   ```
   1 block/SM:  A: [compute][wait][compute][wait][compute]      ← math idle during every wait
   2 blocks/SM: A: [compute][wait][compute][wait]
                B: [wait][compute][wait][compute]               ← waits overlap with B's math
   ```
2. **Start and end of each block.** Filling the pipeline (prologue) and writing C (epilogue) leave
   the math units idle. Another block on the same SM covers that.
3. **Register pressure.** Big tiles → more outputs per thread → more registers. Past 255, values
   **spill** to slow memory, which adds memory traffic back.
4. **Tail effect.** Fewer, bigger blocks → coarser waves, and idle SMs in the last wave. In a
   compute-bound kernel, idle SMs directly cost time.

Tail example, same 8001 × 4017 output at 1 block/SM:

| Tile | Blocks | Waves on 188 SMs | Last-wave use | Efficiency |
|---|---|---|---|---|
| 128×128 | 2,016 | 10.7 → 11 | 72% | ~97% |
| 256×256 | 512 | 2.7 → 3 | 72% | ~91% |

Smaller matrix, 2048 × 2048 with 256×256 tiles: only **64 blocks**, so **124 of 188 SMs do
nothing**. The kernel runs at ~34% of the GPU, about 3× slower than it could.

**One block per SM isn't always bad.** The fastest Tensor Core kernels (CUTLASS, Hopper/Blackwell)
often run 1 big block per SM, but they hide latency *inside* the block with deep pipelines and warp
specialization (Q14). A simple kernel like yours relies on a second block for that.

### The real rule
Choose the **smallest tile that makes the kernel compute-bound**, while keeping
- enough blocks to fill all SMs with a small tail,
- registers below spilling, and
- some way to hide stalls (≥ 2 blocks per SM, or deep pipelining).

**With Tensor Cores** the math gets several times faster, so FLOPs per byte needed rises by the same
factor. That's why Tensor Core matmuls use 128×128 or bigger tiles, while your FP32 FMA kernel does
reasonably with 64×64.

To test this on the real GPU, time configs with `triton.testing.do_bench` and compare achieved
TFLOPS = 128.8 GFLOP ÷ time.

---

## Q26. With 2 blocks per SM, do the same warps work on both blocks, or does each block have its own warps?

**Each block has its own, separate warps.** A warp belongs to exactly one block for its whole life
and never works on another block. With 2 blocks × 4 warps, the SM holds **8 separate warps**, each
with its own registers (that's why registers are reserved per block, Q11).

**What the blocks share is the SM's hardware:** the 4 warp schedulers and the math units (FP32
lanes). Every cycle, each scheduler picks **one ready warp** from its pool (from either block) and
issues that warp's next instruction.

```
SM
├── Scheduler 0: pool = { A.warp0, B.warp0 }  ┐
├── Scheduler 1: pool = { A.warp1, B.warp1 }  │ each cycle, each scheduler issues
├── Scheduler 2: pool = { A.warp2, B.warp2 }  │ one instruction from ONE ready warp
└── Scheduler 3: pool = { A.warp3, B.warp3 }  ┘
    shared: FP32 math units, shared memory, L1
    private per warp: its registers (A's warps 25,600 regs total, B's warps 25,600)

Scheduler 0 over time:
  A.warp0: [FMA FMA FMA][waiting at bar.sync ...........][FMA FMA ...
  B.warp0: [stalled ...][FMA FMA FMA FMA FMA FMA FMA FMA][waiting ...
  issued : A  A  A       B  B  B  B  B  B  B  B          A  A ...     ← math units never idle
```

**Switching costs nothing.** On a CPU, switching threads means saving and restoring registers. On
a GPU, every resident warp's registers *stay* in the register file, so the scheduler can pick a
different warp on the very next cycle. That's why the register file is so large (256 KB per SM),
and why register usage limits how many warps can be resident.

So "one computes while the other waits" = the scheduler issues B's instructions while A's warps
are stalled, and vice versa. No warp ever switches blocks.

---

## Q27. Are warps like units competing for the SM's hardware, each block with its own registers?

**Yes, that's a good picture, with three refinements.**

**1. There are two kinds of resources.** Some are *divided up* when a block arrives; others are
*taken turns on* every cycle.

| Kind | Resource | How it's used |
|---|---|---|
| **Reserved** for the block's whole life (no competition) | Registers | Each warp's own private set |
| | Shared memory **space** | Each block's own 32 KB slice (Q7) |
| | Warp slots (48 per SM) | 4 per block |
| **Shared**, used in turns every cycle | Warp schedulers (issue slots) | One warp per scheduler per cycle |
| | FP32 math units | |
| | Load/store units, shared memory **bandwidth** (the 32 banks) | Space is private, but the access path is shared |
| | L1 cache | |
| **Shared by the whole GPU** (all SMs, all ~376 resident blocks) | L2 cache, DRAM bandwidth | |

So DRAM bandwidth is not just shared between your 2 blocks. Every block on every SM uses it.

**2. Competition is mostly a good thing here.** The SM is *built* to have more warps than it can
run at once, so that whenever some are stalled, others are ready. Contention only hurts when warps
all need the same busy unit at the same moment, e.g. everyone hammering shared memory or DRAM.

**3. Warps aren't purely software.** A thread block is a software grouping. A warp, once resident,
is a hardware-tracked unit: it occupies one of the 48 warp slots, and the scheduler keeps its
program counter and status (ready / stalled) in hardware. That's why switching between warps costs
nothing (Q26).

Also, warps of the **same** block compete with each other in exactly the same way. Blocks are not
special here; the scheduler only sees warps.

---

## Q28. Does the scheduler give compute to one block's warps and memory fetches to the other block's?

**Closer to the second version, but the scheduler doesn't think in "compute warps" vs "fetch warps"
or in blocks at all.**

**What the scheduler actually does, every cycle:**
1. Look at its warps (from any block).
2. Find the ones that are **ready**: their next instruction's inputs have arrived, and the unit that
   instruction needs can take it this cycle.
3. Issue **one instruction** from one ready warp. It can be any kind: FMA, `ld.shared`, `cp.async`,
   address math, a branch.

Roles aren't assigned. Each warp just runs its own instruction list in order. Because warps are at
**different points** in that list, at any moment some happen to be issuing FMAs, some issuing
copies, and some stalled waiting for data. That mix is natural, and it's independent of which block
a warp belongs to.

**Two corrections to "compute units are locked":**
- **Units are pipelined, not locked.** An FMA takes a few cycles to finish, but a new FMA can start
  on the next cycle. One warp issuing an FMA blocks the unit for about one cycle, not for its whole
  compute phase.
- **A memory fetch barely uses the warp.** Issuing `cp.async` takes one issue slot. The transfer
  then runs in the memory system for hundreds of cycles (Q15). The warp isn't "doing the fetch"
  during that time; it's either running other instructions or stalled waiting for the data.

```
Scheduler 0, cycle by cycle (A = block A's warp, B = block B's warp):

cycle:   1      2      3      4        5      6      7      8
A.w0 :  FMA    FMA    FMA    cp.async [stalled: wait_group / bar.sync .....]
B.w0 :  [stalled on ld.shared] ld.sh  FMA    FMA    FMA    FMA
issued: A FMA  A FMA  A FMA  A copy    B FMA  B FMA  B FMA  B FMA
                                        ↑ A stalls → B has ready FMAs → math stays busy
```

**Each cycle, all 4 schedulers issue at once**, so the SM can start up to 4 instructions per cycle
from 4 different warps, e.g. 3 FMAs and 1 shared-memory load.

---

## Q29. So the scheduler doesn't care which block a warp is from, and gives the current task to whichever warp is free?

**First half: correct.** The scheduler treats all its warps the same, whichever block they belong
to.

**Second half: the other way round.** The scheduler doesn't hand tasks to warps. **Each warp
already has its own task**: its own next instruction, in its own copy of the kernel's code, on its
own data. The scheduler picks a warp that is **ready** (its next instruction can run now) and lets
that warp run it.

```
Wrong picture:  scheduler holds a task  → gives it to a free warp
Right picture:  each warp holds its own next instruction → scheduler picks a READY warp → it runs
```
Analogy: a cashier (the math units) and a queue of customers (warps), each holding their own
shopping (instructions). The cashier serves whichever customer is ready. The cashier never hands
shopping to customers.

**Two small details:**
- Each scheduler only sees **its own** warps. A warp is attached to one of the SM's 4 schedulers
  for its whole life.
- The block matters for **barriers**: `bar.sync` waits only for warps of the same block. That
  affects which warps are ready, but the scheduler itself still doesn't look at block identity.
  NVIDIA doesn't publish the exact rule for choosing between several ready warps.

---

## Q30. So the scheduler's only job is to check warp status and pick a ready warp?

**Yes, essentially.** It decides *who goes next*, never *what to do*. What to do is fixed by each
warp's own instruction stream.

Its job each cycle, slightly fuller:
1. **Track readiness.** It keeps a "scoreboard": which registers are still waiting for a result
   (e.g. a load in flight, an FMA not finished). A warp whose next instruction needs a pending
   register is not ready. Warps waiting at `bar.sync` or `wait_group` aren't ready either.
2. **Pick** one ready warp.
3. **Issue** that warp's next instruction: send it to the right unit (FP32 math, load/store,
   etc.) for all 32 threads of the warp at once.

The units then do the actual work, and the scoreboard is updated when results come back.

---

## Q31. Will writing CUDA C++ + profiling make this clearer than Triton?

**Yes, mostly.** CUDA makes *you* take every decision Triton's compiler hid: which thread handles
which element, shared-memory layout, barriers, coalescing, bank conflicts, pipelining. Getting one
wrong and seeing it in the numbers is the fastest way to understand it.

**Two caveats:**
1. **CUDA doesn't show everything either.** Register allocation, blocks per SM, warp scheduling and
   stalls are still done by the compiler and hardware. You see those only by **profiling**. The
   profiler is the real teacher, in both languages.
2. **Triton isn't a black box.** Reading TTGIR and PTX (as in Q1–Q4) shows what it decided. A
   good loop: write it in CUDA to understand it, then check how Triton did the same thing.

**Tools on this pod (checked 2026-10-17):** `nvcc` and `ncu` (Nsight Compute) from CUDA 12.8 at
`/usr/local/cuda/bin`; CUDA 12.8 supports sm_120. Not yet tested here: `ncu` needs GPU
performance-counter permission, which containers sometimes block (error `ERR_NVGPUCTRPERM`). Also,
a vLLM server was holding ~93.5 GB of GPU memory; profile when it's stopped.

### Suggested path: one matmul, improved step by step
Each step fixes one problem. Profile each with `ncu` and compare against Triton and
`torch.matmul` (cuBLAS).

| Step | Kernel | What you learn | `ncu` metric to watch |
|---|---|---|---|
| 1 | Naive: one thread = one output | Baseline; thread indexing | Memory throughput, achieved FLOPs |
| 2 | Coalesced indexing | Why neighbour threads should read neighbour addresses | Global load efficiency |
| 3 | Shared-memory tiling + `__syncthreads` | Reuse, barriers (Q19) | DRAM / L2 traffic drops |
| 4 | Each thread computes a 4×4 (or 8×8) block | Register reuse, same as Triton's `#blocked2` (Q4) | Registers per thread, occupancy |
| 5 | `float4` 16-byte loads | Vectorization, alignment (Q4) | Load instructions issued |
| 6 | Padding / swizzle | Bank conflicts (Q20, Q21) | Shared-memory bank conflicts |
| 7 | `cp.async` double buffering | Async copies, pipelining (Q15, Q20) | Warp stall reasons |
| 8 | Tensor Cores (`wmma` / `mma`) | Why Triton's `#mma` matters | Tensor pipe utilization |

Most useful `ncu` sections: **Occupancy** (theoretical vs achieved, and which limit decides it,
like Tool 2 of the lab page) and **Warp State Statistics** (why warps were *not ready*: the
scheduler picture from Q26–Q30).

Well-known reference that follows this exact path: Simon Boehm, "How to Optimize a CUDA Matmul
Kernel for cuBLAS-like Performance" (siboehm.com).

---

## Q32. Plan: profile my LeetGPU Triton solutions and start CUDA C++ in parallel. How do I profile Triton kernels?

**Plan: good.** Best way to run both tracks: **solve the same LeetGPU problem in both languages**
and profile them side by side. You'll see exactly which decisions Triton made for you, and where
your CUDA version wins or loses.

**Nsight Compute works on Triton kernels too.** At runtime a Triton kernel is an ordinary CUDA
kernel (a cubin), so `ncu` profiles it exactly like a CUDA C++ kernel. The kernel name is your
Python function name:
```bash
ncu --kernel-name regex:matrix_multiplication --launch-skip 1 --launch-count 1 \
    --set full -o matmul_report python kernels/matmul/matmul_triton_tiled.py
```
- `--launch-skip 1` skips the first launch (it includes compiling; with autotune, skip all the
  tuning runs too).
- `--set full` collects every section (occupancy, memory, warp stalls, source).
- The PTX has line info (the `.loc` lines from Q1), so the **Source** page should map hot
  instructions back to your Python lines.

**Triton's own tools:**

| Tool | Gives you | How |
|---|---|---|
| `triton.testing.do_bench` | Reliable kernel time (warm-up, repeats) | `ms = triton.testing.do_bench(lambda: solve(a, b, c, M, N, K))` |
| `triton.testing.perf_report` | Sweep sizes, compare vs `torch`, auto-plot | Decorator on a benchmark function |
| Compile stats | Registers, spills, shared memory | `k.n_regs`, `k.n_spills`, `k.metadata.shared` |
| `TRITON_PRINT_AUTOTUNING=1` | Which autotune config won, and its time | Environment variable |
| **Proton** (`triton.profiler`) | Triton's built-in lightweight profiler (kernel times, scopes) | `import triton.profiler as proton` (installed here) |
| `torch.profiler` | Timeline of all kernels, Triton included | Standard PyTorch profiler |
| Triton cache `~/.triton/cache/` | Every compiled kernel's TTIR / TTGIR / PTX / cubin + a JSON with metadata | Browse the files |
| `TRITON_INTERPRET=1` | Runs on CPU for **debugging correctness** (not speed) | Already commented in your script |

**What to compare when profiling the same problem in both languages:**
- **Time and achieved bandwidth/FLOPs** vs `torch` (`do_bench`).
- **Registers, shared memory, occupancy** and which limit decides blocks per SM.
- **Memory throughput** (% of peak DRAM bandwidth): memory-bound problems (vector add, reductions,
  softmax) should get close to the peak.
- **Warp stall reasons**: why warps weren't ready.

**On this pod (checked 2026-10-17):** `ncu` is installed (CUDA 12.8). `nsys` (Nsight Systems, the
timeline tool) is **not**. Proton is available in Triton 3.4. The Triton cache lives under `/root`,
so it's wiped on pod restart. Still untested here: whether the container allows `ncu` to read GPU
performance counters.

---

## Q33. CUDA's "blocks per grid" vs Triton's grid. Is it the output spread over all available threads?

**CUDA's "blocks per grid" *is* Triton's grid.** Both are the number of thread blocks (= pids) you
launch, and you compute it the same way: **problem size ÷ tile size**. It is **not** tied to how
many threads or SMs the GPU has. The hardware then runs as many blocks at once as fit, in waves
(Q8, Q12).

| Concept | Triton | CUDA C++ |
|---|---|---|
| Number of blocks (grid) | `grid = (cdiv(M,64), cdiv(K,64))` | `gridDim`: `dim3 grid(cdiv(K,64), cdiv(M,64))` |
| Threads per block | `num_warps` × 32 (e.g. `num_warps=8` → 256) | `blockDim`: `dim3 block(256)`, any count ≤ 1,024 (use multiples of 32) |
| Tile size (work per block) | Explicit: `BLOCK_SIZE_*` constexprs | **Implicit**: decided by your code (how many outputs each thread computes) |
| "Which block am I?" | `tl.program_id(0)` | `blockIdx.x` |
| "Which thread am I?" | Hidden (compiler decides, via layouts) | `threadIdx.x` |
| Launch | `kernel[grid](..., num_warps=8)` | `kernel<<<grid, block>>>(...)` |

**Tile size still exists in CUDA; it just isn't a launch parameter.** It follows from threads per
block × outputs per thread, which you write in the kernel:
- 16×16 = 256 threads, 1 output each → 16×16 tile → `grid = cdiv(M,16) × cdiv(K,16)`
- 256 threads, a 4×4 block each → 64×64 tile → `grid = cdiv(M,64) × cdiv(K,64)` (same as your
  Triton kernel with `num_warps=8`)

So in both languages: **choose the tile → grid = problem ÷ tile → choose threads per block.** In
Triton, you write the tile and the compiler divides it among threads. In CUDA, you write the
per-thread work and the tile follows from it.

**The exception: persistent kernels.** Some advanced kernels launch exactly as many blocks as fit
at once (e.g. 188 SMs × 2 = 376), and each block **loops over many tiles**. That's the "spread over
the whole GPU" idea. It avoids per-block start-up cost and gives control over the tail, but it's a
deliberate design choice, not the default.

---

## Q34. Triton defines the tile in output elements, CUDA in threads. Right?

**Almost. Better: both have a tile *in elements* and a thread count. The difference is which one you
write directly, and who connects them.**

| | Triton | CUDA C++ |
|---|---|---|
| You write directly | Tile **in elements** (`BLOCK_SIZE_*` = 64×64) **and** `num_warps` | Threads per block (`blockDim` = 256) **and** each thread's work, in code |
| Tile in elements | Explicit | **Derived**: threads × elements per thread (256 × 16 = 64×64) |
| Who maps elements → threads | **Compiler** (layouts, Q3/Q4) | **You** (index math with `threadIdx`) |

Only if each thread computes exactly **one** output do the CUDA thread count and the tile size
coincide (e.g. 16×16 threads = 16×16 tile). Otherwise, the tile in elements is bigger than the
thread count.

One-liner: **Triton: "here's the tile, here are the warps, you split it." CUDA: "here are the
threads, and here's exactly what each one does."**

---

## Q35. If grid = problem size ÷ threads per block, isn't each thread computing exactly 1 element?

**Yes, in that setup.** With 16×16 threads per block and grid = (N/16, M/16), each block covers a
16×16 output tile and each thread computes **1** element. That's the standard first CUDA matmul.

**But it's a choice, not a rule.** The grid formula is really
```
grid = problem size ÷ (threads per block × outputs per thread)
```
and "1 output per thread" is just the simplest case. Keep 16×16 threads but give each thread a
4×4 block of outputs, and the tile becomes 64×64, so grid = (N/64, M/64).

**1 output per thread (16×16 tile):**
```cpp
__global__ void matmul_1(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}
dim3 block(16, 16);
dim3 grid(cdiv(N, 16), cdiv(M, 16));
```

**4×4 outputs per thread (64×64 tile), same 256 threads:**
```cpp
__global__ void matmul_4x4(const float* A, const float* B, float* C, int M, int N, int K) {
    int row0 = blockIdx.y * 64 + threadIdx.y * 4;   // this thread's first row
    int col0 = blockIdx.x * 64 + threadIdx.x * 4;   // this thread's first col
    float acc[4][4] = {};                           // 16 accumulators in registers
    for (int k = 0; k < K; ++k) {
        float a[4], b[4];
        for (int i = 0; i < 4; ++i) a[i] = (row0 + i < M) ? A[(row0 + i) * K + k] : 0.f;
        for (int j = 0; j < 4; ++j) b[j] = (col0 + j < N) ? B[k * N + col0 + j] : 0.f;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];   // 8 loads → 16 FMAs
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (row0 + i < M && col0 + j < N) C[(row0 + i) * N + col0 + j] = acc[i][j];
}
dim3 block(16, 16);
dim3 grid(cdiv(N, 64), cdiv(M, 64));
```
(No shared memory yet, to keep the indexing visible. Shared-memory tiling is step 3 of the path in
Q31.)

**Rules:**
- The grid formula and the kernel's index math **must agree**. If the grid assumes 64×64 tiles but
  the kernel indexes 16×16, most of C never gets written. In Triton, the `BLOCK_SIZE_*` used in
  both places keeps them in sync automatically.
- Why more outputs per thread: each loaded value is reused 4 times (8 loads → 16 FMAs instead of
  2 loads → 1 FMA). It's the same reason Triton picked `sizePerThread = [4, 4]` (Q4).
- Convention: `x` = columns, `y` = rows, so neighbouring threads (consecutive `threadIdx.x`) read
  neighbouring columns of B and write neighbouring columns of C. That's coalescing (Q3).

---

## Q36. Triton has `triton.autotune`. How do you tune a CUDA C++ kernel?

**There's no built-in autotuner in CUDA C++. You build the same loop yourself:** make the knobs
compile-time parameters, build several versions, time each one, keep the fastest.

**1. Make the knobs template parameters.** Tile sizes decide register-array sizes, shared-memory
sizes and loop unrolling, so they must be known at compile time (like Triton's `tl.constexpr`):
```cpp
template <int BM, int BN, int BK, int TM, int TN>   // tile + per-thread outputs
__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BK][BM], Bs[BK][BN];
    float acc[TM][TN] = {};
    ...
}
```

**2. Instantiate several configs, time each, keep the best:**
```cpp
template <int BM, int BN, int BK, int TM, int TN>
float time_config(/* args */) {
    dim3 block((BN / TN) * (BM / TM));                 // threads per block
    dim3 grid(cdiv(N, BN), cdiv(M, BM));
    for (int i = 0; i < 3; ++i) matmul<BM,BN,BK,TM,TN><<<grid, block>>>(...);   // warm-up
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 20; ++i) matmul<BM,BN,BK,TM,TN><<<grid, block>>>(...);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    return ms / 20;
}
// time_config<64,64,8,4,4>(...);  time_config<128,128,8,8,8>(...);  ... pick the minimum
```
`cudaEvent` timing measures GPU time. Always warm up first and average many runs.

**Easier options:**

| Tool | What it does |
|---|---|
| **Python driver** | Compile CUDA variants from Python (`torch.utils.cpp_extension.load_inline`, or CuPy `RawModule`) and time them with **`triton.testing.do_bench`**. Same workflow as Triton autotune. |
| **Kernel Tuner** (`pip install kernel_tuner`) | Open-source autotuner for CUDA kernels: give it a search space of `#define`s / template values; it compiles, times and reports the best. |
| **NVBench** | NVIDIA's C++ benchmarking library, with built-in parameter sweeps. |
| **`cutlass_profiler`** | Comes with CUTLASS; times every CUTLASS kernel config for a given problem. |
| `cudaOccupancyMaxPotentialBlockSize` | CUDA API suggesting the block size with the highest occupancy. A starting point only: max occupancy ≠ fastest (Q25). |

**How it differs from Triton autotune:**
- **When:** Triton tunes **at runtime**, on the first call for each new shape (`key`). CUDA kernels
  are usually tuned **offline**; the winning configs are hard-coded, or put in a lookup table by
  shape (cuBLAS ships pre-tuned heuristics per GPU and shape).
- **Guided by profiling:** in CUDA you usually change configs because `ncu` showed a specific
  problem (spills, bank conflicts, low occupancy, memory stalls), rather than sweeping blindly.

---

## Q37. Which has more job demand (e.g. on LinkedIn): Triton or CUDA C++? (checked 2026-10-18)

**CUDA, by a wide margin in volume. Triton mostly appears as a companion skill.** No public
head-to-head LinkedIn count exists, and LinkedIn itself couldn't be queried, so this is based on
job listings and aggregators found by web search.

- **CUDA is used far more widely:** AI, HPC, scientific computing, robotics, graphics, finance,
  inference engines. A secondhand report (citing Lightcast; original not found, so unverified) says
  US postings requiring CUDA in Jan–Aug 2026 already exceeded all of 2025.
- **Triton shows up mainly in ML / LLM kernel roles, usually listed with CUDA:** e.g. "CUDA,
  Triton, or Pallas kernels" (RadixArk), "Triton, CUTLASS, custom CUDA kernels" (CoreWeave),
  "Triton, CUDA, or HIP" (AMD). Triton *compiler* roles (NVIDIA, OpenAI) still ask for CUDA, LLVM
  and MLIR.
- **Raw keyword counts mislead:** "Triton" also matches NVIDIA's *Triton Inference Server* and
  unrelated companies named Triton. Search "Triton kernel" or "OpenAI Triton" instead.

**Takeaway:** learn both, CUDA as the foundation. Interviews for GPU-kernel roles test CUDA-level
understanding (memory hierarchy, coalescing, occupancy, shared memory, warps), which is what
Q1–Q36 cover. Triton is the productivity skill on top, valued especially for LLM work and on AMD
(where it also targets ROCm).

---

## Q38. My one-line summary of the core concepts. Is it right? (cheat sheet)

| # | User's summary | Verdict | Refinement |
|---|---|---|---|
| 1 | Memory hierarchy: DRAM → L2 → shared memory → registers | ✓ | **L1** sits beside shared memory (same SRAM, Q17). Each level down is smaller and faster. The goal is to load data once from a slow level and reuse it many times from a fast one (tiles in shared memory, accumulators in registers). |
| 2 | Coalescing: consecutive threads access contiguous memory | ✓ | The payoff: a warp's 32 accesses merge into a few wide memory transactions. Scattered or strided access = many transactions and wasted bandwidth. |
| 3 | Occupancy: whether all SMs are fully busy | ✗ | Occupancy is **per SM**: resident warps ÷ max warps (48 here), set by the tightest of registers / shared memory / threads / block slots (Q9). "Are all SMs busy" is a different question: grid size and the tail (Q25). Higher occupancy ≠ always faster. |
| 4 | Shared memory: reserved per block; swizzling gives single-step row/column access via rotation/XOR | ✓ mostly | The core idea is **bank conflicts**: 32 banks; a warp is served in one step only if its threads hit different banks. Swizzle or padding avoids conflicts (Q20). Also: shared memory is a scratchpad **you** control, used for reuse and for passing data between threads of a block. |
| 5 | Warps: groups of 32 threads | ✓ | A warp is the unit the scheduler issues: all 32 threads run the **same instruction** together (Q26–Q30). |

### Related core topics (interviews ask about these too)
- **Latency hiding:** keep math units busy while data is in flight, with many ready warps or lots
  of independent work per warp (Q25, Q26).
- **Compute-bound vs memory-bound (roofline / arithmetic intensity):** FLOPs per byte of the kernel
  vs FLOPs per byte the GPU can do (~65 here) (Q25).
- **Register pressure and spills** (Q9, Q25).
- **Barriers and async copies** (Q19, Q20).
- **Warp divergence (not yet covered):** if threads of one warp take different branches of an `if`,
  the warp runs **both** paths, with the threads that didn't take each path switched off. Costs
  time in branchy code.

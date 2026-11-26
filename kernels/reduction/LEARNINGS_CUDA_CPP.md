# reduction: CUDA C++ learnings

What writing the CUDA C++ reductions taught. Lessons that apply beyond reduction are in
[`learnings/CUDA_CPP_GENERAL.md`](../../learnings/CUDA_CPP_GENERAL.md). The problem, variants,
performance and open issues are in [`README.md`](README.md).

---

## 1. Accumulate in a register, write shared memory once
*Added 2026-11-26, from reviewing [`reduction_smem_tree.cu`](reduction_smem_tree.cu).*

In `reduction_smem_tree.cu`, each thread first sums its share of the block's elements into its own
shared-memory slot `blockpartialsum_smem[tid]`, and then the block does a tree reduction over the
slots. The first phase reads and writes shared memory on **every** loop iteration. We can avoid that:
each thread accumulates in a **register** (a local `float`), and writes the result to its
shared-memory slot **once**, just before the tree reduction.

**Current code:**
```cpp
blockpartialsum_smem[threadIdx.x] = 0;

__syncthreads();

for (int i = tid; i < BLOCK_SIZE; i += blockDim.x)
{
    if ((block_start + i) < N)
    {
        blockpartialsum_smem[tid] += input[block_start + i];   // smem read + write every iteration
    }
}

__syncthreads(); // tree based reduction should only start once the shared memory array is fully computed
```

**With a register accumulator:**
```cpp
float acc = 0.0f;                                // one register per thread

for (int i = tid; i < BLOCK_SIZE; i += blockDim.x)
{
    if ((block_start + i) < N)
    {
        acc += input[block_start + i];           // register only, no shared memory traffic
    }
}

blockpartialsum_smem[tid] = acc;                 // one shared-memory write per thread

__syncthreads(); // tree based reduction should only start once the shared memory array is fully computed
```

The tree reduction after this is unchanged.

**Three points when making the change:**

1. **`0.0f`, not `0`.** The `f` makes it a float literal. Plain `0` works here, but the habit
   matters: an unsuffixed literal like `0.5` is a `double`, and a `double` in an expression pulls
   the calculation into slow double-precision math on the GPU.
2. **The old zeroing and the `__syncthreads()` after it go away.** `blockpartialsum_smem[tid] = acc;`
   sets every slot, so there's nothing left to zero. (That sync wasn't needed even before: each
   thread zeroed and then accumulated into **its own** slot, and a thread always sees its own
   earlier writes. `__syncthreads()` is needed only when a thread reads data that a **different**
   thread wrote, as in the tree reduction, where thread `tid` reads slot `tid + stride`.)
3. **The write `blockpartialsum_smem[tid] = acc;` goes before the `__syncthreads()`, and every
   thread does it unconditionally.** That includes threads in the last block whose elements are
   all past `N`: their `acc` stays `0.0f`. The tree reads all `blockDim.x` slots, so a slot that
   was never written would add garbage. Keeping the `if` inside the loop and the write after the
   loop handles this.

**Expected speedup: small.** At N = 100M the kernel was already at 94% of peak DRAM bandwidth
(0.421 ms, RTX 4090), so the shared-memory traffic removed here wasn't the bottleneck. The gains
are cleaner code and one fewer barrier. Re-time after the change to confirm.

---

## 2. Why N = 100M can fail the tolerance: too many adds into one float
*Added 2026-11-26, from testing [`reduction_smem_tree.cu`](reduction_smem_tree.cu) and
[`reduction_smem_atomic.cu`](reduction_smem_atomic.cu) on an RTX 4090.*

Both kernels pass the LeetGPU check (`|got - ref| <= atol + rtol * |ref|`, atol = rtol = 1e-5,
`ref` summed in `double` on the host) for N = 1 … 15M. At N = 100M with values in [0, 1000] both fail:
error ~6.7e5 against a tolerance of 5.0e5.

**This isn't a bug; it comes from the design.** Each block adds its partial sum into `output[0]`
with one `atomicAdd`, so the final float receives N / 1024 ≈ **97,700 additions**. The running total
grows to ~5e10, and at that size neighbouring floats are 4096 apart, so each add rounds by up to
±2048. Those rounding errors pile up. The error is set by **how many additions go into one float
and how large that float is**, not by the kernel's logic.

- With values in [-1000, 1000] the sum mostly cancels (~5e6), so the rounding stays small and the
  check passes, but the margin is thin because `rtol * |ref|` is also small.
- LeetGPU's large test is N = 15M (values in [0, 1000]), which passes.
- **The fix is structural:** fewer partial sums going into the final float. A grid-stride loop
  (a fixed number of blocks, each summing many elements), or a second pass over the per-block
  partial sums. Both are the next variants on the roadmap.

---

## 3. All threads atomically adding to one address: contention and run-to-run variation
*Added 2026-11-26, from timing [`reduction_smem_atomic.cu`](reduction_smem_atomic.cu).*

| N = 100M, RTX 4090 | time | GB/s | % of 1,008 GB/s spec |
|---|---|---|---|
| `reduction_smem_tree.cu` | 0.422 ms | 948 | 94.1% |
| `reduction_smem_atomic.cu` | 2.594 ms | 154 | 15.3% |

**Why the atomic version is ~6x slower:** all 128 threads of a block `atomicAdd` into the **same
shared-memory address** (`block_sum`). Atomics to one address can't run in parallel; the hardware
does them one after another. That's 1,024 sequential adds per block, so the kernel is limited by
that serial chain, not by DRAM. The tree version does the same work with `log2(128) = 7` parallel
rounds.

**The result changes from run to run.** The order in which the atomics land isn't fixed, and
floating-point addition isn't associative, so the last bits change between runs. At N = 1024 with
values in [-1000, 1000] (sum ≈ 530), 5 runs gave errors from 1.2e-4 to 6.4e-3 against a tolerance of
5.3e-3: 2 of 5 failed. A 100M run once failed with error 9.8 against a tolerance of 4.6 (sum ≈ 4.6e5).
The tree version varies too (its per-block `atomicAdd` order isn't fixed), but much less.

Lesson: atomics on a **single** hot address serialize. Use them for the last step (one per block),
not for every element.

---

## 4. Timing a reduction: bytes, the copy reference, and the L2 cache
*Added 2026-11-26.*

- **Bytes moved = N x 4** (the input is read once; the single-float write is negligible). The D2D
  copy moves 2 x N x 4 (read + write).
- **"% of D2D copy" above 100% is normal here** (the tree reduction: 103.6% at N = 100M). A read-only
  stream is a little easier on DRAM than the copy's mix of reads and writes, so a reduction can beat
  the copy's GB/s. Compare against the spec too.
- **Small N measures the L2 cache, not DRAM.** At LeetGPU's benchmark size, N = 4,194,304 (16 MB),
  the input fits in the 4090's 72 MB L2. With back-to-back cudaEvent timing (warm L2) the tree
  kernel showed "157% of spec". `do_bench` flushes L2 before every call, so the Triton script
  doesn't show this. See
  [`learnings/PROFILING_GENERAL.md`](../../learnings/PROFILING_GENERAL.md) §6 (do_bench flushes L2)
  and §8 (problem-size regimes). Use N = 100M for DRAM numbers.

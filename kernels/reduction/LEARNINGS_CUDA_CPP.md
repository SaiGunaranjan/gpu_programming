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
- **LeetGPU's benchmark size measures the L2 cache, not DRAM bandwidth.** N = 4,194,304 floats is
  only 16 MB, which fits in the RTX 4090's 72 MB L2. When the same input is timed over and over
  without flushing L2 (our `time_gpu()` loop), every call after the first reads the input from L2,
  so the "GB/s" is L2 bandwidth: the tree kernel shows **135% of the DRAM spec**, which DRAM can't
  deliver. `do_bench` flushes L2 before every call, so the Triton and torch numbers at this size
  *are* DRAM reads, but of a transfer too small to reach full speed (launch and ramp-up time
  dominate). Either way, **the 4M number isn't the true DRAM bandwidth**: use N = 100M (400 MB, far
  bigger than L2) for that. Whether LeetGPU's own timer flushes L2 isn't known, so its score for
  this problem may include L2 hits. See
  [`learnings/PROFILING_GENERAL.md`](../../learnings/PROFILING_GENERAL.md) §6 (do_bench flushes L2)
  and §8 (problem-size regimes).

**All variants at both sizes** (RTX 4090, 1,008 GB/s spec; medians, 3 runs each, which agreed to
within ~1 us; measured 2026-11-26):

| variant | timer | N = 4,194,304 (16 MB) | N = 100,000,000 (400 MB) |
|---|---|---|---|
| `reduction_smem_tree.cu` | `time_gpu` (L2 warm) | 12.3 us, 1,365 GB/s, **135% (L2)** | 0.422 ms, 948 GB/s, **94.1%** |
| `reduction_smem_atomic.cu` | `time_gpu` (L2 warm) | 161.8 us, 104 GB/s, 10.3% | 2.594 ms, 154 GB/s, 15.3% |
| `reduction_triton.py` (multi-pass) | `do_bench` (L2 flushed) | 36.0-36.9 us, ~460 GB/s, ~46% | 0.488 ms, 819 GB/s, 81.2% |
| `torch.sum` | `do_bench` (L2 flushed) | 32.8 us, 512 GB/s, 50.8% | 0.483 ms, 828 GB/s, 82.1% |
| D2D copy (reference) | each script's own | `cudaMemcpy` ~2,980 GB/s (L2, not DRAM); `copy_` 819 GB/s | `cudaMemcpy` 915 GB/s; `copy_` 914 GB/s |

- **At 100M the comparison is fair** (both timers read DRAM; L2 can't hold 400 MB), and the
  shared-memory tree kernel beats `torch.sum` by ~13 percentage points (94.1% vs 82.1%).
- **At 4M the CUDA and Triton/torch rows aren't comparable:** the CUDA rows read from a warm L2,
  the others from DRAM after a flush.
- At 100M the tree kernel failed the tolerance in 1 of 3 runs (a small sum, -6.5e5, so a small
  `rtol x |ref|`; see §2).

---

## 5. First warp shuffle: `__shfl_down_sync` for the last 5 tree rounds
*Added 2026-11-27, from [`reduction_smem_tree_warp_shuffle.cu`](reduction_smem_tree_warp_shuffle.cu).*

The tree reduction in `reduction_smem_tree.cu` runs down to stride 1 through shared memory, with a
`__syncthreads()` after every round. Once the stride is 16 or less, all the slots still being added
belong to warp 0 (threads 0-31), so the last 5 rounds can use a warp shuffle instead:

```cpp
while (stride > 0)   // stride = 16, 8, 4, 2, 1
{
    blockpartialsum_smem[tid] += __shfl_down_sync(0xffffffff, blockpartialsum_smem[tid], stride);
    stride /= 2;
}
```

**What `__shfl_down_sync(mask, value, delta)` does:** every lane passes in `value` and gets back the
`value` of lane `lane + delta` of the **same warp**. The data moves from register to register,
without going through shared memory. It works only inside one warp (32 lanes): to combine values
from different warps you still need shared memory (or atomics).

**The mask** is a 32-bit value with one bit per lane (`0xffffffff` = all 32). The lanes set in it
must all call the same `__shfl_down_sync`, and the call waits until they have all arrived. So no
`if (tid < stride)` around it (as the shared-memory tree has): every lane in the mask has to execute
it. That wait covers **only the masked lanes of this warp**, never other warps, so it can't replace a
`__syncthreads()` when data crosses warps.

**Why the last 5 rounds need no `__syncthreads()`:** a barrier is needed only when a thread reads
shared memory that a **different** thread wrote. Here each thread reads and writes only its own slot
`smem[tid]`, and the other lane's value arrives through the shuffle, already synchronized by it.

**Lanes whose source is past lane 31 get their own value back.** At stride 16, lanes 16-31 have no
lane `+16` in the warp, so they receive their own value and double it (in the shared-memory tree,
the `if (tid < stride)` just skips them). This is harmless: each round lane 0 reads lane `stride`,
and lanes below `stride` always hold correct partial sums, so the doubled values never reach lane 0.
Warps 1-3 also run the loop on their own slots; their results are never used.

**Tested 2026-11-26 (RTX 4090):**
- On a fixed seed, the result is bit-identical to `reduction_smem_tree.cu` for every single-block
  size (N = 1 … 1025). Lane `i` adds lane `i + stride`, the same pairing as the tree, so the
  rounding is the same.
- Larger N differs only through the order of the per-block `atomicAdd`s.
- `compute-sanitizer`: memcheck 0 errors, racecheck 0 hazards.
- Same time: 0.421 ms vs 0.4211 ms at N = 100M (the kernel is DRAM-bound, and the 5 rounds were a
  tiny part of it).
- The occasional failure at N = 100M with values in [-1000, 1000] is the §2 tolerance limit, seen
  with the tree version too (1 of 10 vs 2 of 10 random runs), not the shuffle.

**Why this isn't the optimal pattern:** the values still live in the 128-float shared-memory array,
and each shuffle round reads and writes it. The usual pattern keeps the per-thread sum in a register
(§1), shuffles it within **every** warp, writes one value per warp to a small shared array
(4 floats), does one `__syncthreads()`, and lets warp 0 shuffle those 4 values. That's the next
variant.

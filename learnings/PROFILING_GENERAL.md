# Profiling: general learnings

How to time a GPU kernel correctly, turn the time into GB/s and FLOP/s, and tell whether it is
memory-bound or compute-bound. This applies to CUDA C++, Triton and torch alike. Worked example
throughout: vector add on an RTX A4000 ([`kernels/vector_add/`](../kernels/vector_add/)).

Measured 2026-10-30 to 11-03 on an RTX A4000 (sm_86), CUDA 12.8, driver 595.91.07,
torch 2.8.0+cu128, Triton 3.4.0. The GPU varies between pods: re-derive the peaks (§1) on a new one.

## 1. The hardware limits, and how to get them

| quantity | where it comes from | RTX A4000 |
| --- | --- | --- |
| SM count, L2 size | `torch.cuda.get_device_properties(0)` | 48 SMs, 4 MB L2 |
| memory bus width | `cudaDeviceGetAttribute(&v, cudaDevAttrGlobalMemoryBusWidth, 0)` | 256 bits |
| memory clock | `cudaDevAttrMemoryClockRate`, or `nvidia-smi -q -d CLOCK` | 7001 MHz |
| SM clock (rated boost) | `cudaDevAttrClockRate` | 1560 MHz (max under boost: 2100) |
| FP32 cores per SM | not queryable: CUDA Programming Guide, "arithmetic instructions" throughput table | 128 (cc 8.6) |

- **Peak DRAM bandwidth** = memory clock x 2 (GDDR6 transfers twice per clock) x bus width in bytes
  = 7001 MHz x 2 x 32 B = **448 GB/s**.
- **Peak FP32** = SMs x FP32 cores per SM x 2 (an FMA is 2 FLOPs) x clock
  = 48 x 128 x 2 x 1.56 GHz = **19.2 TFLOP/s**.
- The clock under load can be above the rated 1.56 GHz, and no kernel reaches the spec bandwidth.
  So also **measure a practical peak**: a device-to-device copy (`cudaMemcpy(..., cudaMemcpyDeviceToDevice)`
  or `c.copy_(a)`), which reads one array and writes one. On the A4000: 386 GB/s, 86% of spec.
  It is a reference, not a hard ceiling: vector add (2 reads per write) reaches 393 GB/s, because
  DRAM loses time switching between reading and writing and the copy switches more.

## 2. Arithmetic intensity and the ridge point

**Arithmetic intensity** = FLOPs the operation must do / minimum bytes it must move to and from DRAM
(each input read once, each output written once). float32 = 4 bytes.

| operation | FLOPs | minimum bytes | intensity |
| --- | --- | --- | --- |
| vector / matrix add | 1 per element | 12 per element | 0.083, at every size |
| matmul, A M x N, B N x K (LeetGPU naming) | 2MNK | 4(MN + NK + MK) | square n: n/6 (170 at n = 1024) |

"Minimum" matters: a real kernel can move more (a naive matmul re-reads A and B many times), so
its real intensity is lower. Tiling into shared memory raises it back toward the minimum.

**Ridge point** = peak FLOP/s / peak bandwidth = 19.2e12 / 448e9 = **~43 FLOP/byte** on the A4000.
A kernel below it is memory-bound, above it can be compute-bound.

This is the same test as comparing times: compute-bound when FLOPs / peak_FLOPs > bytes / peak_BW.
Rearranged: FLOPs / bytes > peak_FLOPs / peak_BW, i.e. intensity > ridge.

| | memory time | compute time | verdict |
| --- | --- | --- | --- |
| vector add, 64M elements | 805 MB / 448 GB/s = 1.8 ms | 67 MFLOP / 19.2 TFLOP/s = 3.5 us | memory-bound, 500x (0.083 vs 43) |
| matmul, n = 1024 | 12.6 MB: 28 us | 2.15 GFLOP: 112 us | compute-bound, 4x (170 vs 43) |

The ratio form separates the kernel (intensity) from the hardware (ridge), and it is the x-axis of
the **roofline**: the best possible performance at intensity I is min(peak_FLOPs, I x peak_BW).
A measured kernel is a dot below that line; the gap is what it leaves on the table.

## 3. What happens between compile and run

- `nvcc` compiles **ahead of time**: `.cu` -> PTX (portable intermediate code) -> SASS (machine
  code for the GPU, e.g. sm_86 with `-arch=native`). Both are embedded in the executable.
- The first CUDA call (e.g. `cudaMalloc`) creates the **context**.
- **Lazy loading** (the default since CUDA 12.2): at a kernel's first launch, the driver copies its
  SASS into GPU memory (DRAM; instructions are then fetched through L2 and the SMs' instruction
  caches). If only PTX matches the GPU, the driver JIT-compiles it (cached in `~/.nv/ComputeCache`).
- **Triton** compiles at the first call, from Python (-> PTX -> cubin), caches it in
  `~/.triton/cache`, then loads it the same way. The first call takes hundreds of ms.
- Each launch (`<<<>>>` or a Triton/torch call) packs the grid, block and arguments into a command
  on the stream's queue and **returns immediately**. The GPU reads commands in order; the block
  scheduler hands blocks to SMs as they free up.

## 4. Timing with CUDA events

An event is a marker in the GPU's queue; the GPU writes a timestamp when it reaches it.

```cpp
cudaEventRecord(start);                    // queued, returns immediately
for (int r = 0; r < R; ++r) kernel<<<g, b>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);                // the only blocking call
cudaEventElapsedTime(&ms, start, stop);    // GPU time between the markers, ms
```

- A host timer (`std::chrono`) without a sync only measures queuing the launches.
- Resolution is ~0.5 us, so time many calls, not one.
- `cudaDeviceSynchronize()` waits for all GPU work; `cudaEventSynchronize(e)` waits until that marker.
- torch exposes the same API: `torch.cuda.Event(enable_timing=True)`, `.record()`,
  `.synchronize()`, `.elapsed_time(other)`.
- Reusable harness: `time_gpu(fn, warmup, reps)` in
  [`kernels/vector_add/vector_add.cu`](../kernels/vector_add/vector_add.cu): an event before the
  first call and after every call, so each call gets its own time; returns median, min and mean.

**Warm-up** pays the one-time costs before timing: the lazy module load, clocks ramping up from
idle, Triton's compile. Measured on vector add, 25M elements, no warm-up: launch 0 took ~850 us,
every later launch ~763 us. No clock ramp was visible (memory-bound, memory clock already up), but
compute-bound kernels depend on the SM clock, so keep a few warm-up calls.

**Repeats, and the median.** Even after warm-up, times vary: clocks drift with temperature and
power, DRAM refreshes, kernels have small gaps between them. Report the median of many calls; it
ignores outliers. On vector add the median was stable across runs to ~0.1%.

## 5. What event markers actually measure

**Event time is time on the GPU's clock between two markers, including any time the GPU sat idle
waiting for the host to queue the next command.**

Back-to-back calls are pipelined across calls (while the GPU runs call k, the host queues k+1),
so the steady-state time per call is **max(host launch time, GPU time)**:

```
Host slower (Triton, small N: host 20 us, GPU 4 us)
host : [launch 1 ....20us....][launch 2 ....20us....][launch 3 ...
GPU  :              [k1 4us]  idle 16us   [k2 4us]  idle 16us  [k3]
                    -> 20 us per call: you measure Python

GPU slower (25M: host 20 us, GPU 767 us)
host : [l1][l2][l3][l4]... runs ahead, queue fills up
GPU  :     [k1 ......767us......][k2 ......767us......][k3 ...
                    -> 767 us per call: you measure the GPU
```

Per-call markers don't help when the host is slower: the host records `start`, then spends ~20 us
launching the kernel, while the GPU has already reached `start` and waits. An interval excludes
host time only if everything inside it was **already queued** when the GPU reached the start marker.

Two different "launch overheads":

| | what | measured |
| --- | --- | --- |
| host-side | CPU time to queue one launch | Triton from Python ~21 us, torch ~6 us, C++ `<<<>>>` ~7 us |
| GPU-side | fixed cost per kernel on the GPU (start, drain, events) | ~4 us floor, however little work |

## 6. `triton.testing.do_bench`

`do_bench(fn, warmup=25, rep=100, quantiles=None, return_mode="mean")`, from the Triton 3.4 source:

- `warmup` and `rep` are **milliseconds**, not counts. It times 5 calls to estimate one, then derives
  the counts.
- It **flushes L2 before every timed run**, by zeroing a 256 MB buffer. There is no option to turn
  that off; for warm-L2 timing, write your own loop.
- Each run has its own start/end events, **after** the flush, so the flush is not timed. The flush
  (~0.6 ms of GPU work) also gives Python time to queue start, kernel and end, so the result is
  close to pure GPU time even for tiny kernels (§5).
- **The default return is the mean**, although its docstring says median. Use
  `return_mode="median"` or `"all"`.
- Compare like for like: `torch.add(a, b, out=c)`, not `a + b`, which allocates an output every call.

## 7. Turning a time into a verdict

The measured time can't be split into "memory time" and "compute time": the GPU overlaps them, so
ideally t ~ max(t_mem, t_compute). Instead compute both rates from the same time:

```
achieved GB/s    = minimum bytes / t     -> % of peak bandwidth
achieved FLOP/s  = FLOPs / t             -> % of peak FLOP/s
```

Whichever is near its peak is the limit. Vector add at 25M: 88% of bandwidth, 0.2% of compute.
If **neither** is near its peak, something else limits the kernel: latency, low occupancy, more
bytes than the minimum (re-reads, uncoalesced access), L1 / L2 / shared-memory bandwidth,
instruction overhead. That is the profiler's job (§10).

GB/s from minimum bytes is **effective bandwidth**; the actual DRAM traffic can be higher.

Use decimal units (GB = 1e9 bytes), like NVIDIA's spec sheets.

## 8. Problem size: the regimes

Vector add, N = 2^10 .. 2^28, two timings per size: cold L2 (`do_bench`) and warm L2 (200 calls
back to back, no flush, one event pair):

| regime | sizes (vector add) | what you see |
| --- | --- | --- |
| launch-bound | warm: up to ~512K (Triton), ~256K (torch) | time flat at the host launch cost (Triton ~20 us, torch ~6 us); GB/s measures Python |
| GPU fixed cost | cold: small N | ~4 us floor |
| L2 | warm: working set <= 4 MB | above DRAM's 448 GB/s: torch 519 GB/s at 262K (3.15 MB), copy 627 GB/s at 524K (4 MiB). Triton never shows it: it's launch-bound at every size that fits in L2 |
| ramp | cold: 32K .. 4M | GB/s climbs as the fixed cost is spread out and enough loads are in flight |
| DRAM | from ~4M elements (50 MB moved) | flat at ~392 GB/s; cold = warm |

"Memory-bound" is a property of the kernel **and** the size. Benchmark (and autotune) at a size
inside the regime you care about; LeetGPU's 25M (300 MB) is well inside the DRAM regime. On the
roofline, a size sweep moves the dot only vertically: the intensity doesn't change.

## 9. Configuration, occupancy and loads in flight

**Triton vs CUDA parameters.** CUDA's block size is threads; Triton's `BLOCK_SIZE` is elements per
program, with `num_warps x 32` threads, so elements per thread = `BLOCK_SIZE / (num_warps x 32)`.
Triton does not choose the grid or `num_warps` (defaults: 4 warps, 3 stages); `@triton.autotune`
picks among the configs you list (once per new `key` value, by benchmarking each). What Triton
automates is inside a program: coalesced and vectorized loads, shared-memory staging, pipelining,
Tensor Cores for `tl.dot`.

**Occupancy** = resident warps per SM / the maximum. On sm_86 the limits per SM are 1536 threads
(48 warps), 16 blocks, 64K registers and ~100 KB shared memory; the tightest wins, and blocks fit
only whole:

| threads per block | blocks per SM | warps | occupancy |
| --- | --- | --- | --- |
| 32 | 16 (block limit) | 16 | 33% |
| 64 | 16 (block limit) | 32 | 67% |
| 128 .. 512 | 1536 / size | 48 | 100% |
| 1024 | 1 | 32 | 67% |

**Why it matters: loads in flight.** A warp that has issued its loads waits several hundred ns for
DRAM; the SM hides that by running other warps, which issue more loads. Little's law:
bandwidth = bytes in flight / latency. ~400 GB/s x ~500 ns ~ 200 KB across the GPU, ~4 KB per SM.
One vector-add warp has 2 loads x 32 threads x 4 B = 256 B in flight: 16 warps ~ 4 KB (borderline),
48 warps ~ 12 KB.

**Occupancy is a means, not the goal.** The goal is enough *useful* bytes in flight, from more
warps or from more independent loads per thread (instruction-level parallelism). And the
occupancy metric counts warps that do no useful work.

Vector add at 25M (details: [`kernels/vector_add/README.md`](../kernels/vector_add/README.md)):
- 36 of 40 Triton configs, and CUDA 64..1024 threads per block, are within ~2%. Tuning a
  memory-bound kernel buys 1-2%.
- CUDA 1024 threads per block (67% occupancy) was the fastest: 754 us. Triton 4096 / 1 warp
  (128 elements per thread) runs at full speed.
- CUDA 32 threads per block: 16 warps per SM, 793 us (4% slower).
- Triton 128 / 16 warps: 512 threads for 128 elements, 3/4 idle, 902 us.
- Triton 256+ elements per thread with few warps: **register spills**, 1.5-2.8x slower. Spilled
  values go to local memory, which is in DRAM (cached in L1/L2): extra traffic. Check with
  `k = kernel[grid](...)`, then `k.n_regs`, `k.n_spills`; in CUDA, `nvcc -Xptxas -v`.

## 10. Profilers on the pod

- **Nsight Compute (`ncu`)** explains *why* a kernel sits where it does (Speed of Light, memory
  workload, roofline). On the current pod it fails with `ERR_NVGPUCTRPERM`:
  `/proc/driver/nvidia/params` has `RmProfilingAdminOnly: 1` and the container lacks
  `CAP_SYS_ADMIN`. Being root in the container is not enough; it needs the host driver setting
  (`NVreg_RestrictProfilingToAdminUsers=0`) or a container started with `--cap-add SYS_ADMIN`.
- **Nsight Systems (`nsys`)**: a timeline of launches, copies and gaps. Not installed on the pod.
  Its tracing does not need the restricted counters.
- Until then, timing (§4-§8) plus the arithmetic (§1, §2, §7) is enough to classify a kernel.

# relu: CUDA C++ learnings

What writing the ReLU kernels taught. Most of it is general and lives in
[`learnings/CUDA_CPP_GENERAL.md`](../../learnings/CUDA_CPP_GENERAL.md): `float4` and alignment
(§8), and branches vs predication, read from these kernels' SASS (§10). The problem, variants and
performance are in [`README.md`](README.md). This file keeps the two ReLU-specific lessons.

---

## 1. A memory-bound kernel can hide redundant work
*Added 2026-11-26, from the first version of [`relu_float4.cu`](relu_float4.cu) (fixed in `b11ee94`, 2026-11-15).*

The first `float4` version mixed units: the loop start, bound and check counted **floats**, but
the index was used on a **`float4`** pointer. Every `float4` was loaded and stored 4 times. (The
same unit mix-up as the first `float4` transpose, 2026-11-10.)

It still gave the right answer (writing the same correct value 4 times is harmless), and it still
ran at copy speed: 0.2186 ms before the fix, 0.2174 ms after. The repeated accesses most likely
hit the L1/L2 cache, so DRAM traffic stayed about the same (not measured: `ncu` is blocked on this pod). So **neither the correctness check nor the timing
caught it**. It was found by reading the index arithmetic.

Lesson: for `float4` code, check the units by hand. Write down what one step of `i` means (one
float or one `float4`?) for the start, the bound, the check and the index, and make them all
the same. A correct, fast result doesn't prove each element was touched once.

---

## 2. The comparison's direction decides what happens to NaN
*Added 2026-11-26, from the `float4` tail in [`relu_float4.cu`](relu_float4.cu) (2026-11-16).*

Every comparison with NaN is false. So two ways of writing ReLU that look the same differ on NaN:

| code | NaN input gives | matches torch? |
|---|---|---|
| `x < 0 ? 0.0f : x` | NaN (the test is false, so `x` is returned) | yes: `torch.relu(NaN)` is NaN |
| `x > 0 ? x : 0.0f` | 0 (the test is false, so 0 is returned) | no |

The first tail version used `> 0` while the main body used `< 0`, so a NaN became 0 in the last
1-3 floats and stayed NaN everywhere else. Use one form everywhere, the one that matches the
reference. (`tl.maximum(x, 0.0)` in Triton and `fmaxf` in CUDA have their own NaN rules: check
them before relying on them.)

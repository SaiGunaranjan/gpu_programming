# conv1d: Triton learnings

Problem-specific Triton lessons. Cross-problem ones go in
[`learnings/TRITON_GENERAL.md`](../../learnings/TRITON_GENERAL.md). The CUDA C++ conv1d lessons
(shared-memory tiles, static vs dynamic shared memory, barrier placement, early `return` vs
guarding each access) were learned on these kernels but are general, so they live in
[`learnings/CUDA_CPP_GENERAL.md`](../../learnings/CUDA_CPP_GENERAL.md) §2, §4.1 and §6.

---

## 1. When `assert_close` against torch fails, check torch too
*Added 2026-10-29, from `conv1d_triton_per_output.py` (input 1,500,000, filter 2047, RTX A4000)*

The script's `torch.testing.assert_close(output_triton, F.conv1d(...))` fails: 14.8% of elements
differ, by up to 4.3e-4 absolute. Comparing **both** against a float64 reference shows which side
is off:

| result | max abs error vs float64 |
| --- | --- |
| Triton kernel | **2.7e-5** |
| `F.conv1d` (cuDNN), TF32 allowed | 4.3e-4 |
| `F.conv1d` (cuDNN), `torch.backends.cudnn.allow_tf32 = False` | 4.3e-4 |

So torch is ~16x **less** accurate here, and TF32 isn't the reason (turning it off changes
nothing). cuDNN most likely picks a faster but less precise algorithm (e.g. FFT- or
Winograd-style) for a 2047-tap filter. Each output sums 2047 products of size ~1, so values reach
~±100, and a float32 `assert_close` (atol 1e-5, rtol 1.3e-6) is far too strict for that.

**Takeaways**
- A failed comparison against a library doesn't mean the kernel is wrong: the library is a
  float32 result too. Use a float64 reference (`F.conv1d(x.double(), w.double())`) as the
  arbiter.
- Scale the tolerance to the size of the sum: error grows with the number of terms and their
  magnitudes, e.g. `|err| / (max|x| * sum|w|)`. With that measure this kernel's worst case over 9
  shapes (1 to 100,000 inputs, 1 to 2048 taps) was 3.5e-8.

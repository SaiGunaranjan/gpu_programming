# matrix_transpose: Triton learnings

## 1. The 64 x 64 kernel runs at copy speed (2026-11-03, RTX A4000, 7000 x 6000)

| | time (us) | GB/s | % of D2D copy |
| --- | --- | --- | --- |
| D2D copy (`c.copy_(a)`, same bytes) | 869 | 387 | 100 |
| Triton, 64 x 64 tile, 4 warps | 869 | 387 | 100 |
| torch `output.copy_(input.T)` | 1861 | 181 | 47 |

A transpose moves exactly a copy's bytes (each element read once, written once), so the copy is
the right reference.

**Why Triton is at copy speed:** `tl.trans` changes which thread owns which element, so the
compiler inserts a layout conversion (`convert_layout` in the TTGIR) that goes **through shared
memory**: the PTX has 32 `st.shared` and 32 `ld.shared`. The tile is loaded with the input's
layout (coalesced, 128-bit `ld.global.v4`), exchanged through shared memory, and stored with the
output's layout. That is the shared-memory tiled transpose, written by the compiler.

The kernel uses 16,640 bytes of shared memory = 64 x **65** x 4: the tile is padded by one column,
the standard trick against shared-memory bank conflicts (see
[`learnings/MEMORY_ACCESS_GENERAL.md`](../../learnings/MEMORY_ACCESS_GENERAL.md), Q2–Q5).

How to look: `k = kernel[grid](...)`, then `k.asm['ttgir']`, `k.asm['ptx']`, `k.metadata.shared`,
`k.n_regs`, `k.n_spills`.

**torch** (`copy_` from a transposed view) reaches only 47% of the copy: it's torch's generic
strided copy, not a tiled transpose, so one side of the access isn't coalesced. Here the Triton
kernel beats the torch baseline by 2.1x.

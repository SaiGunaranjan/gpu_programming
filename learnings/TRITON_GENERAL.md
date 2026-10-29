# Triton: general learnings

Lessons that apply across problems: the programming model, `tl.constexpr`, masking, autotuning,
reading the generated PTX, and so on. Lessons specific to one problem live next to its code, in
`kernels/<op>/LEARNINGS_TRITON.md`. CUDA C++ counterpart: [`CUDA_CPP_GENERAL.md`](CUDA_CPP_GENERAL.md).

Grouped by topic, not by date. Each entry says **when** it was added and **which kernel** it came
from.

*Empty for now. The existing Triton notes are matmul-centred and arrive as
`kernels/matmul/LEARNINGS_TRITON.md`. Their general parts get split out into this file later.*

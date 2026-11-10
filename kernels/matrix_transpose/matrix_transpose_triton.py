import torch
import triton
import triton.language as tl

import numpy as np
import os
# os.environ["TRITON_INTERPRET"] = "1"
# torch.manual_seed(0)

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

# Search space for the autotuner (see learnings/TRITON_GENERAL.md §1.1).
# BLOCK_M and BLOCK_K are chosen independently, so rectangular tiles (e.g. 32x128, 128x32) are included.
# Tile size and num_warps are chosen together: the cross product is filtered so every config has
# MIN_ELEMS..MAX_ELEMS elements per thread. That drops the corners (huge tile with 1 warp -> register
# spills and minutes-long ptxas compiles; tiny tile with many warps -> idle threads).
TILE_SIZES = [16, 32, 64, 128]          # per tile dimension
NUM_WARPS = [1, 2, 4, 8, 16]            # threads per program = num_warps x 32
MIN_ELEMS, MAX_ELEMS = 4, 64            # elements per thread = BLOCK_M x BLOCK_K / (num_warps x 32)
CONFIGS = [triton.Config({'BLOCK_M': bm, 'BLOCK_K': bk}, num_warps=w)
           for bm in TILE_SIZES for bk in TILE_SIZES for w in NUM_WARPS
           if MIN_ELEMS <= bm * bk // (w * 32) <= MAX_ELEMS]


@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc,BLOCK_M: tl.constexpr,BLOCK_K: tl.constexpr):

    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)

    block_start_y = pid_y*BLOCK_M
    offsets_y_2d = block_start_y + tl.arange(0,BLOCK_M)

    block_start_x = pid_x*BLOCK_K
    offsets_x_2d = block_start_x + tl.arange(0,BLOCK_K)

    offsets_input = (offsets_y_2d*stride_ir)[:,None] + (offsets_x_2d*stride_ic)[None,:]
    mask_input = (offsets_y_2d[:,None] < rows) & (offsets_x_2d[None,:]<cols)

    tile_data = tl.load(input+offsets_input,mask=mask_input)

    offsets_output = (offsets_y_2d*stride_oc)[None,:] + (offsets_x_2d*stride_or)[:,None]
    mask_output = (offsets_y_2d[None,:] < rows) & (offsets_x_2d[:,None]<cols)

    tl.store(output+offsets_output,tl.trans(tile_data),mask=mask_output)

# The same kernel wrapped in the autotuner (decorator syntax applied by hand), so
# matrix_transpose_kernel itself stays usable with an explicit BLOCK_M, BLOCK_K and num_warps.
# An autotuned kernel can't be given BLOCK_SIZE at the call ("Conflicting meta-parameters").
# On the first call for each new rows x cols it times every config and keeps the fastest.
matrix_transpose_autotuned = triton.autotune(configs=CONFIGS,key=['rows','cols'])(matrix_transpose_kernel)

def solve_autotuned(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = input.stride(0), input.stride(1)
    stride_or, stride_oc = output.stride(0), output.stride(1)

    grid = lambda meta: (triton.cdiv(rows,meta["BLOCK_M"]), triton.cdiv(cols,meta["BLOCK_K"]))
    matrix_transpose_autotuned[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int, BLOCK_M: int = 64, BLOCK_K: int = 64, num_warps: int = 4):
    stride_ir, stride_ic = input.stride(0), input.stride(1)
    stride_or, stride_oc = output.stride(0), output.stride(1)

    # BLOCK_M = 64
    # BLOCK_K = 64
    grid = (triton.cdiv(rows,BLOCK_M), triton.cdiv(cols,BLOCK_K))
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc,
        BLOCK_M = BLOCK_M, BLOCK_K = BLOCK_K, num_warps = num_warps
    )


def bench(fn, warmup_ms, reps_ms):
    """Times fn with do_bench; returns (median, min, mean) in ms per call."""
    rawtimes_ms = triton.testing.do_bench(fn, warmup=warmup_ms, rep=reps_ms, return_mode='all')
    return np.median(rawtimes_ms), np.min(rawtimes_ms), np.mean(rawtimes_ms)


def gb_per_s(num_bytes, ms):
    """Bytes moved / seconds, in GB/s (decimal GB, like NVIDIA's spec sheets)."""
    return num_bytes / (ms * 1e-3) / 1e9


def print_row(name, times_ms, bw, copy_bw, peak_spec):
    median_ms, min_ms, mean_ms = times_ms
    print(f"{name:<16}: {median_ms*1e3:.1f} / {min_ms*1e3:.1f} / {mean_ms*1e3:.1f} us {bw:.1f} GB/s "
          f"( {(bw/peak_spec)*100:.1f}% of spec, {(bw/copy_bw)*100:.1f}% of D2D copy)")


if __name__ == '__main__':


    M = 7000
    K = 6000
    input = torch.randn(size=(M,K),device='cuda')
    output = torch.full((K,M), float('-inf'), device='cuda')
    output_dummy = torch.full((M,K), float('-inf'), device='cuda')

    solve_autotuned(input, output, M, K)

    output_torch = input.T

    # print(input)
    # print(output)
    torch.testing.assert_close(output_torch,output)
    print("OK  max abs diff =", (output_torch - output).abs().max().item())


    warmup_ms = 100
    reps_ms = 100

    copy_times = bench(lambda: output_dummy.copy_(input), warmup_ms, reps_ms)
    manual_times = bench(lambda: solve(input,output,M,K,BLOCK_M=64,BLOCK_K=64,num_warps=8), warmup_ms, reps_ms)
    autotuned_times = bench(lambda: solve_autotuned(input, output, M, K), warmup_ms, reps_ms)
    torch_times = bench(lambda: output.copy_(input.transpose(0,1)), warmup_ms, reps_ms)

    copy_bytes = 2 * M * K * input.element_size()
    kern_bytes = 2 * M * K * input.element_size()
    # peak_spec = 448 # In GBps, RTX A4000: 7001 MHz x 2 (GDDR6) x 32 B bus
    peak_spec = 2039; # A100 SXM: 1593 MHz x 2 (HBM2e) x 640 B bus. Update when the pod's GPU changes

    copy_bw = gb_per_s(copy_bytes, copy_times[0])     # medians
    manual_bw = gb_per_s(kern_bytes, manual_times[0])
    autotuned_bw = gb_per_s(kern_bytes, autotuned_times[0])
    torch_bw = gb_per_s(kern_bytes, torch_times[0])

    print(f"MxK={M}x{K}, warm-up {warmup_ms} ms, reps {reps_ms} ms (median / min / mean)")
    print_row("D2D copy", copy_times, copy_bw, copy_bw, peak_spec)
    print_row("triton 64x64/8", manual_times, manual_bw, copy_bw, peak_spec)
    print_row("triton autotuned", autotuned_times, autotuned_bw, copy_bw, peak_spec)
    print_row("torch", torch_times, torch_bw, copy_bw, peak_spec)

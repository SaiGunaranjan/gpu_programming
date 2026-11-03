import torch
import triton
import triton.language as tl

import numpy as np
# import os
# os.environ["TRITON_INTERPRET"] = "1"
# torch.manual_seed(0)

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

    solve(input, output, M, K)

    output_torch = input.T

    # print(input)
    # print(output)
    torch.testing.assert_close(output_torch,output)
    print("OK  max abs diff =", (output_torch - output).abs().max().item())


    warmup_ms = 100
    reps_ms = 100

    copy_times = bench(lambda: output_dummy.copy_(input), warmup_ms, reps_ms)
    manual_times = bench(lambda: solve(input,output,M,K), warmup_ms, reps_ms)
    torch_times = bench(lambda: output.copy_(input.transpose(0,1)), warmup_ms, reps_ms)

    copy_bytes = 2 * M * K * input.element_size()
    kern_bytes = 2 * M * K * input.element_size()
    peak_spec = 448 # In GBps, RTX A4000: 7001 MHz x 2 (GDDR6) x 32 B bus

    copy_bw = gb_per_s(copy_bytes, copy_times[0])     # medians
    manual_bw = gb_per_s(kern_bytes, manual_times[0])
    torch_bw = gb_per_s(kern_bytes, torch_times[0])

    print(f"MxK={M}x{K}, warm-up {warmup_ms} ms, reps {reps_ms} ms (median / min / mean)")
    print_row("D2D copy", copy_times, copy_bw, copy_bw, peak_spec)
    print_row("triton 64x64/4", manual_times, manual_bw, copy_bw, peak_spec)
    print_row("torch", torch_times, torch_bw, copy_bw, peak_spec)

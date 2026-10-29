import torch
import triton
import triton.language as tl

torch.manual_seed(0)

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
):
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)

    offsets_a = pid_y*stride_am + tl.arange(0,16384) # 16384
    mask_a = offsets_a < (pid_y*stride_am + N)

    # if pid_y == 1 and pid_x == 0:
    #     tl.device_print("pid_y 1 offsets_a = ", offsets_a)
    #     tl.device_print("pid_y 1 mask_a = ", mask_a)

    offsets_b = pid_x + stride_bn*tl.arange(0,16384) # 16384
    mask_b = offsets_b < N*K

    # if pid_y == 0 and pid_x == K-1:
    #     tl.device_print("pid_x 5 offsets_a = ", offsets_b)
    #     tl.device_print("pid_x 5 mask_a = ", mask_b)

    a_ = tl.load(a+offsets_a,mask=mask_a,other=0)
    b_ = tl.load(b+offsets_b,mask=mask_b,other=0)

    c_ = tl.sum(a_*b_)

    # if pid_x==0 and pid_y==0:
    #     tl.device_print("value",c_)

    offsets_c = pid_y*stride_cm + pid_x

    tl.store(c+offsets_c,c_)





# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = (M, K)
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
    )

M = 8192
N = 6144
K = 4096
a = torch.randint(high=10,size=(M,N),device='cuda').float()
b = torch.randint(high=10,size=(N,K),device='cuda').float()
c = torch.empty(M,K,device='cuda')
solve(a,b,c,M,N,K)

c_torch = torch.matmul(a,b)

torch.testing.assert_close(c,c_torch)
print("OK  max abs diff =", (c - c_torch).abs().max().item())


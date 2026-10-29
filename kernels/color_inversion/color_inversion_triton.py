import torch
import triton
import triton.language as tl

import os
# os.environ["TRITON_INTERPRET"] = "1"


@triton.jit
def invert_kernel(image, width, height, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE*4
    offsets = block_start + tl.arange(0,BLOCK_SIZE*4)
    mask = offsets < height*width*4

    tile_data = tl.load(image+offsets,mask=mask,other=0)

    temp = 255#tl.full((BLOCK_SIZE*4,),value=255, dtype=tl.uint8)
    mod_tile_data = temp - tile_data

    alpha_mask = (offsets % 4) == 3
    mod_tile_data =tl.where(alpha_mask,tile_data,mod_tile_data)

    tl.store(image+offsets, mod_tile_data, mask=mask)



# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](image, width, height, BLOCK_SIZE)


image = torch.tensor([255, 0, 128, 255, 0, 255, 0, 255],device='cuda',dtype=torch.uint8)
width=1; height=2

temp = 255*torch.ones(height*width*4,device='cuda',dtype=torch.uint8)
torch_image = temp - image
torch_image[3::4] = image[3::4]

solve(image,width,height)

print(torch_image)
print(image)

import torch
import triton
import triton.language as tl

@triton.jit
def vector_scale_kernel(x, y, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # block id

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # list of data indices in the block

    mask = offsets < n_elements

    x_block = tl.load(x + offsets, mask=mask)
    y_block = alpha * x_block  # scalar broadcasting: alpha is broadcasted to all elements
    tl.store(y + offsets, y_block, mask=mask)

def solve(x: torch.Tensor, alpha: float, y: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_scale_kernel[grid](x, y, alpha, N, BLOCK_SIZE)

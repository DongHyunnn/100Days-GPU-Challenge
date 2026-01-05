import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0) # block id

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # list of data indices in the block

    mask = offsets < n_elements

    a_block = tl.load(a + offsets, mask=mask)
    b_block = tl.load(b + offsets, mask=mask)
    c_block = a_block + b_block
    tl.store(c + offsets, c_block, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N : int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
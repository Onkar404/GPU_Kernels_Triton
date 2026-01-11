import triton
import triton.language as tl 
import torch

@triton.jit 
def add_kernel(x_ptr,y_ptr,out_ptr,n_ele,BLOCK_SIZE:tl.constexpr):
  pid=tl.program_id(0)
  offset=pid*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)
  mask=offset<n_ele
  x=tl.load(x_ptr+offset,mask=mask)
  y=tl.load(y_ptr+offset,mask=mask)
  out=x+y
  tl.store(out_ptr+offset,out,mask=mask)


def add(x:torch.tensor,y:torch.tensor):
  out=torch.empty_like(x)
  n_ele=out.numel()
  grid=lambda meta: (triton.cdiv(n_ele,meta['BLOCK_SIZE']),)
  add_kernel[grid](x,y,out,n_ele,BLOCK_SIZE=1024)
  return out
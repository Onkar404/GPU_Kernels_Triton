
import triton 
import torch
import triton.language as tl


@triton.jit 
def softmax_kernel(x_ptr,y_ptr,n_rows,n_cols,BLOCK_SIZE:tl.constexpr):
  row_start=tl.program_id(0)
  row_step=tl.num_programs(0)
  cols=tl.arange(0,BLOCK_SIZE)

  for row_idx in range(row_start,n_rows,row_step):
    row_start_ptr=x_ptr+row_idx*n_cols
    row=tl.load(row_start_ptr+cols,mask=n_cols>cols,other=-float('inf'))
    row_max=tl.max(row,0)
    row_exp=tl.exp(row-row_max)
    sum_row_exp=tl.sum(row_exp)
    y=row_exp/sum_row_exp
    yptr=y_ptr+row_idx*n_cols
    tl.store(yptr+cols,y,mask=n_cols>cols)


def softmax(x):
  y=torch.empty_like(x)
  n_rows,n_cols=y.shape
  grid=(n_rows,)
  num_stages=4
  BLOCK_SIZE = triton.next_power_of_2(n_cols)
  softmax_kernel[grid](x,y,n_rows,n_cols,BLOCK_SIZE,num_stages)
  return y 
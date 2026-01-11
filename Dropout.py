import triton 
import triton.language as tl 
import torch 


@triton.jit 
def Dropout_kernel(x_ptr,y_ptr,p,seed,n_rows,n_cols,row_stride,BLOCK_SIZE:tl.constexpr):
  row_start=tl.program_id(0)
  row_step=tl.num_programs(0)
  col_offset=tl.arange(0,BLOCK_SIZE)

  for row_idx in range(row_start,n_rows,row_step):
    row_start_ptr=x_ptr+row_idx*row_stride
    mask=col_offset<n_cols
    row=tl.load(row_start_ptr+col_offset,mask=mask,other=0.0)
    random = tl.rand(seed,row_idx*row_stride+col_offset)
    keep=random>p
    row=tl.where(keep,row/(1-p),0.0)
    yptr=y_ptr+row_idx*row_stride
    tl.store(yptr+col_offset,row,mask=mask)



def Dropout(x,p,seed,training=True):
  if not training:
    return x
  if p==0:
    return x
  elif p==1:
    return torch.zeros_like(x)
  else:
    y=torch.empty_like(x)
    n_rows,n_cols=x.shape
    row_stride=x.stride(0)
    grid=(n_rows,1)
    BLOCK_SIZE=triton.next_power_of_2(n_cols)
    Dropout_kernel[grid](x,y,p,seed,n_rows,n_cols,row_stride,BLOCK_SIZE)
    return y

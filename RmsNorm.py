import triton
import triton.language as tl
import torch

@triton.jit 
def RMSNorm(x_ptr,y_ptr,W,R,eps,n_cols,row_stride,BLOCK_SIZE:tl.constexpr):
  row=tl.program_id(0)
  xptr=x_ptr+row*row_stride
  offsets=tl.arange(0,BLOCK_SIZE)
  mask=offsets<n_cols
  x = tl.load(xptr + offsets,mask=mask,other=0.0).to(tl.float32)
  w = tl.load(W + offsets,mask=mask,other=0.0).to(tl.float32)
  x_sqr=x*x
  rms=tl.sqrt( (tl.sum(x_sqr,axis=0) / n_cols) + eps)
  y=((x/rms)*w).to(w.dtype)
  tl.store(R+row,rms)
  yptr=y_ptr+row*row_stride
  tl.store(yptr+offsets,y,mask=mask)


  
 '''TODO  Back pass'''

@triton.jit 
def rms_bwd_dw_dx(x_ptr,dy_ptr,R,W,DX,DW_par,n_cols,stride,BLOCK_SIZE:tl.constexpr):
  row=tl.program_id(0)
  row_stride=stride
 
  xptr=x_ptr+row*row_stride
  dyptr=dy_ptr+row*row_stride
  offsets= tl.arange(0,BLOCK_SIZE)
  mask=offsets<n_cols

  x=tl.load(xptr+offsets,mask=mask,other=0.0).to(tl.float32)
  w=tl.load(W+offsets,mask=mask,other=0.0).to(tl.float32)
  dy=tl.load(dyptr+offsets,mask=mask,other=0.0).to(tl.float32)
  rms=tl.load(R+row)

  dyw=dy*w
  s=tl.sum(dyw*x,axis=0)
  dx=(dyw/rms)-(x*s)/(n_cols*rms*rms*rms)
  dw_par=(dy*(x/rms))

  tl.store(DX+row*stride+offsets,dx,mask=mask)
  tl.store(DW_par+row*n_cols+offsets,dw_par,mask=mask)


@triton.jit
def reduce_dw_par(DW_par, DW,n_rows, n_cols,BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(0)

  offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_cols
  acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
  for r in range(n_rows):
      dw = tl.load(DW_par + r * n_cols + offsets,mask=mask, other=0.0).to(tl.float32)
      acc += dw
  tl.store(DW + offsets, acc, mask=mask)


class RMS_Norm(torch.autograd.Function):
  @staticmethod
  def forward(ctx,x,w,eps):
    shape=x.shape
    x=x.reshape(-1,shape[-1])
    y=torch.empty_like(x).to('cuda')
    n_rows,n_cols=x.shape
    BLOCK_SIZE=triton.next_power_of_2(n_cols)
    grid=(n_rows,)
    R=torch.empty((n_rows,)).to('cuda')
    RMSNorm[grid](x,y,w,R,eps,n_cols,x.stride(0),BLOCK_SIZE)
    ctx.BLOCK_SIZE=BLOCK_SIZE
    ctx.eps=eps
    ctx.save_for_backward(x,w,R)
    return y.view(*shape)

  @staticmethod
  def backward(ctx,dy):
    x,w,R=ctx.saved_tensors
    BLOCK_SIZE=ctx.BLOCK_SIZE
    eps=ctx.eps
    shape=dy.shape
    dy=dy.reshape(-1,shape[-1])
    n_rows,n_cols=x.shape
    stride=x.stride(0)
    DX=torch.empty_like(x).to('cuda')
    DW_par=torch.empty_like(x).to('cuda')
    grid=(n_rows,)
    rms_bwd_dw_dx[grid](x,dy,R,w,DX,DW_par,n_cols,stride,BLOCK_SIZE)
    DW=torch.empty_like(w).to('cuda')
    reduce_dw_par[(triton.cdiv(n_cols, BLOCK_SIZE),)](DW_par,DW,n_rows,n_cols,BLOCK_SIZE=BLOCK_SIZE)
    return DX.view(*shape),DW,None
      

def triton_rms_norm(x, w, eps=1e-5):
   return RMS_Norm.apply(x, w, eps)
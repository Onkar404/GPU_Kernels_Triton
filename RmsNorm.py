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
  
 '''TODO  Back pass'''

def triton_rms_norm(x, w, eps=1e-5):
   return RMS_Norm.apply(x, w, eps)
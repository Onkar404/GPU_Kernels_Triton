import triton
import torch
import triton.language as tl

@triton.jit
def layer_norm_kernel(x_ptr,y_ptr,eps,W,B,RSTD,MEAN,row_stride,n_cols,BLOCK_SIZE:tl.constexpr):
  row=tl.program_id(0)
  xptr=x_ptr+row*row_stride
  yptr=y_ptr+row*row_stride

  offset=tl.arange(0,BLOCK_SIZE)

  x=tl.load(xptr+offset,mask=offset<n_cols,other=0.0)
  w=tl.load(W+offset,mask=offset<n_cols,other=0.0)
  b=tl.load(B+offset,mask=offset<n_cols,other=0.0)

  mean=tl.sum(x,axis=0)/n_cols
  x_minus_mean=x-mean

  var=tl.sum(x_minus_mean*x_minus_mean,axis=0)/n_cols
  rstd= 1 / tl.sqrt(var + eps)

  tl.store(RSTD+row,rstd)
  tl.store(MEAN+row,mean)

  y=x_minus_mean*(rstd)*w + b
  tl.store(yptr+offset,y,mask=offset<n_cols)


@triton.jit
def ln_bwd_dx_dwdb_partial(
    DX, DY, X, W, Mean, Rstd,
    DW_partial, DB_partial,
    stride, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    X += row * stride
    DY += row * stride
    DX += row * stride
    DW_partial += row * N
    DB_partial += row * N

    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)

    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    xhat = (x - mean) * rstd
    wdy = dy * w

    c1 = tl.sum(wdy * xhat, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N

    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + cols, dx, mask=mask)
    
    db_par=(dy * xhat).to(w.dtype)
    dw_par=dy.to(w.dtype)

    # partial gradients (no reduction yet)
    tl.store(DW_partial + cols,db_par, mask=mask)
    tl.store(DB_partial + cols, dw_par , mask=mask)


@triton.jit
def ln_bwd_reduce_dwdb(
    DW_partial, DB_partial,
    DW, DB,
    num_rows, N,
    BLOCK_SIZE: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dw = tl.zeros([BLOCK_SIZE], tl.float32)
    db = tl.zeros([BLOCK_SIZE], tl.float32)

    for r in range(num_rows):
        dw += tl.load(DW_partial + r * N + cols, mask=mask, other=0)
        db += tl.load(DB_partial + r * N + cols, mask=mask, other=0)

    tl.store(DW + cols, dw, mask=mask)
    tl.store(DB + cols, db, mask=mask)


class Layer_Norm(torch.autograd.Function):
  @staticmethod 
  def forward(ctx,x,w,b,eps):
    shape=x.shape
    x=x.reshape(-1,shape[-1])
    n_rows,n_cols=x.shape
    device=x.device
    y=torch.empty_like(x).to(device)
    rstd=torch.empty((n_rows,)).to(device)
    mean=torch.empty((n_rows,)).to(device)
    BLOCK_SIZE=triton.next_power_of_2(n_cols)
    grid=(n_rows,)
    layer_norm_kernel[grid](x,y,eps,w,b,rstd,mean,x.stride(0),n_cols,BLOCK_SIZE)
    ctx.eps = eps
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.save_for_backward(x, w, b, rstd, mean)
    return y.view(*shape)

  @staticmethod
  def backward(ctx, dy):
    shape=dy.shape
    dy=dy.reshape(-1,shape[-1])
    
    x,w,b,rstd,mean=ctx.saved_tensors
    eps=ctx.eps
    BLOCK_SIZE=ctx.BLOCK_SIZE
    n_rows,n_cols=x.shape
    device=x.device

    DX = torch.empty_like(x).to(device)
    DW_partial = torch.empty((n_rows, n_cols), device=x.device)
    DB_partial = torch.empty((n_rows, n_cols), device=x.device)

    ln_bwd_dx_dwdb_partial[(n_rows,)](
        DX, dy, x, w, mean, rstd,
        DW_partial, DB_partial,
        X.stride(0), n_cols,
        BLOCK_SIZE=ctx.BLOCK_SIZE
    )

    DW = torch.empty_like(w).to(device)
    DB = torch.empty_like(w).to(device)

    grid = (triton.cdiv(n_cols, ctx.BLOCK_SIZE),)
    ln_bwd_reduce_dwdb[grid](
        DW_partial, DB_partial,
        DW, DB,
        n_rows, n_cols,
        BLOCK_SIZE=ctx.BLOCK_SIZE
    )

    return DX.view(*shape), DW, DB, None



def triton_layer_norm(x, w, b, eps=1e-5):
    return Layer_Norm.apply(x, w, b, eps)
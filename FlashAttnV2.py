import triton
import triton.language as tl
import torch


@triton.jit 
def flash_attn_kernel(q_ptr,k_ptr,v_ptr,o_ptr,Mi,Li,H,stride_b,stride_h,stride_q,stride_k,stride_v,N_CTX:tl.constexpr,HEAD_DIM:tl.constexpr,BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr):
  pid=tl.program_id(0)
  bhid=tl.program_id(1)

  b=bhid // H
  h=bhid % H

  offs_m=pid*BLOCK_M+tl.arange(0,BLOCK_M)
  offs_d=tl.arange(0,HEAD_DIM)
  qptr=q_ptr+b*stride_b+h*stride_h+offs_m[:,None]*stride_q+offs_d[None,:]

  mi=tl.zeros([BLOCK_M],dtype=tl.float32) - float('inf')
  li=tl.zeros([BLOCK_M],dtype=tl.float32)
  acc=tl.zeros([BLOCK_M,HEAD_DIM],dtype=tl.float32)
  mask_q=(offs_m[:,None]<N_CTX) & (offs_d[None,:]<HEAD_DIM)

  for i in range(0,N_CTX,BLOCK_N):

    offs_n=i+tl.arange(0,BLOCK_N)
    mask_kv=(offs_n[:,None]<N_CTX) & (offs_d[None,:]<HEAD_DIM)

    kptr=k_ptr+b*stride_b+h*stride_h+offs_n[:,None]*stride_k+offs_d[None,:]
    vptr=v_ptr+b*stride_b+h*stride_h+offs_n[:,None]*stride_v+offs_d[None,:]

    q=tl.load(qptr,mask=mask_q)
    v=tl.load(vptr,mask=mask_kv)
    k=tl.load(kptr,mask=mask_kv)

    scale = 1.0 / tl.sqrt(tl.full([], HEAD_DIM, tl.float32))

    attn=tl.dot(q,tl.trans(k))*scale
    caus_mask=offs_n[None,:] <= offs_m[:,None]
    attn=tl.where(caus_mask,attn,-float('inf'))

    attn_max=tl.max(attn,axis=1)
    new_max=tl.maximum(attn_max,mi)

    attn_exp = tl.where(caus_mask,tl.exp(attn - new_max[:, None]),0.0)

    alpha=tl.exp(mi-new_max)
    li=li*alpha+tl.sum(attn_exp,axis=1)
    acc=acc*alpha[:,None] + tl.dot(attn_exp,v)

    mi=new_max

  mask_m = offs_m < N_CTX
  tl.store(Mi+offs_m,mi,mask=mask_m)
  tl.store(Li+offs_m,li,mask=mask_m)

  mask_o = (offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
  o=acc/(li[:,None]+ 1e-9)
  optr=o_ptr+b*stride_b+h*stride_h+offs_m[:,None]*stride_q+offs_d[None,:]
  tl.store(optr,o,mask=mask_o)
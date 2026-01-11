
import triton 
import triton.language as tl
import torch


@triton.jit 
def matmul_kernel(a_ptr,b_ptr,c_ptr,
           M,N,K,
           stride_am,stride_ak,
           stride_bk,stride_bn,
           stride_cm,stride_cn,
           BLOCK_SIZE_M:tl.constexpr,BLOCK_SIZE_N:tl.constexpr,BLOCK_SIZE_K:tl.constexpr,
           GROUP_SIZE_M:tl.constexpr ):
  
  pid=tl.program_id(0)
  num_pid_m=tl.cdiv(M,BLOCK_SIZE_M)
  num_pid_n=tl.cdiv(N,BLOCK_SIZE_N)
  num_pid_group=num_pid_n*GROUP_SIZE_M
  group_id=pid//num_pid_group
  first_pid_m=group_id*GROUP_SIZE_M
  group_size_m=min(GROUP_SIZE_M,num_pid_m-first_pid_m)
  pid_m=first_pid_m+((pid%num_pid_group)%group_size_m)
  pid_n=(pid%num_pid_group)//group_size_m

  offs_m=pid_m*BLOCK_SIZE_M+tl.arange(0,BLOCK_SIZE_M)
  offs_n=pid_n*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N)
  offs_k=tl.arange(0,BLOCK_SIZE_K)

  aptr=a_ptr+offs_m[:,None]*stride_am+offs_k[None,:]*stride_ak
  bptr=b_ptr+offs_k[:,None]*stride_bk+offs_n[None,:]*stride_bn

  acc=tl.zeros([BLOCK_SIZE_M,BLOCK_SIZE_N],dtype=tl.float32)

  for k in range(0,tl.cdiv(K,BLOCK_SIZE_K)):
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
    b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)
    a=tl.load(aptr,mask=a_mask, other=0.0)
    b=tl.load(bptr,mask=b_mask, other=0.0)
    acc=tl.dot(a,b,acc)

    aptr+=stride_ak*BLOCK_SIZE_K
    bptr+=stride_bk*BLOCK_SIZE_K
  c=acc.to(tl.float16)
  cptr=c_ptr+offs_m[:,None]*stride_cm+offs_n[None,:]*stride_cn
  c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
  tl.store(cptr,c,mask=c_mask)


def matmul(a,b):
  assert a.shape[1] == b.shape[0] 
  assert a.is_contiguous()
  M, K = a.shape
  K, N = b.shape
  
  c = torch.empty((M, N), device=a.device, dtype=torch.float16)


  grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
  matmul_kernel[grid](
      a, b, c,  
      M, N, K,  
      a.stride(0), a.stride(1),  
      b.stride(0), b.stride(1),  
      c.stride(0), c.stride(1),  
      BLOCK_SIZE_M=64,
      BLOCK_SIZE_N=64,
      BLOCK_SIZE_K=64,
      GROUP_SIZE_M=8,
       
  )
  return c


import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch as th
import torch.linalg as la
from torch.nn import Parameter
import torch.linalg as linalg

class Model(torch.nn.Module):

    def __init__(self, hidden_size=8, num_of_heads=8, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_of_heads = num_of_heads
        self.sqrt_hd = math.sqrt(hidden_size)
        self.q_linear = torch.nn.Linear(hidden_size, hidden_size * num_of_heads)
        self.k_linear = torch.nn.Linear(hidden_size, hidden_size * num_of_heads)
        self.v_linear = torch.nn.Linear(hidden_size, hidden_size * num_of_heads)
        self.o_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.attn_mask = torch.zeros((num_of_heads, 1, 1, hidden_size), dtype=torch.float16)

    def attention(self, q, k, v):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q = torch.cat(torch.split(q, self.hidden_size, -1), 0)
        k = torch.cat(torch.split(k, self.hidden_size, -1), 0)
        v = torch.cat(torch.split(v, self.hidden_size, -1), 0)
        q = q.view(self.num_of_heads, -1, 1, self.hidden_size).transpose(-2, -1)
        k = k.view(self.num_of_heads, -1, 1, self.hidden_size).transpose(-2, -1)
        v = v.view(self.num_of_heads, -1, 1, self.hidden_size).transpose(-2, -1)
        qk = q @ k / self.sqrt_hd
        qk = qk + self.attn_mask
        qk_softmax = torch.softmax(qk, dim=-1)
        qkv = qk_softmax @ v
        qkv = qkv.transpose(0, 1).contiguous()
        qkv = qkv.view(qkv.size(0), qkv.size(2), -1)
        m = self.o_linear(qkv)
        return m

    def forward(self, query, key, value):
        output = self.attention(query, key, value)
        return output


func = Model().to('cuda:0')


q = torch.randn(1, 8, 32)

k = torch.randn(1, 8, 64)

v = torch.randn(1, 8, 64)

test_inputs = [q, k, v]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (8x32 and 8x64)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 32)), Parameter(FakeTensor(..., device='cuda:0', size=(64, 8), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(64,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [8, 32] X [8, 64].

from user code:
   File "<string>", line 46, in forward
  File "<string>", line 27, in attention
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
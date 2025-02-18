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

    def __init__(self, key_dim, num_heads):
        super().__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.q_proj = torch.nn.Linear(key_dim, num_heads * key_dim)
        self.k_proj = torch.nn.Linear(key_dim, num_heads * key_dim)
        self.v_proj = torch.nn.Linear(key_dim, num_heads * key_dim)
        self.fc = torch.nn.Linear(num_heads * key_dim, key_dim)

    def forward(self, x1, x2):
        q = self.q_proj(x1)
        k = self.k_proj(x2)
        v = self.v_proj(x2)
        d_k = self.key_dim
        scale = 1 / math.sqrt(d_k)
        q = q * scale
        x = torch.matmul(q, k.transpose(-2, -1))
        x = x.softmax(dim=-1)
        return self.fc(x.matmul(v))


key_dim = 1
num_heads = 1

func = Model(key_dim, num_heads).to('cuda:0')


key = torch.randn(1, 1, 3, 3)

query = torch.randn(1, 2, 3, 3)

test_inputs = [key, query]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (3x3 and 1x1)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 1, 3, 3)), Parameter(FakeTensor(..., device='cuda:0', size=(1, 1), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(1,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [3, 3] X [1, 1].

from user code:
   File "<string>", line 25, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
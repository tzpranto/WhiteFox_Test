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

    def __init__(self):
        super().__init__()
        self.nhead = 2
        self.head_dim = 32
        self.n_queries = 8
        self.n_keys = 4
        self.n_values = 4
        self.qkv = torch.nn.Linear(32, 32 * 3)
        self.out = torch.nn.Linear(32, 32)
        self.scale = self.head_dim ** (-0.5)

    def forward(self, x1, x2):
        qkv = self.qkv(x1)
        (q, k, v) = torch.chunk(qkv, chunks=3, dim=-1)
        q = q.unsqueeze(1).repeat(1, self.n_queries, 1)
        k = k.unsqueeze(1).repeat(1, self.n_keys, 1)
        v = v.unsqueeze(1).repeat(1, self.n_values, 1)
        att = torch.bmm(q.permute(0, 2, 1), k)
        inv_scale = 1.0 / np.sqrt(self.head_dim)
        scaled_att = inv_scale * att
        weights = scaled_att.softmax(dim=-1)
        o = torch.bmm(weights, v)
        o = o.flatten(0, 1)
        o = self.out(o)
        return o


func = Model().to('cuda:0')


x1 = torch.randn(2, 32, 3)

x2 = torch.randn(2, 32, 3)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (64x3 and 32x96)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(2, 32, 3)), Parameter(FakeTensor(..., device='cuda:0', size=(96, 32), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(96,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [64, 3] X [32, 96].

from user code:
   File "<string>", line 27, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
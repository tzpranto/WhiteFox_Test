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

    def __init__(self, key_dim, query_dim, value_dim):
        super().__init__()
        self.key_dim = key_dim
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.wq = torch.nn.Linear(query_dim, query_dim, bias=False)
        self.wk = torch.nn.Linear(key_dim, key_dim, bias=False)
        self.wv = torch.nn.Linear(value_dim, value_dim)
        self.dropout = torch.nn.Dropout2d(p=0.0)

    def forward(self, x1, x2, x3):
        v1 = self.wq(x1)
        v2 = self.wk(x2)
        v3 = torch.matmul(v1, v2.transpose(-2, -1)) / math.sqrt(v1.shape[-1])
        v4 = v3.softmax(dim=-1)
        v5 = self.dropout(x3)
        v6 = self.wv(v5)
        v7 = torch.matmul(v4, v6)
        return v7


key_dim = 1
query_dim = 1
value_dim = 1
func = Model(64, 512, 512).to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

x2 = torch.randn(1, 3, 512)

x3 = torch.randn(1, 3, 512)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (192x64 and 512x512)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)), Parameter(FakeTensor(..., device='cuda:0', size=(512, 512), requires_grad=True)), None), **{}):
a and b must have same reduction dim, but got [192, 64] X [512, 512].

from user code:
   File "<string>", line 26, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
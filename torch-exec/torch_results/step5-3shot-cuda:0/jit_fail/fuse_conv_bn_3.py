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
        self.f1 = torch.nn.Linear(4, 6, bias=False)
        torch.manual_seed(3)
        self.f1.weight = torch.nn.Parameter(torch.zeros(self.f1.weight.shape[0], self.f1.weight.shape[1]))
        bn = torch.nn.BatchNorm1d(6)
        bn.running_mean = torch.arange(6, dtype=torch.float)
        bn.running_var = torch.arange(6, dtype=torch.float) * 2 + 1
        self.f2 = torch.nn.Sequential(self.f1, bn)

    def forward(self, x):
        x1 = self.f1(x)
        x2 = x1[0:6]
        x3 = x2.mean()
        p = torch.mul(x3, 100)
        return p



func = Model().to('cuda:0')


x = torch.randn(64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x64 and 4x6)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(64,)), Parameter(FakeTensor(..., device='cuda:0', size=(6, 4), requires_grad=True)), None), **{}):
a and b must have same reduction dim, but got [1, 64] X [4, 6].

from user code:
   File "<string>", line 26, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
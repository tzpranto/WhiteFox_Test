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

class Model2(torch.nn.Module):

    def __init__(self, dim1):
        super().__init__()
        self.dim1 = dim1
        self.fc1 = torch.nn.Linear(10, dim1)
        self.fc2 = torch.nn.Linear(10, dim1)
        self.fc3 = torch.nn.Linear(dim1, 10)
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.bn2 = torch.nn.BatchNorm1d(10)

    def forward(self, x1):
        v1 = self.bn1(x1)
        v2 = self.bn2(x1)
        v3 = torch.cat([v1, v2], 1)
        v4 = self.fc1(v3)
        v5 = self.fc2(v3)
        v6 = torch.addmm(v4, v5, torch.eye(self.dim1))
        v7 = self.fc3(v6)
        return v7


dim1 = 1
func = Model2(10).to('cpu')


x1 = torch.randn(1, 10)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x20 and 10x10)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 20)), Parameter(FakeTensor(..., size=(10, 10), requires_grad=True)), Parameter(FakeTensor(..., size=(10,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 20] X [10, 10].

from user code:
   File "<string>", line 28, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
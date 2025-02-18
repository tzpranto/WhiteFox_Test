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
        self._linear1 = torch.nn.Linear(5, 1, bias=True)
        self._linear2 = torch.nn.Linear(2, 1, bias=True)

    def forward(self, x0):
        r0 = self._linear1(x0).reshape(-1)
        r1 = torch.unsqueeze(r0, 1)
        v0 = torch.cat((r0, r1), 0)
        v1 = self._linear1(v0)
        return v1



func = Model().to('cuda:0')


x1 = torch.randn(1, 1, 5, 4)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (5x4 and 5x1)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 1, 5, 4)), Parameter(FakeTensor(..., device='cuda:0', size=(1, 5), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(1,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [5, 4] X [5, 1].

from user code:
   File "<string>", line 21, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
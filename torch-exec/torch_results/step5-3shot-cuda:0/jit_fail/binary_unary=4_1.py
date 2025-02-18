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
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        v2 = v1 + other if other is not None else v1
        v3 = F.relu(v2)
        return v3



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

x3 = torch.randn(1, 3, 64, 64)

test_inputs = [x1, x3]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (192x64 and 3x8)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)), Parameter(FakeTensor(..., device='cuda:0', size=(8, 3), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(8,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [192, 64] X [3, 8].

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
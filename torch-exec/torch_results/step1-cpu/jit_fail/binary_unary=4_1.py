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

    def __init__(self, in_out):
        super().__init__()
        self.linear = torch.nn.Linear(in_out, in_out)

    def forward(self, x, other=1):
        v1 = self.linear(x)
        v2 = v1 + other
        v2 = v2.relu()
        return v2


in_out = 1
func = Model(8).to('cpu')


x = torch.randn(1, 8, 3, 3)

y = torch.randn(1, 8, 1, 1)

test_inputs = [x, y]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (24x3 and 8x8)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 8, 3, 3)), Parameter(FakeTensor(..., size=(8, 8), requires_grad=True)), Parameter(FakeTensor(..., size=(8,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [24, 3] X [8, 8].

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
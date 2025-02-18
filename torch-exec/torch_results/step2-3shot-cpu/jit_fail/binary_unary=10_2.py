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
        self.linear = torch.nn.Linear(64 * 64 * 3, 1)
        self.other = torch.nn.Parameter(torch.randn(1))

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = relu(v2)
        return v3


func = Model().to('cpu')


x1 = torch.randn(8, 64, 64, 3)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (32768x3 and 12288x1)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(8, 64, 64, 3)), Parameter(FakeTensor(..., size=(1, 12288), requires_grad=True)), Parameter(FakeTensor(..., size=(1,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [32768, 3] X [12288, 1].

from user code:
   File "<string>", line 21, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
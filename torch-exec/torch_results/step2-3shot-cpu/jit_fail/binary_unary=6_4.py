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
        self.linear = torch.nn.Linear(16, 32, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 0.1
        v3 = self.relu(v2)
        return v3


func = Model().to('cpu')


x1 = torch.randn(1, 8)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x8 and 16x32)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 8)), Parameter(FakeTensor(..., size=(32, 16), requires_grad=True)), None), **{}):
a and b must have same reduction dim, but got [1, 8] X [16, 32].

from user code:
   File "<string>", line 21, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
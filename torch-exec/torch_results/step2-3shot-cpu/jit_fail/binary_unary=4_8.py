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
        self.linear = torch.nn.Linear(8, 6)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3


func = Model().to('cpu')


other = torch.randn(1, 6)

test_inputs = [other]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x6 and 8x6)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 6)), Parameter(FakeTensor(..., size=(6, 8), requires_grad=True)), Parameter(FakeTensor(..., size=(6,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 6] X [8, 6].

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
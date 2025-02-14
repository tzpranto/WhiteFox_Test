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

    def forward(self, v, x):
        return v * self.linear(x) + self.other


func = Model().to('cpu')


v = torch.ones(1)

x = torch.ones(1, 3, 64, 64)

test_inputs = [v, x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (192x64 and 16x32)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 3, 64, 64)), Parameter(FakeTensor(..., size=(32, 16), requires_grad=True)), None), **{}):
a and b must have same reduction dim, but got [192, 64] X [16, 32].

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
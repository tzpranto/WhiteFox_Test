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

    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
        self.other = other

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3


other = 1
func = Model(torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8])).to('cuda:0')


x1 = torch.randn(1, 8, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (512x64 and 8x8)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 64, 64)), Parameter(FakeTensor(..., device='cuda:0', size=(8, 8), requires_grad=True)), None), **{}):
a and b must have same reduction dim, but got [512, 64] X [8, 8].

from user code:
   File "<string>", line 21, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
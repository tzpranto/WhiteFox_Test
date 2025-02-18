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
        self.linear = torch.nn.Linear(8, 4)

    def forward(self, x1, min_value=-1.0, max_value=-1.0):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        ret = torch.clamp_max(v2, max_value)
        return ret


func = Model().to('cuda:0')


x1 = torch.randn(2, 4, 2, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (16x2 and 8x4)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(2, 4, 2, 2)), Parameter(FakeTensor(..., device='cuda:0', size=(4, 8), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(4,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [16, 2] X [8, 4].

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
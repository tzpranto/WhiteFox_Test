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
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(128, 64)

    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.linear(v1.view(v1.shape[0], -1))
        v3 = torch.addmm(x2, v2, v1)
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

x2 = torch.randn(1, 64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x34848 and 128x64)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 34848)), Parameter(FakeTensor(..., device='cuda:0', size=(64, 128), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(64,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 34848] X [128, 64].

from user code:
   File "<string>", line 22, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.linear1 = torch.nn.Linear(960, 800)
        self.linear2 = torch.nn.Linear(800, 8)

    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.flatten(v3, start_dim=1)
        v5 = self.linear1(v4)
        v6 = self.linear2(v5)
        return v6


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x8192 and 960x800)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 8192)), Parameter(FakeTensor(..., size=(800, 960), requires_grad=True)), Parameter(FakeTensor(..., size=(800,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 8192] X [960, 800].

from user code:
   File "<string>", line 28, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
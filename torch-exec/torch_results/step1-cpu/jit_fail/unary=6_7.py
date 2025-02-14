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
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 + 3
        v4 = v3.clamp(0, 6)
        v5 = v1 * v4
        v6 = torch.div(v5, 6)
        return v6


func = Model().to('cpu')


x = torch.randn(1, 16, 32, 32)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [64, 32, 3, 3], expected input[1, 16, 32, 32] to have 32 channels, but got 16 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f1e3105f1c0>(*(FakeTensor(..., size=(1, 16, 32, 32)), Parameter(FakeTensor(..., size=(64, 32, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., size=(64,), requires_grad=True)), (1, 1), (1, 1), (1, 1), 1), **{}):
Given groups=1, weight of size [64, 32, 3, 3], expected input[1, 16, 32, 32] to have 32 channels, but got 16 channels instead

from user code:
   File "<string>", line 22, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
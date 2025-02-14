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
        self.conv = torch.nn.Conv2d(1, 256, 3, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(1, 256, 3, stride=2, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = torch.nn.Sigmoid()(self.conv1(v6))
        return torch.nn.ReLU()(v7)


func = Model().to('cpu')


x = torch.randn(1, 1, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [256, 1, 3, 3], expected input[1, 256, 32, 32] to have 1 channels, but got 256 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7fae4365f1c0>(*(FakeTensor(..., size=(1, 256, 32, 32)), Parameter(FakeTensor(..., size=(256, 1, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., size=(256,), requires_grad=True)), (2, 2), (1, 1), (1, 1), 1), **{}):
Given groups=1, weight of size [256, 1, 3, 3], expected input[1, 256, 32, 32] to have 1 channels, but got 256 channels instead

from user code:
   File "<string>", line 27, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
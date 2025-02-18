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
        self.conv = torch.nn.Conv2d(1, 1, 2)
        self.bn = torch.nn.BatchNorm2d(1)
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        x1 = self.bn(x)
        x2 = self.linear(x1)
        return x2

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3)
        self.conv1 = torch.nn.Conv2d(4, 7, 2, stride=1)
        self.conv2 = torch.nn.Conv2d(8, 6, 2, stride=2, bias=False)
        self.conv3 = torch.nn.Conv1d(2, 3, 1, bias=False)
        self.conv4 = torch.nn.Conv3d(3, 5, 3)

    def forward(self, x):
        x1 = self.conv4(self.conv3(self.conv2(self.conv1(self.conv(x)))))
        return x1



func = Model().to('cpu')


x = torch.randn(1, 1, 4, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [8, 4, 3, 3], expected input[1, 1, 4, 4] to have 4 channels, but got 1 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7fd98045f1c0>(*(FakeTensor(..., size=(1, 1, 4, 4)), Parameter(FakeTensor(..., size=(8, 4, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., size=(8,), requires_grad=True)), (1, 1), (0, 0), (1, 1), 1), **{}):
Given groups=1, weight of size [8, 4, 3, 3], expected input[1, 1, 4, 4] to have 4 channels, but got 1 channels instead

from user code:
   File "<string>", line 37, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
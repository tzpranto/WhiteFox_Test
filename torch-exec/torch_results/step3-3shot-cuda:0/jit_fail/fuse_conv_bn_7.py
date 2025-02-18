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
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)

    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.bn1(x1)
        x2 = self.conv(x1)
        x2 = self.bn2(x2)
        x3 = self.conv(x2)
        y1 = torch.cat([x1, x2, x3], 1)
        return y1



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 4, 4)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Calculated padded input size per channel: (2 x 2). Kernel size: (3 x 3). Kernel size can't be greater than actual input size

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f7fb105f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 2, 2)), Parameter(FakeTensor(..., device='cuda:0', size=(3, 3, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(3,), requires_grad=True)), (1, 1), (0, 0), (1, 1), 1), **{}):
Calculated padded input size per channel: (2 x 2). Kernel size: (3 x 3). Kernel size can't be greater than actual input size

from user code:
   File "<string>", line 24, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
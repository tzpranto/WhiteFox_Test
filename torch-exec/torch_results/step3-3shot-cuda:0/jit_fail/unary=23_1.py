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
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 5, 6, stride=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(5, 2, 3, stride=2)
        self.conv = torch.nn.Conv2d(2, 6, 5, stride=1)

    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv(v1)
        v3 = self.conv_transpose_2(v2)
        return v3



func = Model().to('cuda:0')


x1 = torch.randn(2, 3, 32, 32)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [6, 2, 5, 5], expected input[2, 5, 68, 68] to have 2 channels, but got 5 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f6a94a5f1c0>(*(FakeTensor(..., device='cuda:0', size=(2, 5, 68, 68)), Parameter(FakeTensor(..., device='cuda:0', size=(6, 2, 5, 5), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(6,), requires_grad=True)), (1, 1), (0, 0), (1, 1), 1), **{}):
Given groups=1, weight of size [6, 2, 5, 5], expected input[2, 5, 68, 68] to have 2 channels, but got 5 channels instead

from user code:
   File "<string>", line 23, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
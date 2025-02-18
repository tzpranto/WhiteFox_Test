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
        self.convT = torch.nn.ConvTranspose2d(1, 10, 4, padding=1)
        self.conv = torch.nn.Conv2d(10, 1, kernel_size=1)
        self.conv1 = torch.nn.Conv2d(4, 7, kernel_size=(2, 3), padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(7, 8, kernel_size=(4, 3), padding=1)

    def forward(self, x1):
        v2 = torch.relu(self.convT(x1))
        v3 = torch.sigmoid(self.conv(v2))
        v1 = torch.relu(self.conv1(x1))
        v2 = self.conv2(v1)
        v2 = torch.sigmoid(v2)
        result = v2 * v3
        return result



func = Model().to('cpu')


x1 = torch.randn(1, 1, 16, 16)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [7, 4, 2, 3], expected input[1, 1, 16, 16] to have 4 channels, but got 1 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f15fca5f1c0>(*(FakeTensor(..., size=(1, 1, 16, 16)), Parameter(FakeTensor(..., size=(7, 4, 2, 3), requires_grad=True)), Parameter(FakeTensor(..., size=(7,), requires_grad=True)), (1, 1), (1, 1), (1, 1), 1), **{}):
Given groups=1, weight of size [7, 4, 2, 3], expected input[1, 1, 16, 16] to have 4 channels, but got 1 channels instead

from user code:
   File "<string>", line 25, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, 1, stride=1)
        self.conv = torch.nn.Conv2d(4, 5, (3, 5), stride=2)

    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv(v2)
        v4 = torch.tanh(v3)
        return v4



func = Model().to('cuda:0')


x1 = torch.randn(1, 4, 4, 4)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Calculated padded input size per channel: (4 x 4). Kernel size: (3 x 5). Kernel size can't be greater than actual input size

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f972ca5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 4, 4, 4)), Parameter(FakeTensor(..., device='cuda:0', size=(5, 4, 3, 5), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(5,), requires_grad=True)), (2, 2), (0, 0), (1, 1), 1), **{}):
Calculated padded input size per channel: (4 x 4). Kernel size: (3 x 5). Kernel size can't be greater than actual input size

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
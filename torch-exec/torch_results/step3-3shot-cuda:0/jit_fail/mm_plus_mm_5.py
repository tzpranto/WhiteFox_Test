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
        super(Model, self).__init__()
        self.conv2d = torch.nn.Conv2d(100, 284, 16, stride=16)

    def forward(self, input1):
        t1 = self.conv2d(input1)
        return t1



func = Model().to('cuda:0')


input1 = torch.randn(1, 100, 7, 7)

test_inputs = [input1]

# JIT_FAIL
'''
direct:
Calculated padded input size per channel: (7 x 7). Kernel size: (16 x 16). Kernel size can't be greater than actual input size

jit:
Failed running call_function <built-in method conv2d of type object at 0x7efd99a5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 100, 7, 7)), Parameter(FakeTensor(..., device='cuda:0', size=(284, 100, 16, 16), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(284,), requires_grad=True)), (16, 16), (0, 0), (1, 1), 1), **{}):
Calculated padded input size per channel: (7 x 7). Kernel size: (16 x 16). Kernel size can't be greater than actual input size

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
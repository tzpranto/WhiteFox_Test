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

class Example(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Conv2d(3, 10, 5), torch.nn.BatchNorm2d(10), torch.nn.MaxPool2d(2), torch.nn.ReLU(), torch.nn.Conv2d(10, 2, 5))

    def forward(self, x):
        x = self.layers(x)
        return x



func = Example().to('cuda:0')


x = torch.randn(1, 3, 10, 10)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Calculated padded input size per channel: (3 x 3). Kernel size: (5 x 5). Kernel size can't be greater than actual input size

jit:
Failed running call_function <built-in method conv2d of type object at 0x7eff4b05f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 10, 3, 3)), Parameter(FakeTensor(..., device='cuda:0', size=(2, 10, 5, 5), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(2,), requires_grad=True)), (1, 1), (0, 0), (1, 1), 1), **{}):
Calculated padded input size per channel: (3 x 3). Kernel size: (5 x 5). Kernel size can't be greater than actual input size

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/container.py", line 250, in forward
    input = module(input)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
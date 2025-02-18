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
        self.layers = torch.nn.Sequential(torch.nn.Conv2d(37, 68, (17, 17), stride=1, padding=2), torch.nn.ReLU(), torch.nn.Conv2d(68, 56, (17, 17), stride=2, padding=9, dilation=2), torch.nn.ReLU(), torch.nn.ConvTranspose2d(56, 53, 20, 29, 10, output_padding=2), torch.nn.ReLU(), torch.nn.Conv2d(53, 48, (7, 7), stride=3, padding=2), torch.nn.ReLU(), torch.nn.Conv2d(48, 3, (17, 17), stride=1, padding=1))

    def forward(self, x1):
        v1 = self.layers(x1)
        return nn.Sigmoid()(v1)



func = Model().to('cuda:0')


x1 = torch.randn(1, 37, 1, 1)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Calculated padded input size per channel: (5 x 5). Kernel size: (17 x 17). Kernel size can't be greater than actual input size

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f325b65f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 37, 1, 1)), Parameter(FakeTensor(..., device='cuda:0', size=(68, 37, 17, 17), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(68,), requires_grad=True)), (1, 1), (2, 2), (1, 1), 1), **{}):
Calculated padded input size per channel: (5 x 5). Kernel size: (17 x 17). Kernel size can't be greater than actual input size

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
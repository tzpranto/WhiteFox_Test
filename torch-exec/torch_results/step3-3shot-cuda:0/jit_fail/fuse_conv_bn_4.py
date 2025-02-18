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

    def forward(self, x1):
        x2 = torch.cat([x1, x1, x1, x1], 1)
        conv1 = torch.nn.Conv2d(4, 4, 1)
        x2 = conv1(x2)
        bn_module = torch.nn.BatchNorm2d(4)
        bn_module.running_mean = torch.ones_like(bn_module.bias, dtype=torch.float32)
        bn_module.running_var = torch.ones_like(bn_module.bias, dtype=torch.float32)
        x2 = bn_module(x2)
        conv2 = torch.nn.Conv2d(4, 1, 1)
        x2 = conv2(x2)
        return x2



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 4, 4)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [4, 4, 1, 1], expected input[1, 12, 4, 4] to have 4 channels, but got 12 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f7fb105f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 12, 4, 4)), Parameter(FakeTensor(..., size=(4, 4, 1, 1), requires_grad=True)), Parameter(FakeTensor(..., size=(4,), requires_grad=True)), (1, 1), (0, 0), (1, 1), 1), **{}):
Given groups=1, weight of size [4, 4, 1, 1], expected input[1, 12, 4, 4] to have 4 channels, but got 12 channels instead

from user code:
   File "<string>", line 21, in torch_dynamo_resume_in_forward_at_20
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
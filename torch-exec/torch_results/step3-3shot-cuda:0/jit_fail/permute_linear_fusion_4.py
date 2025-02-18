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
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(3, 3, 1)

    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.conv(v2)



func = Model().to('cuda:0')


x1 = torch.randn(1, 2, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [3, 3, 1, 1], expected input[1, 1, 2, 2] to have 3 channels, but got 1 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f77d845f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 2, 2)), Parameter(FakeTensor(..., device='cuda:0', size=(3, 3, 1, 1), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(3,), requires_grad=True)), (1, 1), (0, 0), (1, 1), 1), **{}):
Given groups=1, weight of size [3, 3, 1, 1], expected input[1, 1, 2, 2] to have 3 channels, but got 1 channels instead

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
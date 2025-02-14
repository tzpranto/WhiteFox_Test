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
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(32, 16)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.linear(v1.view(100, 32))
        v3 = v2 * 0.5
        v4 = v2 * 0.044715
        v5 = v4 * v2
        v6 = v2 * 0.7978845608028654
        v7 = torch.tanh(v6 + v5)
        v8 = v3 * v7
        v9 = v8 + 1
        return v9


func = Model().to('cpu')


x = torch.randn(1, 32, 16, 16)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [8, 3, 3, 3], expected input[1, 32, 16, 16] to have 3 channels, but got 32 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f2be6e5f1c0>(*(FakeTensor(..., size=(1, 32, 16, 16)), Parameter(FakeTensor(..., size=(8, 3, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., size=(8,), requires_grad=True)), (1, 1), (1, 1), (1, 1), 1), **{}):
Given groups=1, weight of size [8, 3, 3, 3], expected input[1, 32, 16, 16] to have 3 channels, but got 32 channels instead

from user code:
   File "<string>", line 21, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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
        self.pool = torch.nn.MaxPool2d(1, 1, 1)
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)

    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = self.conv(v1)
        v3 = F.relu(v2)
        return v3



func = Model().to('cpu')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
pad should be at most half of effective kernel size, but got pad=1, kernel_size=1 and dilation=1

jit:
Failed running call_function <function boolean_dispatch.<locals>.fn at 0x7fb362735ca0>(*(FakeTensor(..., size=(1, 3, 64, 64)), 1, 1, 1, 1), **{'ceil_mode': False, 'return_indices': False}):
pad should be at most half of effective kernel size, but got pad=1, kernel_size=1 and dilation=1

from user code:
   File "<string>", line 21, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/pooling.py", line 213, in forward
    return F.max_pool2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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
        self.conv0 = torch.nn.Conv2d(3, 8, 5, stride=4, padding=1)
        self.conv1 = torch.nn.Conv2d(3, int(85 / 7), 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(int(85 / 7), 85, 3, stride=1, padding=1)

    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv1(x1)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.conv2(v6)
        v14 = v13 * 0.5
        v15 = v13 * 0.7071067811865476
        v16 = torch.erf(v15)
        v17 = v16 + 1
        v18 = v14 * v17
        return v18



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [85, 12, 3, 3], expected input[1, 8, 16, 16] to have 12 channels, but got 8 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f3c80e5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 16, 16)), Parameter(FakeTensor(..., device='cuda:0', size=(85, 12, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(85,), requires_grad=True)), (1, 1), (1, 1), (1, 1), 1), **{}):
Given groups=1, weight of size [85, 12, 3, 3], expected input[1, 8, 16, 16] to have 12 channels, but got 8 channels instead

from user code:
   File "<string>", line 34, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
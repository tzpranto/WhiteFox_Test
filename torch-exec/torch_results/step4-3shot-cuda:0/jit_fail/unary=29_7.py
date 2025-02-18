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

    def __init__(self, min_value=1.165433630886078, max_value=9.495014390253015):
        super().__init__()
        self.linear = torch.nn.Linear(12, 8)
        self.max_value = max_value
        self.min_value = min_value
        self.relu = torch.nn.ReLU()
        self.t = torch.nn.Conv3d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.relu(v1)
        v3 = self.t(v2)
        v4 = torch.clamp(v3, self.min_value, self.max_value)
        v5 = self.relu(v1)
        return v4



func = Model().to('cuda:0')


x1 = torch.randn(1, 12)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Expected 4D (unbatched) or 5D (batched) input to conv3d, but got input of size: [1, 8]

jit:
Failed running call_function <built-in method conv3d of type object at 0x7f0a7745f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 8)), Parameter(FakeTensor(..., device='cuda:0', size=(8, 3, 1, 1, 1), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(8,), requires_grad=True)), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1), **{}):
Expected 4D (unbatched) or 5D (batched) input to conv3d, but got input of size: [1, 8]

from user code:
   File "<string>", line 26, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 725, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 720, in _conv_forward
    return F.conv3d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
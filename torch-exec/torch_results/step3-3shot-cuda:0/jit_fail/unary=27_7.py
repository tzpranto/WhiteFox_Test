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

    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 2, stride=1, padding=1)
        self.min = min
        self.max = max

    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = torch.max(v3, x2)
        return v4



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

x2 = torch.randn(1, 3, 64, 64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
The size of tensor a (65) must match the size of tensor b (64) at non-singleton dimension 3

jit:
Failed running call_function <built-in method max of type object at 0x7f251ae5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 65, 65)), FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64))), **{}):
Attempting to broadcast a dimension of length 64 at -1! Mismatching argument at index 1 had torch.Size([1, 3, 64, 64]); but expected shape should be broadcastable to [1, 8, 65, 65]

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
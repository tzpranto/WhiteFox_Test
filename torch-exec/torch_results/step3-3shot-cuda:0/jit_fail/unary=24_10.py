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
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 > 0
        v3 = torch.max(v2, torch.full([1], 0.01))
        v4 = torch.where(v2, v1, v3)
        return v4



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

jit:
Failed running call_function <built-in method max of type object at 0x7ff8e7e5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 66, 66), dtype=torch.bool), FakeTensor(..., size=(1,))), **{}):
Unhandled FakeTensor Device Propagation for aten.maximum.default, found two different devices cuda:0, cpu

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
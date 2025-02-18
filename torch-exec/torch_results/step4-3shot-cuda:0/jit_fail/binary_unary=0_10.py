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

class Model1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3

class Model2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1.transpose(0, -1) + x2.transpose(0, -1)
        v3 = torch.relu(v2).transpose(0, -1)
        return v3



func = Model2().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

x2 = torch.randn(1, 3, 64, 64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
The size of tensor a (66) must match the size of tensor b (64) at non-singleton dimension 2

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(66, 8, 66, 1)), FakeTensor(..., device='cuda:0', size=(64, 3, 64, 1))), **{}):
Attempting to broadcast a dimension of length 64 at -2! Mismatching argument at index 1 had torch.Size([64, 3, 64, 1]); but expected shape should be broadcastable to [66, 8, 66, 1]

from user code:
   File "<string>", line 33, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
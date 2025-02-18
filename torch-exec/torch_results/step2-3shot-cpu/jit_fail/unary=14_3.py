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
        self.deconv1 = torch.nn.ConvTranspose2d(1, 1, 20, stride=10, padding=5)

    def forward(self, x1):
        v1 = self.deconv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

class Model2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.deconv2 = torch.nn.ConvTranspose2d(1, 1, 20, stride=10, padding=9)

    def forward(self, x1):
        v1 = self.deconv2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.m1 = Model1()
        self.m2 = Model2()

    def forward(self, x1):
        v1 = self.m1(x1)
        v2 = self.m2(x1)
        v3 = v1 + v2
        return v3



func = Model().to('cpu')


x1 = torch.randn(1, 1, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
The size of tensor a (640) must match the size of tensor b (632) at non-singleton dimension 3

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(1, 1, 640, 640)), FakeTensor(..., size=(1, 1, 632, 632))), **{}):
Attempting to broadcast a dimension of length 632 at -1! Mismatching argument at index 1 had torch.Size([1, 1, 632, 632]); but expected shape should be broadcastable to [1, 1, 640, 640]

from user code:
   File "<string>", line 47, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
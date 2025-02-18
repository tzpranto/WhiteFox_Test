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
        self.TConv1 = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=(1, 1))
        self.TConv2 = torch.nn.ConvTranspose2d(6, 8, 5, stride=1, padding=(2, 2), dilation=1)

    def forward(self, x1):
        v1 = self.TConv1(x1)
        v2 = self.TConv2(torch.cat((v1, x1), 1))
        t2 = torch.relu(v2)
        return t2



func = Model().to('cpu')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Sizes of tensors must match except in dimension 1. Expected size 127 but got size 64 for tensor number 1 in the list.

jit:
Failed running call_function <built-in method cat of type object at 0x7fb4be25f1c0>(*((FakeTensor(..., size=(1, 8, 127, 127)), FakeTensor(..., size=(1, 3, 64, 64))), 1), **{}):
Sizes of tensors must match except in dimension 1. Expected 127 but got 64 for tensor number 1 in the list

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
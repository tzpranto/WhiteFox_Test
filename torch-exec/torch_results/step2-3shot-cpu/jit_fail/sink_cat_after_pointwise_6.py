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

    def forward(self, x1, x2):
        v1 = torch.split(x1, 2, dim=1)
        v2 = torch.split(x2, 2, dim=1)
        v3 = torch.cat(v1 + v2, dim=1)
        v4 = torch.cat([x2, x2], dim=1)
        v5 = v3 + v4
        v6 = torch.cat([v5, v4], dim=1)
        v7 = torch.cat([v3, v6], dim=1)
        v8 = torch.relu(v7)
        v9 = torch.reshape(v8, -1)
        return v9



func = Model().to('cpu')


x1 = torch.randn(2, 4)

x2 = torch.randn(1, 16)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Sizes of tensors must match except in dimension 1. Expected size 2 but got size 1 for tensor number 2 in the list.

jit:
Failed running call_function <built-in method cat of type object at 0x7f180d25f1c0>(*((FakeTensor(..., size=(2, 2)), FakeTensor(..., size=(2, 2)), FakeTensor(..., size=(1, 2)), FakeTensor(..., size=(1, 2)), FakeTensor(..., size=(1, 2)), FakeTensor(..., size=(1, 2)), FakeTensor(..., size=(1, 2)), FakeTensor(..., size=(1, 2)), FakeTensor(..., size=(1, 2)), FakeTensor(..., size=(1, 2))),), **{'dim': 1}):
Sizes of tensors must match except in dimension 1. Expected 2 but got 1 for tensor number 2 in the list

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
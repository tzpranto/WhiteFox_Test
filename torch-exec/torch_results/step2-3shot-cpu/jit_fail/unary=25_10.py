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

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.randn(3 * 3 * 8).reshape(3, 3, 8), bias=None)
        v2 = v1 > 0
        v3 = v1 * self.alpha
        v4 = torch.where(v2, v1, v3)
        return v4


func = Model().to('cpu')


x1 = torch.randn(1, 8, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
t() expects a tensor with <= 2 dimensions, but self is 3D

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 8, 64, 64)), FakeTensor(..., size=(3, 3, 8))), **{'bias': None}):
t() expects a tensor with <= 2 dimensions, but self is 3D

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
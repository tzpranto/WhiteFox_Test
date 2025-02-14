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

class _MM(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mm(a, b)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mm1 = _MM()
        self.mm2 = _MM()

    def forward(self, x, y):
        v1 = self.mm1.forward(x, y)
        v2 = self.mm2.forward(x, y)
        v3 = v1 + v2
        return v3


func = Model().to('cpu')


x = torch.randn(1, 3, 2, 4)

y = torch.randn(1, 8, 4, 1)

test_inputs = [x, y]

# JIT_FAIL
'''
direct:
self must be a matrix

jit:
Failed running call_function <built-in method mm of type object at 0x7fce11a5f1c0>(*(FakeTensor(..., size=(1, 3, 2, 4)), FakeTensor(..., size=(1, 8, 4, 1))), **{}):
a must be 2D

from user code:
   File "<string>", line 29, in forward
  File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
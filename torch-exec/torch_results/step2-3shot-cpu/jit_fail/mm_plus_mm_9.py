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

    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x4)
        v3 = v1 + v2
        return v3



func = Model().to('cpu')


x1 = torch.randn(5, 4, 3)

x2 = torch.randn(4, 3, 2)

x3 = torch.randn(5, 4, 3)

x4 = torch.randn(4, 3, 2)

test_inputs = [x1, x2, x3, x4]

# JIT_FAIL
'''
direct:
self must be a matrix

jit:
Failed running call_function <built-in method mm of type object at 0x7f5f4505f1c0>(*(FakeTensor(..., size=(5, 4, 3)), FakeTensor(..., size=(4, 3, 2))), **{}):
a must be 2D

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
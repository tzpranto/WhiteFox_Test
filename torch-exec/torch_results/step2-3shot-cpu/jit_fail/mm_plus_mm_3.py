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
        x5 = torch.rand(1, 4)
        x6 = torch.rand(1, 2)
        v3 = v1 + v2
        v4 = torch.mm(x5, x6)
        v5 = v3 + v4
        r1 = torch.abs(v2 + v5)
        r2 = torch.max(v1 + v4 + r1, r1 + r1 * r1, r1 * r1 - v3)
        v6 = v1 / v6
        s1 = torch.nn.functional.sigmoid(0.2 * s2 - 3.4 * s1)
        s2 = torch.nn.functional.sigmoid(s2 - s1) * s1 * torch.nn.functional.sigmoid(0.3 * s1 + 1.4 * s2)
        return v6



func = Model().to('cpu')


x1 = torch.randn(3, 3)

x2 = torch.randn(3, 3)

x3 = torch.randn(4, 4)

x4 = torch.randn(3, 3)

test_inputs = [x1, x2, x3, x4]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (4x4 and 3x3)

jit:
Failed running call_function <built-in method mm of type object at 0x7f5f4505f1c0>(*(FakeTensor(..., size=(4, 4)), FakeTensor(..., size=(3, 3))), **{}):
a and b must have same reduction dim, but got [4, 4] X [3, 3].

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
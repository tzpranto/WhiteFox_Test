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

    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x4)
        v3 = torch.mm(v1, v2)
        v4 = v3 + x5
        return v4



func = Model().to('cpu')


x1 = torch.randn(5, 35)

x2 = torch.randn(35, 30)

x3 = torch.randn(5, 4)

x4 = torch.randn(4, 35)

x5 = torch.randn(5)

test_inputs = [x1, x2, x3, x4, x5]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (5x30 and 5x35)

jit:
Failed running call_function <built-in method mm of type object at 0x7f5f4505f1c0>(*(FakeTensor(..., size=(5, 30)), FakeTensor(..., size=(5, 35))), **{}):
a and b must have same reduction dim, but got [5, 30] X [5, 35].

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
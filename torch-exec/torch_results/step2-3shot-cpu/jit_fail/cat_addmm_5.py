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

    def forward(self, x1, x2, x3):
        v1 = torch.addmm(x1, x2, x3)
        return torch.cat([v1], dim=1)


func = Model().to('cpu')


x1 = torch.randn(3, 2, 4)

x2 = torch.randn(2, 4, 5)

x3 = torch.randn(3, 4, 5)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
mat1 must be a matrix, got 3-D tensor

jit:
Failed running call_function <built-in method addmm of type object at 0x7fd8c685f1c0>(*(FakeTensor(..., size=(3, 2, 4)), FakeTensor(..., size=(2, 4, 5)), FakeTensor(..., size=(3, 4, 5))), **{}):
a must be 2D

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
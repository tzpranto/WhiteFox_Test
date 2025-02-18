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

    def forward(self, x1):
        v1 = torch.addmm(x1, x1, x1)
        v2 = v1.unsqueeze(0)
        v3 = torch.cat([v1, v2, v1])
        return v3


func = Model().to('cpu')


x1 = torch.randn(8, 4)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (8x4 and 8x4)

jit:
Failed running call_function <built-in method addmm of type object at 0x7fd8c685f1c0>(*(FakeTensor(..., size=(8, 4)), FakeTensor(..., size=(8, 4)), FakeTensor(..., size=(8, 4))), **{}):
a and b must have same reduction dim, but got [8, 4] X [8, 4].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
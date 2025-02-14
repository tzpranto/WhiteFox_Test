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

    def __init__(self, other=0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(3, 3))
        self.other1 = other

    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.w)
        v2 = v1 - self.other1
        return torch.nn.functional.relu(v2)


func = Model(0.1).to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (192x64 and 3x3)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 3, 64, 64)), Parameter(FakeTensor(..., size=(3, 3), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [192, 64] X [3, 3].

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
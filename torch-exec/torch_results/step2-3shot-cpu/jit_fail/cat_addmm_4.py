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

    def forward(self, input, mat1, mat2, dim=1):
        v1 = torch.addmm(input, mat1, mat2)
        v2 = torch.cat([v1], dim)
        return v2


func = Model().to('cpu')


input = torch.randn(2, 2, 4096, 1024)

mat1 = torch.randn(2, 1024, 512)

mat2 = torch.randn(2, 1024, 512)

test_inputs = [input, mat1, mat2]

# JIT_FAIL
'''
direct:
mat1 must be a matrix, got 3-D tensor

jit:
Failed running call_function <built-in method addmm of type object at 0x7fd8c685f1c0>(*(FakeTensor(..., size=(2, 2, 4096, 1024)), FakeTensor(..., size=(2, 1024, 512)), FakeTensor(..., size=(2, 1024, 512))), **{}):
a must be 2D

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
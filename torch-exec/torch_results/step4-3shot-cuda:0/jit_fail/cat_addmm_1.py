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

    def forward(self, input, mat1, mat2):
        v1 = torch.addmm(input, mat1, mat2)
        t2 = torch.cat([v1], dim)
        return t2


func = Model().to('cuda:0')


input = torch.randn(1, 64, 64)

mat1 = torch.randn(1, 64, 32)

mat2 = torch.randn(1, 64, 32)

test_inputs = [input, mat1, mat2]

# JIT_FAIL
'''
direct:
mat1 must be a matrix, got 3-D tensor

jit:
Failed running call_function <built-in method addmm of type object at 0x7fd0dd25f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 64, 64)), FakeTensor(..., device='cuda:0', size=(1, 64, 32)), FakeTensor(..., device='cuda:0', size=(1, 64, 32))), **{}):
a must be 2D

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
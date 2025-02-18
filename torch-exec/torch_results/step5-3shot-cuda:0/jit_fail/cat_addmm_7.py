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

    def forward(self, x1, m1, m2):
        v1 = torch.addmm(x1, m1, m2)
        v2 = torch.cat([v1], dim)
        return v2


func = Model().to('cuda:0')


x1 = torch.randn(1, 64)

m1 = torch.randn(8, 64)

m2 = torch.randn(8, 64)

test_inputs = [x1, m1, m2]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (8x64 and 8x64)

jit:
Failed running call_function <built-in method addmm of type object at 0x7fad2425f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 64)), FakeTensor(..., device='cuda:0', size=(8, 64)), FakeTensor(..., device='cuda:0', size=(8, 64))), **{}):
a and b must have same reduction dim, but got [8, 64] X [8, 64].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
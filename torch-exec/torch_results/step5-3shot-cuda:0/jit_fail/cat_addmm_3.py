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
        t1 = torch.addmm(x1, x2, x3)
        t2 = torch.cat(t1, dim)
        return t2


func = Model().to('cuda:0')


x1 = torch.tensor(10.0)

x2 = torch.randn(1, 3, 64, 64)

x3 = torch.randn(1, 3, 64, 64)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
mat1 must be a matrix, got 4-D tensor

jit:
Failed running call_function <built-in method addmm of type object at 0x7fad2425f1c0>(*(FakeTensor(..., device='cuda:0', size=()), FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)), FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64))), **{}):
a must be 2D

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
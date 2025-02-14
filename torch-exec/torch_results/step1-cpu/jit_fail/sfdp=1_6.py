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
        self.m1 = torch.nn.Linear(16, 32)
        self.m2 = torch.nn.Linear(32, 64)

    def forward(self, m1_x, m2_x):
        v1 = self.m1(m1_x)
        v2 = self.m2(v1)
        v3 = torch.matmul(m2_x, v2.t())
        v4 = v3 / (math.sqrt(v2.shape[-1]) or 1)
        v5 = torch.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=0.1)
        v7 = torch.matmul(v6, v2)
        return v7


func = Model().to('cpu')


m1_x = torch.randn(1, 16)

m2_x = torch.randn(1, 64, 32)

test_inputs = [m1_x, m2_x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (64x32 and 64x1)

jit:
Failed running call_function <built-in method matmul of type object at 0x7feaa285f1c0>(*(FakeTensor(..., size=(1, 64, 32)), FakeTensor(..., size=(64, 1))), **{}):
a and b must have same reduction dim, but got [64, 32] X [64, 1].

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
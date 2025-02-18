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

    def __init__(self, N, D):
        super().__init__()
        self.N = N
        self.d1 = torch.nn.Linear(D, D)
        self.d2 = torch.nn.Linear(D, D)
        self.d3 = torch.nn.Linear(D, D)

    def forward(self, x1, x2, x3):
        v1 = self.d1(x1)
        v2 = torch.matmul(x2, v1.transpose(-2, -1))
        v3 = v2 * self.N ** (-0.5)
        v4 = nn.functional.softmax(v3, dim=-1)
        v5 = torch.matmul(v4, x3)
        v6 = self.d2(v5)
        v7 = self.d3(v6)
        return v7


N = 2
D = 512
func = Model(N, D).to('cuda:0')


x1 = torch.randn(32, 512)

D = 512
N = 2
x2 = torch.randn(N, D, 32)

D = 512
N = 2
x3 = torch.randn(N, D, 32)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1024x32 and 512x32)

jit:
Failed running call_function <built-in method matmul of type object at 0x7f96f665f1c0>(*(FakeTensor(..., device='cuda:0', size=(2, 512, 32)), FakeTensor(..., device='cuda:0', size=(512, 32))), **{}):
a and b must have same reduction dim, but got [1024, 32] X [512, 32].

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
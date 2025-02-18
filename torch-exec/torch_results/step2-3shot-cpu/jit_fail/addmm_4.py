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

class Model1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(12, 6))
        self.w2 = torch.nn.Parameter(torch.randn(6, 6))

    def forward(self, x2, inp):
        v1 = torch.mm(x2, self.w1)
        v2 = v1 + inp
        return torch.mm(v2, self.w2)



func = Model1().to('cpu')


x2 = torch.randn(12, 6)

inp = torch.randn(6, 6)

test_inputs = [x2, inp]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (12x6 and 12x6)

jit:
Failed running call_function <built-in method mm of type object at 0x7f421bc5f1c0>(*(FakeTensor(..., size=(12, 6)), Parameter(FakeTensor(..., size=(12, 6), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [12, 6] X [12, 6].

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
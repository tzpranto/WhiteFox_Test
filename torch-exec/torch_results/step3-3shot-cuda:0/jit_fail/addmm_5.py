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

    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, x2)
        v2 = v1 + x1
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(6, 12)

x2 = torch.randn(6, 6)

inp = torch.randn(6, 12)

test_inputs = [x1, x2, inp]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (6x12 and 6x6)

jit:
Failed running call_function <built-in method mm of type object at 0x7f889a85f1c0>(*(FakeTensor(..., device='cuda:0', size=(6, 12)), FakeTensor(..., device='cuda:0', size=(6, 6))), **{}):
a and b must have same reduction dim, but got [6, 12] X [6, 6].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
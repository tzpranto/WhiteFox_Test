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

    def forward(self, x1, x2, inp1, inp2):
        v1 = torch.mm(x1, x2) + inp1
        v2 = v1 + inp2
        return v2



func = Model().to('cpu')


x1 = torch.randn(1, 1, 6, 12)

x2 = torch.randn(1, 1, 12, 6)

inp1 = torch.randn(6, 6)

inp2 = torch.randn(6, 6)

test_inputs = [x1, x2, inp1, inp2]

# JIT_FAIL
'''
direct:
self must be a matrix

jit:
Failed running call_function <built-in method mm of type object at 0x7f421bc5f1c0>(*(FakeTensor(..., size=(1, 1, 6, 12)), FakeTensor(..., size=(1, 1, 12, 6))), **{}):
a must be 2D

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
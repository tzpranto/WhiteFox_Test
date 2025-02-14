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
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x, inp):
        v1 = self.conv(x)
        v2 = torch.add(torch.mm(v1, v1), inp)
        return v2


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

inp = torch.randn(1, 16, 8, 8)

test_inputs = [x, inp]

# JIT_FAIL
'''
direct:
self must be a matrix

jit:
Failed running call_function <built-in method mm of type object at 0x7fe5fba5f1c0>(*(FakeTensor(..., size=(1, 8, 64, 64)), FakeTensor(..., size=(1, 8, 64, 64))), **{}):
a must be 2D

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
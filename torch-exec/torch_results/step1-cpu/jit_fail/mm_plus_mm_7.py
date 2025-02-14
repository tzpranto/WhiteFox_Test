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

    def __init__(self, arg):
        super().__init__()
        self.arg = arg

    def forward(self, x):
        v1 = self.arg
        v2 = torch.mm(x, self.arg)
        v3 = torch.mm(self.arg, v2)
        return v3 + v2


arg = 1
func = Model(torch.randn(128, 64)).to('cpu')


x = torch.randn(1, 64, 1)

test_inputs = [x]

# JIT_FAIL
'''
direct:
self must be a matrix

jit:
Failed running call_function <built-in method mm of type object at 0x7fce11a5f1c0>(*(FakeTensor(..., size=(1, 64, 1)), FakeTensor(..., size=(128, 64))), **{}):
a must be 2D

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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
        self.fc1 = nn.Linear(2, 4, bias=True)
        self.cnd = nn.Linear(1, 1, bias=True)
        self.cnd.weight.data.fill_(0.25)
        self.cnd.bias.data.fill_(0.0)

    def forward(self, x1, x2):
        a1 = self.fc1(x1)
        b1 = self.fc1(x2)
        a2 = torch.mm(a1, b1)
        a3 = self.cnd(a2)
        a4 = a3 + a3
        a4 = a4 + a3
        return a4



func = Model().to('cpu')


x1 = torch.FloatTensor([[1, 1], [1, 1]])

x2 = torch.FloatTensor([[1, 1], [0, 0]])

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (2x4 and 2x4)

jit:
Failed running call_function <built-in method mm of type object at 0x7f89b925f1c0>(*(FakeTensor(..., size=(2, 4)), FakeTensor(..., size=(2, 4))), **{}):
a and b must have same reduction dim, but got [2, 4] X [2, 4].

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
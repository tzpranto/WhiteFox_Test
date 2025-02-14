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
        self.linear1 = torch.nn.Linear(3, 8)
        self.linear2 = torch.nn.Linear(8, 8)

    def forward(self, x, inp):
        v1 = self.linear1(x)
        v2 = self.linear2(v1)
        v3 = torch.mm(v2, v2)
        return inp + v3


func = Model().to('cpu')


x = torch.randn(1, 3)
inp = 1

test_inputs = [x, inp]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x8 and 1x8)

jit:
Failed running call_function <built-in method mm of type object at 0x7fe5fba5f1c0>(*(FakeTensor(..., size=(1, 8)), FakeTensor(..., size=(1, 8))), **{}):
a and b must have same reduction dim, but got [1, 8] X [1, 8].

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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
        self.fc = torch.nn.Linear(100, 100)

    def forward(self, x, inp=1):
        v1 = self.fc(x)
        v2 = torch.mm(v1, v1)
        return v2 + inp


func = Model().to('cpu')


x = torch.randn(1, 100)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x100 and 1x100)

jit:
Failed running call_function <built-in method mm of type object at 0x7fe5fba5f1c0>(*(FakeTensor(..., size=(1, 100)), FakeTensor(..., size=(1, 100))), **{}):
a and b must have same reduction dim, but got [1, 100] X [1, 100].

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
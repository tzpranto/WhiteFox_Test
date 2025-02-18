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
        self.fc = torch.nn.Linear(2, 3)

    def forward(self, x1):
        m = torch.empty(x1.shape[0], 2, 3, dtype=torch.float32, device=x1.device)
        x = self.fc(x1)
        torch.addmm(m, x, x)
        x = torch.cat([x, m], dim=2)
        return x


func = Model().to('cpu')


x1 = torch.randn(1, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x3 and 1x3)

jit:
Failed running call_function <built-in method addmm of type object at 0x7fd8c685f1c0>(*(FakeTensor(..., size=(1, 2, 3)), FakeTensor(..., size=(1, 3)), FakeTensor(..., size=(1, 3))), **{}):
a and b must have same reduction dim, but got [1, 3] X [1, 3].

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
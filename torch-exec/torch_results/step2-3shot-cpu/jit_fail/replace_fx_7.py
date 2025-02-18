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

class SubModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(2, 2))

    def forward(self, input):
        return input + self.x

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = SubModule()
        self.dropout1 = torch.nn.Dropout(0.8)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x, y):
        v1 = x * y
        v2 = self.l1(x)
        v3 = v1 + v2
        v4 = torch.matmul(v1, v2)
        v5 = self.dropout1(v3)
        v6 = v4 + v5
        v7 = self.dropout2(v6)
        return v7.sum()



func = Model().to('cpu')


x = torch.randn(10)

y = torch.randn(10)

test_inputs = [x, y]

# JIT_FAIL
'''
direct:
The size of tensor a (10) must match the size of tensor b (2) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(10,)), Parameter(FakeTensor(..., size=(2, 2), requires_grad=True))), **{}):
Attempting to broadcast a dimension of length 2 at -1! Mismatching argument at index 1 had torch.Size([2, 2]); but expected shape should be broadcastable to [1, 10]

from user code:
   File "<string>", line 32, in forward
  File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, x, y):
        v1 = x * 1
        v2 = self.linear(v1)
        v3 = v2 + y
        return v3


func = Model().to('cpu')


x = torch.randn(16, 16)

y = torch.randn(8, 16)

test_inputs = [x, y]

# JIT_FAIL
'''
direct:
The size of tensor a (8) must match the size of tensor b (16) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(16, 8)), FakeTensor(..., size=(8, 16))), **{}):
Attempting to broadcast a dimension of length 16 at -1! Mismatching argument at index 1 had torch.Size([8, 16]); but expected shape should be broadcastable to [16, 8]

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
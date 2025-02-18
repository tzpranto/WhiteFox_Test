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
        self.linear1 = torch.nn.Linear(10, 2)
        self.linear2 = torch.nn.Linear(2, 5)

    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = torch.relu(v2) - 1.5
        v4 = torch.relu(v3) - 2.0
        v5 = v4 * 6.5
        v6 = v1 / torch.sigmoid(v5)
        v7 = v6 % x2
        v8 = v1 * v6
        return v8


func = Model().to('cpu')


x1 = torch.randn(1, 10)

x2 = torch.randn(1, 5)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 1

jit:
Failed running call_function <built-in function truediv>(*(FakeTensor(..., size=(1, 2)), FakeTensor(..., size=(1, 5))), **{}):
Attempting to broadcast a dimension of length 5 at -1! Mismatching argument at index 1 had torch.Size([1, 5]); but expected shape should be broadcastable to [1, 2]

from user code:
   File "<string>", line 26, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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
        self.linear = torch.nn.Linear(64, 32)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.add(v1, x)
        return v2


func = Model().to('cpu')


x = torch.randn(1, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
The size of tensor a (32) must match the size of tensor b (64) at non-singleton dimension 1

jit:
Failed running call_function <built-in method add of type object at 0x7f3e3e05f1c0>(*(FakeTensor(..., size=(1, 32)), FakeTensor(..., size=(1, 64))), **{}):
Attempting to broadcast a dimension of length 64 at -1! Mismatching argument at index 1 had torch.Size([1, 64]); but expected shape should be broadcastable to [1, 32]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
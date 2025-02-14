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
        self.conv = torch.nn.Conv2d(6, 16, 3, stride=1, padding=1)

    def forward(self, x, other):
        v1 = self.conv(x)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3


func = Model().to('cpu')


x = torch.randn(1, 6, 64, 64)

other = torch.randn(1, 16, 32, 32)

test_inputs = [x, other]

# JIT_FAIL
'''
direct:
The size of tensor a (64) must match the size of tensor b (32) at non-singleton dimension 3

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(1, 16, 64, 64)), FakeTensor(..., size=(1, 16, 32, 32))), **{}):
Attempting to broadcast a dimension of length 32 at -1! Mismatching argument at index 1 had torch.Size([1, 16, 32, 32]); but expected shape should be broadcastable to [1, 16, 64, 64]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
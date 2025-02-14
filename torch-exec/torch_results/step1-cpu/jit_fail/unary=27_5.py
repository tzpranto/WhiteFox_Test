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

    def __init__(self, min_value=-1.0, max_value=+1.0):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        v1 = torch.clamp(x, min=self.min_value)
        v2 = torch.clamp(self.conv(x), min=self.min_value)
        return torch.clamp(v1 + v2, max=self.max_value)


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
The size of tensor a (3) must match the size of tensor b (8) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(1, 3, 64, 64)), FakeTensor(..., size=(1, 8, 64, 64))), **{}):
Attempting to broadcast a dimension of length 8 at -3! Mismatching argument at index 1 had torch.Size([1, 8, 64, 64]); but expected shape should be broadcastable to [1, 3, 64, 64]

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
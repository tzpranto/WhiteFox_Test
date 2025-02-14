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
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.fc = torch.nn.Linear(256, 32)

    def forward(self, x, other):
        v1 = self.conv(x)
        other1 = self.fc(other)
        v2 = v1 - other1
        v3 = F.relu(v2)
        return v3


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

other = torch.randn(1, 256)

test_inputs = [x, other]

# JIT_FAIL
'''
direct:
The size of tensor a (64) must match the size of tensor b (32) at non-singleton dimension 3

jit:
Failed running call_function <built-in function sub>(*(FakeTensor(..., size=(1, 8, 64, 64)), FakeTensor(..., size=(1, 32))), **{}):
Attempting to broadcast a dimension of length 32 at -1! Mismatching argument at index 1 had torch.Size([1, 32]); but expected shape should be broadcastable to [1, 8, 64, 64]

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
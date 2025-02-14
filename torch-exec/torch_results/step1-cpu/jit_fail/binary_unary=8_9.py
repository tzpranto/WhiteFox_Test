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
        self.bias = torch.nn.Parameter(data=torch.zeros(8, 64, 64), requires_grad=True)

    def forward(self, x):
        x1 = x + self.bias
        x2 = self.conv(x1)
        x3 = torch.relu(x2, 0.1)
        return x3


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
The size of tensor a (3) must match the size of tensor b (8) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(1, 3, 64, 64)), Parameter(FakeTensor(..., size=(8, 64, 64), requires_grad=True))), **{}):
Attempting to broadcast a dimension of length 8 at -3! Mismatching argument at index 1 had torch.Size([8, 64, 64]); but expected shape should be broadcastable to [1, 3, 64, 64]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
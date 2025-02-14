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

    def forward(self, x, bias=None):
        v1 = self.conv(x)
        v2 = v1 + bias if bias is not None else v1
        v3 = F.relu(v2)
        return v3


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

bias = torch.zeros(8)

test_inputs = [x, bias]

# JIT_FAIL
'''
direct:
The size of tensor a (64) must match the size of tensor b (8) at non-singleton dimension 3

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(1, 8, 64, 64)), FakeTensor(..., size=(8,))), **{}):
Attempting to broadcast a dimension of length 8 at -1! Mismatching argument at index 1 had torch.Size([8]); but expected shape should be broadcastable to [1, 8, 64, 64]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
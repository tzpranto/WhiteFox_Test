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

    def forward(self, x):
        v1 = self.conv(x)
        return v1 - other


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
The size of tensor a (8) must match the size of tensor b (3) at non-singleton dimension 1

jit:
Failed running call_function <built-in function sub>(*(FakeTensor(..., size=(1, 8, 64, 64)), FakeTensor(..., size=(1, 3, 64, 64))), **{}):
Attempting to broadcast a dimension of length 3 at -3! Mismatching argument at index 1 had torch.Size([1, 3, 64, 64]); but expected shape should be broadcastable to [1, 8, 64, 64]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
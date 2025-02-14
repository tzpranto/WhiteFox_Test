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

    def __init__(self, negative_slope):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
        self.neg_slope = negative_slope

    def forward(self, x):
        v1 = self.convt(x)
        v2 = torch.gt(x, 0)
        v3 = torch.where(v2, v1, v1 * self.neg_slope)
        return v3


negative_slope = 1
func = Model(0.01).to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
The size of tensor a (3) must match the size of tensor b (8) at non-singleton dimension 1

jit:
Failed running call_function <built-in method where of type object at 0x7f213f25f1c0>(*(FakeTensor(..., size=(1, 3, 64, 64), dtype=torch.bool), FakeTensor(..., size=(1, 8, 64, 64)), FakeTensor(..., size=(1, 8, 64, 64))), **{}):
Attempting to broadcast a dimension of length 8 at -3! Mismatching argument at index 1 had torch.Size([1, 8, 64, 64]); but expected shape should be broadcastable to [1, 3, 64, 64]

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
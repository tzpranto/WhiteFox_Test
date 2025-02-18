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
        self.negative_slope = negative_slope

    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.rand(12, 3), bias=None)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(torch.nn.functional.relu(x1) == 0, v1, v3)
        return v4


negative_slope = 1

func = Model(negative_slope).to('cpu')


x1 = torch.randn(1, 12, 3)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
The size of tensor a (3) must match the size of tensor b (12) at non-singleton dimension 2

jit:
Failed running call_function <built-in method where of type object at 0x7f821d25f1c0>(*(FakeTensor(..., size=(1, 12, 3), dtype=torch.bool), FakeTensor(..., size=(1, 12, 12)), FakeTensor(..., size=(1, 12, 12))), **{}):
Attempting to broadcast a dimension of length 12 at -1! Mismatching argument at index 1 had torch.Size([1, 12, 12]); but expected shape should be broadcastable to [1, 12, 3]

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x1):
        v1 = self.linear(x1)
        x2 = torch.rand(2, 5)
        v2 = v1 + x2
        return v2


linear = torch.nn.Linear(5, 10)
func = Model(linear).to('cuda:0')


x1 = torch.randn(1, 5)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
The size of tensor a (10) must match the size of tensor b (5) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 10)), FakeTensor(..., size=(2, 5))), **{}):
Attempting to broadcast a dimension of length 5 at -1! Mismatching argument at index 1 had torch.Size([2, 5]); but expected shape should be broadcastable to [1, 10]

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
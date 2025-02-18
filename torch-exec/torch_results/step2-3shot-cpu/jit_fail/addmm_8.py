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
        self.input1 = torch.rand(1, 2)

    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1 + self.input1 + inp
        return v2



func = Model().to('cpu')


x1 = torch.randn(6, 12)

x2 = torch.randn(12, 6)

inp = torch.randn(6, 6)

test_inputs = [x1, x2, inp]

# JIT_FAIL
'''
direct:
The size of tensor a (6) must match the size of tensor b (2) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(6, 6)), FakeTensor(..., size=(1, 2))), **{}):
Attempting to broadcast a dimension of length 2 at -1! Mismatching argument at index 1 had torch.Size([1, 2]); but expected shape should be broadcastable to [6, 6]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
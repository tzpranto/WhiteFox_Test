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

    def forward(self, x1, x2, inp):
        v0 = x1 - x2
        inp = inp - 2 * inp
        v1 = torch.addmm(inp, x1, x2)
        v2 = v1 + inp
        return v2



func = Model().to('cpu')


x1 = torch.randn(6, 12)

x2 = torch.randn(12, 6)

inp = torch.randn(6, 6)

test_inputs = [x1, x2, inp]

# JIT_FAIL
'''
direct:
The size of tensor a (12) must match the size of tensor b (6) at non-singleton dimension 1

jit:
Failed running call_function <built-in function sub>(*(FakeTensor(..., size=(6, 12)), FakeTensor(..., size=(12, 6))), **{}):
Attempting to broadcast a dimension of length 6 at -1! Mismatching argument at index 1 had torch.Size([12, 6]); but expected shape should be broadcastable to [6, 12]

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
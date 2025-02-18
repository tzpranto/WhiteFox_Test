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
        v1 = torch.mm(inp, x1)
        v2 = v1 * x2
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1262, 666)

x2 = torch.randn(1262, 321)

inp = torch.randn(321, 1262)

test_inputs = [x1, x2, inp]

# JIT_FAIL
'''
direct:
The size of tensor a (666) must match the size of tensor b (321) at non-singleton dimension 1

jit:
Failed running call_function <built-in function mul>(*(FakeTensor(..., device='cuda:0', size=(321, 666)), FakeTensor(..., device='cuda:0', size=(1262, 321))), **{}):
Attempting to broadcast a dimension of length 321 at -1! Mismatching argument at index 1 had torch.Size([1262, 321]); but expected shape should be broadcastable to [321, 666]

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
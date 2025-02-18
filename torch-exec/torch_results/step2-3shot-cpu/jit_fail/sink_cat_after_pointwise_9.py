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

    def forward(self, x1, x2):
        v1 = torch.cat((x1, x2), dim=1)
        v3 = torch.cat((x1, x2), dim=1)
        v4 = torch.cat((v1, v3), dim=1)
        v5 = torch.cat((v4, v3), dim=1)
        v6 = torch.cat((v1, v5), dim=1)
        v2 = x1 + v6
        v7 = v2.view(-1)
        return v7



func = Model().to('cpu')


x1 = torch.randn(1, 2)

x2 = torch.randn(1, 3)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
The size of tensor a (2) must match the size of tensor b (20) at non-singleton dimension 1

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., size=(1, 2)), FakeTensor(..., size=(1, 20))), **{}):
Attempting to broadcast a dimension of length 20 at -1! Mismatching argument at index 1 had torch.Size([1, 20]); but expected shape should be broadcastable to [1, 2]

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
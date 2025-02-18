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
        v1 = torch.cat((x1, x2), dim=2)
        v3 = torch.cat((x2, x2), dim=1)
        v4 = torch.cat((v1, v3), dim=1)
        v5 = torch.cat((v4, v3), dim=1)
        v6 = torch.cat((v1, v5), dim=1)
        v2 = torch.relu(v6)
        v7 = v2.view(-1)
        return v7



func = Model().to('cpu')


x1 = torch.randn(1, 2, 3)

x2 = torch.randn(1, 3)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Tensors must have same number of dimensions: got 3 and 2

jit:
Failed running call_function <built-in method cat of type object at 0x7f180d25f1c0>(*((FakeTensor(..., size=(1, 2, 3)), FakeTensor(..., size=(1, 3))),), **{'dim': 2}):
Number of dimensions of tensors must match.  Expected 3-D tensors, but got 2-D for tensor number 1 in the list

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
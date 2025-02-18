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
        self.conv = torch.nn.Conv2d(5, 5, 3, stride=2, padding=1)

    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = v1 + x2
        v3 = v2[:, :, :, :] + v2[:, :, :, :]
        v4 = torch.relu(v3)
        return v4[:, :, :, :]



func = Model().to('cuda:0')


x1 = torch.randn(1, 5, 64, 64)

x2 = torch.randn(2, 5, 19, 19)

x3 = torch.randn(2, 5, 1, 1)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
The size of tensor a (32) must match the size of tensor b (19) at non-singleton dimension 3

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 5, 32, 32)), FakeTensor(..., device='cuda:0', size=(2, 5, 19, 19))), **{}):
Attempting to broadcast a dimension of length 19 at -1! Mismatching argument at index 1 had torch.Size([2, 5, 19, 19]); but expected shape should be broadcastable to [1, 5, 32, 32]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
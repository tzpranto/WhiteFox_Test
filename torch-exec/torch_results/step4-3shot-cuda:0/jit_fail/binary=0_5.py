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
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.t1 = torch.randn(8, 8, 2, 2)

    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = self.t1
        v2 = v1 + other
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
The size of tensor a (66) must match the size of tensor b (2) at non-singleton dimension 3

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 66, 66)), FakeTensor(..., device='cuda:0', size=(8, 8, 2, 2))), **{}):
Attempting to broadcast a dimension of length 2 at -1! Mismatching argument at index 1 had torch.Size([8, 8, 2, 2]); but expected shape should be broadcastable to [1, 8, 66, 66]

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
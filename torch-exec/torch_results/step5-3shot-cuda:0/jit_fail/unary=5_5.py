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
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 4, stride=4, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return self.bn(x1) + self.bn(v6)



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 38, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
The size of tensor a (64) must match the size of tensor b (254) at non-singleton dimension 3

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 38, 64)), FakeTensor(..., device='cuda:0', size=(1, 3, 150, 254))), **{}):
Attempting to broadcast a dimension of length 254 at -1! Mismatching argument at index 1 had torch.Size([1, 3, 150, 254]); but expected shape should be broadcastable to [1, 3, 38, 64]

from user code:
   File "<string>", line 27, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
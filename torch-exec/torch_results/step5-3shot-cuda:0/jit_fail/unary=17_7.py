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
        self.conv_layer = torch.nn.Conv2d(3, 8, (15, 15))
        self.transpose_layer = torch.nn.ConvTranspose2d(3, 8, 15)

    def forward(self, x1):
        v1 = self.transpose_layer(x1)
        v2 = self.conv_layer(x1)
        v3 = torch.relu(v1)
        v4 = torch.relu(v2)
        return v3 + v4



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
The size of tensor a (78) must match the size of tensor b (50) at non-singleton dimension 3

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 78, 78)), FakeTensor(..., device='cuda:0', size=(1, 8, 50, 50))), **{}):
Attempting to broadcast a dimension of length 50 at -1! Mismatching argument at index 1 had torch.Size([1, 8, 50, 50]); but expected shape should be broadcastable to [1, 8, 78, 78]

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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
        self.conv_transpose1 = torch.nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.negative_slope = 0.01

    def forward(self, x1, x2):
        v1 = self.conv_transpose1(x1) + self.conv_transpose2(x2)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4



func = Model().to('cuda:0')


x1 = torch.randn(1, 8, 64, 64)

x2 = torch.randn(1, 3, 64, 64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
The size of tensor a (128) must match the size of tensor b (62) at non-singleton dimension 3

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 128, 128)), FakeTensor(..., device='cuda:0', size=(1, 8, 62, 62))), **{}):
Attempting to broadcast a dimension of length 62 at -1! Mismatching argument at index 1 had torch.Size([1, 8, 62, 62]); but expected shape should be broadcastable to [1, 3, 128, 128]

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
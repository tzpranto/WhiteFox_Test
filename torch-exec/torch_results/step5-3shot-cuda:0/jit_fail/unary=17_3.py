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
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1):
        v1 = F.conv_transpose2d(x1, weight=torch.randn(8, 3, 1, 1), bias=torch.randn(8), stride=1, padding=1, output_padding=1)
        v2 = F.relu(v1)
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [8, 3, 1, 1], expected input[1, 3, 64, 64] to have 8 channels, but got 3 channels instead

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7fe5f045f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)),), **{'weight': FakeTensor(..., size=(8, 3, 1, 1)), 'bias': FakeTensor(..., size=(8,)), 'stride': 1, 'padding': 1, 'output_padding': 1}):
Given transposed=1, weight of size [8, 3, 1, 1], expected input[1, 3, 64, 64] to have 8 channels, but got 3 channels instead

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
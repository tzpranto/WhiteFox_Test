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

class Model3(torch.nn.Module):

    def __init__(self):
        super().__init__()
        x: int = 8
        y: int = 8
        z: int = 3
        self.conv = torch.nn.ConvTranspose2d(z, x // 2, 2, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(y, y // 2, 3, stride=1, padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(y // 2, y, 1, stride=1, padding=0)
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = v5 * v6
        y = torch.cat((v7, x2, x3), -3)
        y = torch.abs(y)
        y = torch.sigmoid(y)
        y = self.linear(y)
        return y



func = Model3().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

x2 = torch.randn(1, 4, 16, 16)

x3 = torch.randn(1, 10, 32, 32)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [8, 4, 3, 3], expected input[1, 4, 126, 126] to have 8 channels, but got 4 channels instead

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7fd84ec5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 4, 126, 126)), Parameter(FakeTensor(..., device='cuda:0', size=(8, 4, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(4,), requires_grad=True)), (1, 1), (0, 0), (0, 0), 1, (1, 1)), **{}):
Given transposed=1, weight of size [8, 4, 3, 3], expected input[1, 4, 126, 126] to have 8 channels, but got 4 channels instead

from user code:
   File "<string>", line 29, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 1162, in forward
    return F.conv_transpose2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
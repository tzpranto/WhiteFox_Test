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

class Model2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        x: int = 8
        y: int = 8
        z: int = 3
        self.conv = torch.nn.ConvTranspose2d(x, y, 1, stride=1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(y + x, y, 1, stride=1, padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(z, x, 1, stride=1, padding=0)

    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = torch.cat((x1, x2), dim=1)
        v5 = self.conv2(v4)
        v6 = torch.sigmoid(v5)
        v7 = v4 + v6
        v8 = self.conv3(v7)
        return (v5, v6, v8)



func = Model2().to('cuda:0')


x1 = torch.randn(1, 8, 64, 64)

x2 = torch.randn(1, 3, 64, 64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [16, 8, 1, 1], expected input[1, 11, 64, 64] to have 16 channels, but got 11 channels instead

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7fd84ec5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 11, 64, 64)), Parameter(FakeTensor(..., device='cuda:0', size=(16, 8, 1, 1), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(8,), requires_grad=True)), (1, 1), (0, 0), (0, 0), 1, (1, 1)), **{}):
Given transposed=1, weight of size [16, 8, 1, 1], expected input[1, 11, 64, 64] to have 16 channels, but got 11 channels instead

from user code:
   File "<string>", line 29, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 1162, in forward
    return F.conv_transpose2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 16, 3, stride=3)
        self.linear = torch.nn.Linear(64, 32)

    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.linear(v6)
        return v7



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [8, 16, 3, 3], expected input[1, 3, 64, 64] to have 8 channels, but got 3 channels instead

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7f232525f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)), Parameter(FakeTensor(..., device='cuda:0', size=(8, 16, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(16,), requires_grad=True)), (3, 3), (0, 0), (0, 0), 1, (1, 1)), **{}):
Given transposed=1, weight of size [8, 16, 3, 3], expected input[1, 3, 64, 64] to have 8 channels, but got 3 channels instead

from user code:
   File "<string>", line 22, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 1162, in forward
    return F.conv_transpose2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
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
        self.conv_transpose = torch.nn.ConvTranspose2d(11, 10, 1, stride=1, padding=1, bias=False)
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(15, 5, 1, stride=1, padding=1, bias=False)

    def forward(self, x5):
        v1 = self.conv_transpose(x5)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv_transpose_0(torch.tanh(v9))
        return v10



func = Model().to('cuda:0')


x5 = torch.randn(1, 11, 9, 4)

test_inputs = [x5]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [15, 5, 1, 1], expected input[1, 10, 7, 2] to have 15 channels, but got 10 channels instead

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7f714ae5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 10, 7, 2)), Parameter(FakeTensor(..., device='cuda:0', size=(15, 5, 1, 1), requires_grad=True)), None, (1, 1), (1, 1), (0, 0), 1, (1, 1)), **{}):
Given transposed=1, weight of size [15, 5, 1, 1], expected input[1, 10, 7, 2] to have 15 channels, but got 10 channels instead

from user code:
   File "<string>", line 30, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 1162, in forward
    return F.conv_transpose2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
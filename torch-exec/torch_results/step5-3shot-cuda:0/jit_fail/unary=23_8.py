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
        self.conv_transpose1 = torch.nn.ConvTranspose2d(2, 3, 2, stride=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(10, 11, 3, stride=1)

    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = torch.tanh(v2)
        return v3



func = Model().to('cuda:0')


x1 = torch.randn(1, 2, 32, 32)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [10, 11, 3, 3], expected input[1, 3, 33, 33] to have 10 channels, but got 3 channels instead

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7f3a2765f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 33, 33)), Parameter(FakeTensor(..., device='cuda:0', size=(10, 11, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(11,), requires_grad=True)), (1, 1), (0, 0), (0, 0), 1, (1, 1)), **{}):
Given transposed=1, weight of size [10, 11, 3, 3], expected input[1, 3, 33, 33] to have 10 channels, but got 3 channels instead

from user code:
   File "<string>", line 22, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 1162, in forward
    return F.conv_transpose2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
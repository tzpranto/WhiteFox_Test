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
        self.depthwise_conv_transpose = torch.nn.ConvTranspose2d(32, 32, 10, groups=32, stride=2, padding=5)

    def forward(self, x1):
        v1 = self.depthwise_conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5



func = Model().to('cuda:0')


x1 = torch.randn(1, 4, 8, 8)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [32, 1, 10, 10], expected input[1, 4, 8, 8] to have 32 channels, but got 4 channels instead

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7f608f25f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 4, 8, 8)), Parameter(FakeTensor(..., device='cuda:0', size=(32, 1, 10, 10), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(32,), requires_grad=True)), (2, 2), (5, 5), (0, 0), 32, (1, 1)), **{}):
Given transposed=1, weight of size [32, 1, 10, 10], expected input[1, 4, 8, 8] to have 32 channels, but got 4 channels instead

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 1162, in forward
    return F.conv_transpose2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''
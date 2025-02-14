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
        self.conv_tr = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv_tr(x)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v2 * v3
        return v4 / 6


func = Model().to('cpu')


x = torch.randn(1, 1, 2, 2)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [3, 8, 3, 3], expected input[1, 1, 2, 2] to have 3 channels, but got 1 channels instead

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7fbb77a5f1c0>(*(FakeTensor(..., size=(1, 1, 2, 2)), Parameter(FakeTensor(..., size=(3, 8, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., size=(8,), requires_grad=True)), (1, 1), (1, 1), (0, 0), 1, (1, 1)), **{}):
Given transposed=1, weight of size [3, 8, 3, 3], expected input[1, 1, 2, 2] to have 3 channels, but got 1 channels instead

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 1162, in forward
    return F.conv_transpose2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''